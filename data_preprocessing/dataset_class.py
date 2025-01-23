from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import joblib
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel, BertModel, BertTokenizer)
from tqdm import tqdm, trange
import multiprocessing
from model import Model
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.nn import BCEWithLogitsLoss

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train'):
        self.examples = []
        self.args=args
        index_filename=file_path #chua danh sach index cua file tranning set valid set hoac test set
        #self.bytecode_embedding = bytecode_embeddings
        #load index
        logger.info("Creating features from index file at %s ", index_filename)
        index_to_sourcecode = {}
        index_to_bytecode = {}
        index_to_opcode = {}
        with open('Data/Dataset/data.jsonl') as f: 
            for line in f:
                line=line.strip()
                js=json.loads(line)
                index_to_sourcecode[js['idx']]=js['source'] # load sourcode theo index
                index_to_bytecode[js['idx']] = js['byte'] # load bytecode theo index
                index_to_opcode[js['idx']] = js['codeop'] # load opcode theo index
            
        #chuyen bytecode ve Dataframe   
        df_bytecode = pd.DataFrame.from_dict(index_to_bytecode, orient='index', columns=['bytecode'])
        df_bytecode.index.name = 'index'
        df_bytecode.reset_index(inplace=True)
        
        #chuyen opcode ve Dataframe
        df_opcode =pd.DataFrame.from_dict(index_to_opcode, orient='index', columns=['opcode'])
        df_opcode.index.name = 'index'
        df_opcode.reset_index(inplace=True)
        
        opcode_matrix = process_opcode(df_opcode)
        #print('Processed opcode: ',opcode_matrix,'\n')
    
        #logger.info('loaded Data')
        #logger.info('df_bytecode :n%s', df_bytecode)

        bytecode_embedding, bytecode_index = preprocess_bytecode(df_bytecode,max_length=20) #embedding bytecode
        #print('Processed bytecode')
        #logger.info('byte code embedding%s', bytecode_embedding.shape,'\n')
        #logger.info('bytecode index%s',bytecode_index,'\n')
        embedding_dict = {index : embedding for index, embedding in zip(bytecode_index, bytecode_embedding)}

        #logger.info('Bytecode embeding%s', embedding_dict)
        #load code function according to index
        data_source = []
        cache={}
        f=open(index_filename)
        with open(index_filename) as f:
            for line in f:
                line=line.strip()
                url1,labels =line.split(' ', 1) # la indx cua tung function
                labels = [int(label) for label in labels.split()] # convert labels tahnh mang [1,0,1,0]
                if url1 not in index_to_sourcecode or url1 not in index_to_bytecode or url1 not in index_to_opcode:
                    continue
                embedding_code = embedding_dict[url1]
                opcode_tensor = opcode_matrix[url1]
                if embedding_code is None:
                    logger.info('Key invalid')
                data_source.append((url1,labels,tokenizer, args,cache,index_to_sourcecode,embedding_code, opcode_tensor))
                
        #only use 10% valid data_source to keep best model        
        if 'valid' in file_path:
            data_source=random.sample(data_source,int(len(data_source)*0.1))
            
        #convert example to input features    
        self.examples=[convert_examples_to_features(x) for x in tqdm(data_source,total=len(data_source))]
        #self.examples la 1 mang Inputfeature cho tung sample
        #[<__main__.InputFeatures object at 0x73b9ea3a5810>, <__main__.InputFeatures object at 0x73b9ee9bd490>,....]

        #logger.info('seld examples %s', self.examples)


        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]): #moi line example chua cac thuoc tinh source_tokens_1,source_ids_1,position_idx_1
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens_1: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                logger.info("input_ids_1: {}".format(' '.join(map(str, example.input_ids))))       
                logger.info("position_idx_1: {}".format(example.position_idx))
                logger.info("dfg_to_code_1: {}".format(' '.join(map(str, example.dfg_to_code))))
                logger.info("dfg_to_dfg_1: {}".format(' '.join(map(str, example.dfg_to_dfg))))
                

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask_1= np.zeros((self.args.code_length+self.args.data_flow_length,
                        self.args.code_length+self.args.data_flow_length),dtype=bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask_1[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].input_ids):
            if i in [0,2]:
                attn_mask_1[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask_1[idx+node_index,a:b]=True
                attn_mask_1[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx):
                    attn_mask_1[idx+node_index,a+node_index]=True  
                    
        out_input_ids = torch.tensor(self.examples[item].input_ids)
        out_position_ids =  torch.tensor(self.examples[item].position_idx)
        out_attn_mask =   torch.tensor(attn_mask_1)
        out_labels = torch.tensor(self.examples[item].label)
        out_bytecode_embedding = torch.tensor(self.examples[item].bytecode_embedding) # la 1 tensor co shape la [20,768]
        out_opcode_tensor = self.examples[item].opcode_tensor
        # logger.info('out_input_ids',out_input_ids.shape)
        # logger.info('out_postion_ids',out_position_ids.shape)
        # logger.info('out_attn_mask',out_attn_mask.shape)
        # logger.info('out_labels',out_labels.shape)
        # logger.info(f'bytecode embedding {item} : {out_bytecode_embedding.shape}')
        # logger.info('Processed source code')
        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(attn_mask_1),    
                out_bytecode_embedding,
                out_opcode_tensor,         
                torch.tensor(self.examples[item].label))

def saved_extract_dataset(dataset, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(dataset,f)

def load_extract_dataset(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_dataset_from_pickle(file_path):
    dataset = joblib.load(file_path)
    return dataset


######### Dinh dang cua input cho model #########
class InputFeatures(object):
    """A single training/test features for an example."""
    def __init__(self, input_tokens, input_ids, position_idx, dfg_to_code,
                  dfg_to_dfg, bytecode_embedding, opcode_tensor, label, url):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.bytecode_embedding = bytecode_embedding
        self.opcode_tensor = opcode_tensor
        self.label = label
        self.url = url

def convert_examples_to_features(item):
    url, label, tokenizer, args, cache, url_to_sourcecode, bytecode_embedding, opcode_tensor = item
    #url index cua sourcode 
    #cache de luu lai cac ma nguon da trich xuat tranh trich xuat trung nhau 
    #url_to_sourcecoe ma nguon tuong ung voi index
    parser = parsers['javascript']
    
    if url not in cache:
        source = url_to_sourcecode[url] #trich xuat soucecode tuong ung voi url
        code_tokens, dfg = extract_dataflow(source, parser, 'javascript') # extract ra duoc danh sach code token va dfg co shape (x,)
        #logger.info(f'Code token cua sourcode {url}: {code_tokens}\n')
        #logger.info(f'dfg cua sourcode {url}: {dfg}\n')
        code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
        #logger.info(f'code token sau khi dc tokenize {code_tokens}\n')
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
        code_tokens = [y for x in code_tokens for y in x]
        
        code_tokens = code_tokens[:args.code_length + args.data_flow_length - 3][:512 - 3]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg = dfg[:args.code_length + args.data_flow_length - len(source_tokens)]
        source_tokens += [x[0] for x in dfg]
        position_idx += [0 for x in dfg]
        source_ids += [tokenizer.unk_token_id for x in dfg]
        padding_length = args.code_length + args.data_flow_length - len(source_ids)
        position_idx += [tokenizer.pad_token_id] * padding_length
        source_ids += [tokenizer.pad_token_id] * padding_length
        
        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
        length = len([tokenizer.cls_token])
        dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
        cache[url] = source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg

    source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg = cache[url]
    # logger.info('source_tokens',source_tokens)
    # logger.info('source_ids',source_ids)
    # logger.info('position_idx',position_idx)
    # logger.info('dfg_to_code',dfg_to_code)
    # logger.info('dfg_to_dfg',dfg_to_dfg)
    # logger.info('label',label)
    # logger.info('url',url)
    return InputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, bytecode_embedding, opcode_tensor, label, url)


