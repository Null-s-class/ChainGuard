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

from torch.utils.data import Dataset

from tqdm import tqdm

from data_processing.bytecode_proc import preprocess_bytecode
from data_processing.opcode_proc import process_opcode
from data_processing.source_code_proc import extract_dataflow

from parser import DFG_javascript
from tree_sitter import Language, Parser

dfg_function={
    'javascript':DFG_javascript
}

#load parsers
#Create a dictionary for progamming language suit with data source code chua cac bo phan tich 
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
logger = logging.getLogger(__name__)

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



class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path="", DRY_RUN_MODE=False, DRY_RUN_DATA_SAMPLES=None):
        self.file_path = file_path
        self.examples = []
        self.args = args
        self.data_max_size = self.args.hidden_size  # 768
        index_filename = self.args.train_data_file

        # Load index files and data
        index_to_sourcecode = {}
        index_to_bytecode = {}
        index_to_opcode = {}

        logger.info("Loading dataset from jsonl")
        with open('Data/Dataset/data.jsonl') as f: 
            for line in f:
                line = line.strip()
                js = json.loads(line)
                index_to_sourcecode[js['idx']] = js['source']
                index_to_bytecode[js['idx']] = js['byte']
                index_to_opcode[js['idx']] = js['codeop']
        
        # Process bytecode and opcode
        df_bytecode = pd.DataFrame.from_dict(index_to_bytecode, orient='index', columns=['bytecode'])
        df_bytecode.index.name = 'index'
        df_bytecode.reset_index(inplace=True)
        
        df_opcode = pd.DataFrame.from_dict(index_to_opcode, orient='index', columns=['opcode'])
        df_opcode.index.name = 'index'
        df_opcode.reset_index(inplace=True)
        
        # Process opcode
        logger.info('Processing opcode')
        opcode_matrix = process_opcode(df_opcode, max_length=self.data_max_size)
        
        # Preprocess bytecode
        logger.info('Processing bytecode')
        bytecode_embedding, bytecode_index = preprocess_bytecode(df_bytecode, max_length=512)
        embedding_dict = {index: embedding for index, embedding in zip(bytecode_index, bytecode_embedding)}

        # Load code function according to index
        data_source = []
        cache = {}
        with open(index_filename) as f:
            for line in f:
                line = line.strip()
                url1, labels = line.split(' ', 1)
                labels = [int(label) for label in labels.split()]
                
                # Skip if any required data is missing
                if url1 not in index_to_sourcecode or url1 not in index_to_bytecode or url1 not in index_to_opcode:
                    continue
                
                embedding_code = embedding_dict[url1]
                opcode_tensor = opcode_matrix[url1]
                
                data_source.append((url1, labels, tokenizer, args, cache, 
                                    index_to_sourcecode, embedding_code, opcode_tensor))
        
        # Implement TEST mode to limit samples
        if DRY_RUN_MODE and DRY_RUN_DATA_SAMPLES is not None:
            data_source = random.sample(data_source, min(DRY_RUN_DATA_SAMPLES, len(data_source)))
        
        # Only use 10% of validation data
        if 'valid' in self.file_path:
            data_source = random.sample(data_source, int(len(data_source)*0.1))
        
        # Convert examples to features
        self.examples = [convert_examples_to_features(x) for x in tqdm(data_source, total=len(data_source))]

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

def save_dataset(dataset, filepath):
    """
    Save dataset to a pickle file using pickle
    
    Args:
        dataset (TextDataset): Dataset to save
        filepath (str): Path to save the pickle file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)

def load_dataset(filepath):
    """
    Load dataset from a pickle file
    
    Args:
        filepath (str): Path to the pickle file
    
    Returns:
        TextDataset: Loaded dataset
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_dataset_joblib(filepath):
    """
    Load dataset using joblib (alternative method)
    
    Args:
        filepath (str): Path to the joblib file
    
    Returns:
        TextDataset: Loaded dataset
    """
    return joblib.load(filepath)

# # Example usage
# def main():
#     # Assuming you have the necessary arguments and tokenizer
#     args = YourArgsClass()  # Replace with your actual args class
#     tokenizer = YourTokenizer()  # Replace with your actual tokenizer
    
#     # Create dataset with full data
#     full_dataset = TextDataset(tokenizer, args, file_path='train')
    
#     # Create dataset in TEST mode with limited samples
#     test_dataset = TextDataset(tokenizer, args, file_path='train', 
#                                DRY_RUN_MODE=True, DRY_RUN_SAMPLES=100)
    
#     # Save full dataset
#     save_dataset(full_dataset, 'full_dataset.pkl')
    
#     # Save test dataset
#     save_dataset(test_dataset, 'test_dataset.pkl')
    
#     # Load dataset
#     loaded_dataset = load_dataset('full_dataset.pkl')

# if __name__ == '__main__':
#     main()