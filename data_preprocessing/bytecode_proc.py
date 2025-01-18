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

cpu_cont = 16
logger = logging.getLogger(__name__)

def preprocess_bytecode(bytecode, max_length = 512, batch_size = 32):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    modelBert = BertModel.from_pretrained('bert-base-cased')
    embedding = [] # danh sach embedding
    indices = [] # danh sach index tuong ung
    num_sample = len(bytecode)
    num_batches = (num_sample + batch_size -1) // batch_size
    logger.info(f'Number of batch size {num_batches}')

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_sample)
        batch_bytecodes = bytecode['bytecode'].iloc[start_idx:end_idx]
        batch_indices = bytecode['index'].iloc[start_idx:end_idx] 
        logger.info(f'number of sample in a batch {end_idx-start_idx}')
        tokenized_texts = tokenizer(batch_bytecodes.tolist(), max_length=max_length, truncation= True, padding= 'max_length', return_tensors= 'pt')
        with torch.no_grad():
            outputs = modelBert(**tokenized_texts)
        batch_embeddings = outputs.last_hidden_state.numpy()
        #logger.info(f'embeding shape {batch_embeddings.shape}\n indices {batch_indices}\n')
        embedding.append(batch_embeddings)
        indices.append(batch_indices)

    embedding = np.concatenate(embedding,axis =0)
    indices = np.concatenate(indices,axis = 0)
    return embedding, indices