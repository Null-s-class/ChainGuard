from __future__ import absolute_import, division, print_function

import os
import logging
import pickle
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

cpu_cont = 16
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)

def preprocess_bytecode(bytecode, load_only = False, max_length=512, batch_size=32, path_save="Data/Dataset/embedding/bytecode/"):
    ver = '2'
    embeddings_path = path_save + f"embedding_bytecode_{ver}.pkl"
    indices_path = path_save + f"indices_bytecode_{ver}.pkl"
    embedding = [] # danh sach embedding
    indices = [] # danh sach index tuong ung

    if load_only == True:
        logger.info("load_only=True : Loading preprocessed embeddings and indices...")
        save_embedding_path = os.path.join( path_save , f"embedding_bytecode_2.pkl") # the ver should be change manually to prevent unwanted overwrite
        save_indices_path = os.path.join( path_save , f"indices_bytecode_2.pkl")
        if os.path.exists(save_embedding_path) and os.path.exists(save_indices_path):
            with open(save_embedding_path, 'rb') as f:
                embedding = pickle.load(f)
            with open(save_indices_path, 'rb') as f:
                indices = pickle.load(f)
            return embedding, indices
        else:
            logger.error(f'Preprocessed embeddings not found at {save_embedding_path} or {save_indices_path}')
            embedding = [] # danh sach embedding
            indices = [] # danh sach index tuong ung
    else:
        logger.info("load_only=False : Processing bytecodes...")


    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    modelBert = BertModel.from_pretrained('bert-base-cased')
  
    num_sample = len(bytecode)
    num_batches = (num_sample + batch_size -1) // batch_size

    logger.info(f'Number of batch size {num_batches}')

    for batch_idx in tqdm(range(num_batches), f"Processing bytecodes"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_sample)

        batch_bytecodes = bytecode['bytecode'].iloc[start_idx:end_idx]
        batch_indices = bytecode['index'].iloc[start_idx:end_idx] 

        
        tokenized_texts = tokenizer(batch_bytecodes.tolist(), max_length=max_length, truncation= True, padding= 'max_length', return_tensors= 'pt')
        with torch.no_grad():
            outputs = modelBert(**tokenized_texts)
        batch_embeddings = outputs.last_hidden_state.numpy()

        #logger.info(f'embeding shape {batch_embeddings.shape}\n indices {batch_indices}\n')
        embedding.append(batch_embeddings)
        indices.append(batch_indices)

    embedding = np.concatenate(embedding,axis =0)
    indices = np.concatenate(indices,axis = 0)

    # Save 
    logger.info(f'Saving embeddings to {embeddings_path}')
    logger.info(f'Saving indices to {indices_path}')
    logger.info(f'You now can using load_only=True param to load saved one without having to reprocess...')

    with open(embeddings_path, 'wb') as f:
        pickle.dump(embedding, f)
    with open(indices_path, 'wb') as f:
        pickle.dump(indices, f)

    logger.info(f"Preprocessing complete. Total embeddings: {len(embedding)}")
    return embedding, indices