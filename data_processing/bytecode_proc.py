from __future__ import absolute_import, division, print_function

import os
import logging
import pickle
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

cpu_cont = 16
logger = logging.getLogger(__name__)

def preprocess_bytecode(bytecode, max_length=512, batch_size=32, path_save="Data/Dataset/embedding/bytecode/"):
    """
    Workflow of Bytecode Preprocessing with Resumable Processing:
    1. Prepare Save Directories
    2. Check for Existing Processed Data
       - If complete data exists, load and return
       - If partial data exists, determine last processed batch
    3. Process Bytecode in Batches
       - Skip already processed batches
       - Generate embeddings for remaining batches
       - Append new embeddings to existing data
    4. Save Processed Data
       - Incrementally save progress
       - Handle potential interruptions
    
    Args:
        bytecode (pandas.DataFrame): DataFrame with 'bytecode' and 'index' columns
        max_length (int): Maximum token length for BERT tokenization
        batch_size (int): Number of samples to process in each batch
        path_save (str): Directory to save processed embeddings
    
    Returns:
        tuple: (embeddings, indices) - Numpy arrays of processed data
    """

    # SAVE_DIR = path_save
    # os.makedirs(SAVE_DIR, exist_ok=True)
    
    # # Paths for saving progress
    # embeddings_path = os.path.join(SAVE_DIR, 'bytecode_embeddings.pkl')
    # indices_path = os.path.join(SAVE_DIR, 'bytecode_indices.pkl')
    # progress_path = os.path.join(SAVE_DIR, 'bytecode_progress.pkl')
    
    existing_embeddings = []
    existing_indices = []
    last_processed_batch = 0

    #logger.info(f'Attempting to load existing data from last run...')
    # # Initialize or load existing data
    # try:
    #     # Try to load existing embeddings and progress
    #     if os.path.exists(embeddings_path) and os.path.getsize(embeddings_path) > 0:
    #         with open(embeddings_path, 'rb') as f:
    #             existing_embeddings = pickle.load(f)
    #         with open(indices_path, 'rb') as f:
    #             existing_indices = pickle.load(f)
            
    #         # Ensure lists if loaded data is not already a list
    #         existing_embeddings = [existing_embeddings] if not isinstance(existing_embeddings, list) else existing_embeddings
    #         existing_indices = [existing_indices] if not isinstance(existing_indices, list) else existing_indices
            
    #         # Try to load progress
    #         last_processed_batch = 0
    #         if os.path.exists(progress_path):
    #             with open(progress_path, 'rb') as f:
    #                 last_processed_batch = pickle.load(f)
            
    #         logger.info(f"Resuming from batch {last_processed_batch}")
    #     else:
    #         existing_embeddings = []
    #         existing_indices = []
    #         last_processed_batch = 0
    # except Exception as e:
    #     logger.error(f"Error loading existing data: {e} continue from begining....")
    #     existing_embeddings = []
    #     existing_indices = []
    #     last_processed_batch = 0
    
    # Prepare processing
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    modelBert = BertModel.from_pretrained('bert-base-cased')
    
    num_sample = len(bytecode)
    num_batches = (num_sample + batch_size - 1) // batch_size

    logger.info(f'Total number of batches: {num_batches} processed {last_processed_batch}, batches left: {num_batches - last_processed_batch} ')
    
    # Process remaining batches
    for batch_idx in tqdm(range(last_processed_batch, num_batches), desc="Processing bytecode: " ):
        try:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_sample)
            
            # Extract current batch
            batch_bytecodes = bytecode['bytecode'].iloc[start_idx:end_idx]
            batch_indices = bytecode['index'].iloc[start_idx:end_idx]
            
            logger.info(f'Processing batch {batch_idx}: {end_idx - start_idx} samples')
            
            # Tokenize and embed
            tokenized_texts = tokenizer(
                batch_bytecodes.tolist(), 
                max_length=max_length, 
                truncation=True, 
                padding='max_length', 
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = modelBert(**tokenized_texts)
            
            batch_embeddings = outputs.last_hidden_state.numpy()
            
            # Append to existing data
            existing_embeddings.append(batch_embeddings)
            existing_indices.append(batch_indices.to_numpy())
            
            # Save progress after each batch
            full_embeddings = np.concatenate(existing_embeddings, axis=0)
            full_indices = np.concatenate(existing_indices, axis=0)
            
            # # Save incremental progress
            # with open(embeddings_path, 'wb') as f:
            #     pickle.dump(full_embeddings, f)
            # with open(indices_path, 'wb') as f:
            #     pickle.dump(full_indices, f)
            # with open(progress_path, 'wb') as f:
            #     pickle.dump(batch_idx + 1, f)
            
            logger.info(f"Progress saved for batch {batch_idx}")
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            continue
            # Optionally, you might want to break or continue based on your error handling strategy
    
    # Final concatenation
    embedding = np.concatenate(existing_embeddings, axis=0)
    indices = np.concatenate(existing_indices, axis=0)
    
    # # Clean up progress file if completed successfully
    # if os.path.exists(progress_path):
    #     os.remove(progress_path)
    
    logger.info(f"Preprocessing complete. Total embeddings: {len(embedding)}")
    return embedding, indices