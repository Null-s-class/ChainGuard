from __future__ import absolute_import, division, print_function

import logging
import os
import pickle
import re
import torch
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


cpu_cont = 16
logger = logging.getLogger(__name__)

def clean_opcode(opcode_str):
    """
    Clean and preprocess opcode string
    
    Args:
        opcode_str (str): Raw opcode string to clean
    
    Returns:
        str: Cleaned opcode string
    """
    # Remove newline characters
    opcode_str = opcode_str.replace('\n', ' ')
    
    # Optional: Uncomment for additional cleaning
    # opcode_str = re.sub(r'0x[a-fA-F0-9]+', '', opcode_str)
    # codes = re.findall(r'[A-Z]+', opcode_str)
    
    return opcode_str

def process_opcode(opcode, max_length=512, batch_size=32, path_save="Data/Dataset/processed_opcodes/"):
    """
    Batch-processing version of process_opcode for large datasets
    Supports resumable processing and incremental saving
    
    Args:
        opcode (pandas.DataFrame): DataFrame with 'opcode' and 'index' columns
        max_length (int): Maximum sequence length for padding
        batch_size (int): Number of samples to process in each batch
        path_save (str): Directory to save processed opcodes
    
    Returns:
        dict: Mapping of indices to opcode tensors
    """
    # SAVE_DIR = path_save
    # os.makedirs(SAVE_DIR, exist_ok=True)
    
    # processed_dict_path = os.path.join(SAVE_DIR, 'opcodes_processed.pkl')
    # progress_path = os.path.join(SAVE_DIR, 'opcodes_progress.pkl')

    # logger.info('Attempting to load existing data from last run...')

    existing_result_dict = {}
    # # Try to load existing processed data
    # try:
    #     if os.path.exists(processed_dict_path) and os.path.getsize(processed_dict_path) > 0:
    #         with open(processed_dict_path, 'rb') as f:
    #             existing_result_dict = pickle.load(f)
    #         logger.info(f"Loaded existing processed opcodes: {len(existing_result_dict)} entries")
    # except Exception as e:
    #     logger.warning(f"Error loading existing processed opcodes: {e} continue from begining....")
    #     existing_result_dict = {}
    
    # Prepare result dictionary
    result_dict = existing_result_dict.copy()
    last_processed_batch = 0
    # # Determine start batch
    # try:
    #     last_processed_batch = 0
    #     if os.path.exists(progress_path):
    #         with open(progress_path, 'rb') as f:
    #             last_processed_batch = pickle.load(f)
    # except Exception as e:
    #     logger.warning(f"Error reading progress file: {e} continue from begining....")
    #     last_processed_batch = 0
    
    # Calculate total batches
    num_samples = len(opcode)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    logger.info(f"Calculating vocabulary...")
    # Vocabulary calculation (using available opcodes)
    try:
        unprocessed_opcodes = opcode[~opcode['index'].isin(result_dict.keys())]
        Vocab_of_size = unprocessed_opcodes['opcode'].nunique()
        
        # Use the full vocabulary including previously processed data
        tokenizer = Tokenizer(num_words=Vocab_of_size)
        tokenizer.fit_on_texts(unprocessed_opcodes['opcode'])
    except Exception as e:
        logger.error(f"Error calculating vocabulary: {e}")
        return result_dict
    
    logger.info(f'Total number of batches: {num_batches} processed {last_processed_batch}, batches left: {num_batches - last_processed_batch} ')

    # Process in batches
    for batch_idx in tqdm(range(last_processed_batch, num_batches), desc="Processing opcodes"):
        try:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            
            # Filter out already processed indices
            batch_opcode = opcode.iloc[start_idx:end_idx]
            batch_opcode = batch_opcode[~batch_opcode['index'].isin(result_dict.keys())]
            
            if batch_opcode.empty:
                logger.info(f"Batch {batch_idx} already processed. Skipping.")
                continue
            
            # Process batch
            sequences = tokenizer.texts_to_sequences(batch_opcode['opcode'])
            opcode_matrix = pad_sequences(sequences, maxlen=max_length)
            opcode_tensor = torch.tensor(opcode_matrix)
            
            # Update result dictionary
            batch_indices = batch_opcode['index'].tolist()
            batch_result = {batch_indices[i]: opcode_tensor[i] for i in range(len(batch_indices))}
            result_dict.update(batch_result)
            
            # # Save progress
            # with open(processed_dict_path, 'wb') as f:
            #     pickle.dump(result_dict, f)
            # with open(progress_path, 'wb') as f:
            #     pickle.dump(batch_idx + 1, f)
            
            logger.info(f"Processed batch {batch_idx + 1}/{num_batches}. Total processed: {len(result_dict)}")
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            # # Optionally, save progress even if a batch fails
            # with open(processed_dict_path, 'wb') as f:
            #     pickle.dump(result_dict, f)
    
    # # Clean up progress file if all batches completed successfully
    # if os.path.exists(progress_path):
    #     os.remove(progress_path)
    
    logger.info(f"Opcode processing complete. Total processed entries: {len(result_dict)}")
    return result_dict