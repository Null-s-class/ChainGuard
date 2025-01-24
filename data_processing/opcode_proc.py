from __future__ import absolute_import, division, print_function

import logging
import re
import torch
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

cpu_cont = 16
logger = logging.getLogger(__name__)

import os
import pickle
import torch
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# def process_opcode(opcode, max_length=512, path_save="Data/Dataset/processed_opcodes/"):
#     """
#     Workflow of Opcode Processing:
#     1. Prepare Save Directories
#     2. Check for Existing Processed Data
#        - If complete data exists, load and return
#        - If partial data exists, determine last processed state
#     3. Process Opcodes
#        - Calculate vocabulary size
#        - Create tokenizer
#        - Convert opcodes to sequences
#        - Pad sequences
#        - Create dictionary mapping indices to tensors
#     4. Save Processed Data
#        - Incrementally save progress
#        - Handle potential interruptions
    
#     Args:
#         opcode (pandas.DataFrame): DataFrame with 'opcode' and 'index' columns
#         max_length (int): Maximum sequence length for padding
#         path_save (str): Directory to save processed opcodes
    
#     Returns:
#         dict: Mapping of indices to opcode tensors
#     """
#     # Prepare save directory
#     SAVE_DIR = path_save
#     os.makedirs(SAVE_DIR, exist_ok=True)
    
#     # Paths for saving progress
#     processed_dict_path = os.path.join(SAVE_DIR, 'processed_opcodes.pkl')
#     progress_path = os.path.join(SAVE_DIR, 'progress.pkl')
    
#     # Try to load existing processed data
#     try:
#         if os.path.exists(processed_dict_path) and os.path.getsize(processed_dict_path) > 0:
#             logger.info("Loading existing processed opcodes...")
#             with open(processed_dict_path, 'rb') as f:
#                 return pickle.load(f)
#     except Exception as e:
#         logger.warning(f"Error loading existing processed opcodes: {e}")
    
#     # Calculate vocabulary and sequence properties
#     Vocab_of_size = opcode['opcode'].nunique()
#     length = opcode['opcode'].str.split().apply(len)
#     avg_length = int(length.mean())
    
#     logger.info(f'Average length {avg_length} --> Max Length {max_length}')
    
#     # Prepare tokenizer and sequences
#     try:
#         tokenizer = Tokenizer(num_words=Vocab_of_size)
#         tokenizer.fit_on_texts(opcode['opcode'])  # Build dictionary
#         sequences = tokenizer.texts_to_sequences(opcode['opcode'])  # Convert to integer sequences
#         opcode_matrix = pad_sequences(sequences, maxlen=max_length)  # Pad to uniform length
#         opcode_tensor = torch.tensor(opcode_matrix)  # Convert to tensor
        
#         # Create result dictionary
#         index_list = opcode['index'].tolist()
#         result_dict = {index_list[i]: opcode_tensor[i] for i in range(len(index_list))}
        
#         # Save processed data
#         try:
#             with open(processed_dict_path, 'wb') as f:
#                 pickle.dump(result_dict, f)
#             logger.info(f"Processed opcodes saved to {processed_dict_path}")
#         except Exception as e:
#             logger.error(f"Error saving processed opcodes: {e}")
        
#         return result_dict
    
#     except Exception as e:
#         logger.error(f"Error processing opcodes: {e}")
#         return {}

# Batch-processing version to dealing with large datasets
def process_opcode(opcode, max_length=512, batch_size=32, path_save="Data/Dataset/processed_opcodes/"):
    """
    Batch-processing version of process_opcode for large datasets
    Supports resumable processing and incremental saving
    """
    SAVE_DIR = path_save
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    processed_dict_path = os.path.join(SAVE_DIR, 'opcodes_processed.pkl')
    progress_path = os.path.join(SAVE_DIR, 'opcodes_progress.pkl')
    
    # Try to load existing processed data
    try:
        if os.path.exists(processed_dict_path) and os.path.getsize(processed_dict_path) > 0:
            logger.info("Loading existing processed opcodes...")
            with open(processed_dict_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        logger.warning(f"Error loading existing processed opcodes: {e}")
    
    # Prepare result dictionary
    result_dict = {}
    
    # Determine start batch
    start_batch = 0
    try:
        if os.path.exists(progress_path):
            with open(progress_path, 'rb') as f:
                start_batch = pickle.load(f)
    except:
        start_batch = 0
    
    # Calculate total batches
    num_samples = len(opcode)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Vocabulary calculation
    Vocab_of_size = opcode['opcode'].nunique()
    tokenizer = Tokenizer(num_words=Vocab_of_size)
    tokenizer.fit_on_texts(opcode['opcode'])
    
    # Process in batches
    for batch_idx in range(start_batch, num_batches):
        try:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            
            batch_opcode = opcode.iloc[start_idx:end_idx]
            
            # Process batch
            sequences = tokenizer.texts_to_sequences(batch_opcode['opcode'])
            opcode_matrix = pad_sequences(sequences, maxlen=max_length)
            opcode_tensor = torch.tensor(opcode_matrix)
            
            # Update result dictionary
            batch_indices = batch_opcode['index'].tolist()
            batch_result = {batch_indices[i]: opcode_tensor[i] for i in range(len(batch_indices))}
            result_dict.update(batch_result)
            
            # Save progress
            with open(processed_dict_path, 'wb') as f:
                pickle.dump(result_dict, f)
            with open(progress_path, 'wb') as f:
                pickle.dump(batch_idx + 1, f)
            
            logger.info(f"Processed batch {batch_idx + 1}/{num_batches}")
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
    
    # Clean up progress file
    if os.path.exists(progress_path):
        os.remove(progress_path)
    
    return result_dict