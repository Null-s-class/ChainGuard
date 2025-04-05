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
def bytecode_to_opcodes(bytecode):
    return bytecode.split()

def preprocess_bytecode(bytecode_df, load_only=False, max_length=512, batch_size=32, max_opcodes=1000, SAVE_PATH = "Data/Dataset/processed_bytecodes/"):
    VERSION = '2'
    embeddings_path = os.path.join(SAVE_PATH, f"embedding_bytecode_{VERSION}.pkl")
    indices_path = os.path.join(SAVE_PATH, f"indices_bytecode_{VERSION}.pkl")
    # freq_path = os.path.join(SAVE_PATH, f"freq_bytecode_{VERSION}.pkl")
    # vocab_path = os.path.join(SAVE_PATH, f"vocab_bytecode_{VERSION}.pkl")

    if load_only:
        logger.info("Loading preprocessed data...")
        try:
            with open(embeddings_path, 'rb') as f:
                embedding = pickle.load(f)
            with open(indices_path, 'rb') as f:
                indices = pickle.load(f)
            # with open(freq_path, 'rb') as f:
            #     freq_vector = pickle.load(f)
            # with open(vocab_path, 'rb') as f:
            #     opcode_vocab = pickle.load(f)

            return embedding, indices
        except FileNotFoundError:
            logger.error(f'Preprocessed data not found in {SAVE_PATH}')
            

    logger.info("Processing bytecodes...")

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')

    # BERT embeddings
    num_samples = len(bytecode_df)
    num_batches = (num_samples + batch_size - 1) // batch_size
    logger.info(f'Number of batches for BERT processing: {num_batches}')

    embedding = []
    indices = []

    for batch_idx in tqdm(range(num_batches), desc="Processing bytecodes with BERT"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)

        batch_bytecodes = bytecode_df['bytecode'].iloc[start_idx:end_idx].tolist()
        batch_indices = bytecode_df['index'].iloc[start_idx:end_idx].tolist()

        tokenized_texts = tokenizer(
            batch_bytecodes, 
            max_length=max_length, 
            truncation=True, 
            padding='max_length', 
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = model(**tokenized_texts)
        
        batch_embeddings = outputs.last_hidden_state.cpu().numpy()

        embedding.append(batch_embeddings)
        indices.extend(batch_indices)

    # Combine batches for BERT embeddings
    embedding = np.concatenate(embedding, axis=0)
    indices = np.array(indices)

    # # Opcode frequency analysis
    # opcodes = []
    # for code in tqdm(bytecode_df['bytecode'], desc="Converting bytecode to opcodes"):
    #     opcodes.append(bytecode_to_opcodes(code))

    # all_opcodes = [op for sublist in opcodes for op in sublist]
    # opcode_vocab = sorted(set(all_opcodes))
    # vocab_size = len(opcode_vocab)

    # logger.info(f"Opcode vocabulary size: {vocab_size}")

    # opcode_to_idx = {op: idx for idx, op in enumerate(opcode_vocab)}
    # opcode_indices = []
    # for op_list in opcodes:
    #     # Pad or truncate to max_opcodes
    #     opcode_indices.append([opcode_to_idx[op] for op in op_list[:max_opcodes]] + 
    #                           [0] * (max_opcodes - min(max_opcodes, len(op_list))))

    # # Compute frequency of opcodes
    # freq_dict = Counter(all_opcodes)
    # freq_vector = [freq_dict.get(op, 0) for op in opcode_vocab]

    # Save results
    logger.info(f"Saving results to {SAVE_PATH}")
    os.makedirs(SAVE_PATH, exist_ok=True)

    with open(embeddings_path, 'wb') as f:
        pickle.dump(embedding, f)
    with open(indices_path, 'wb') as f:
        pickle.dump(indices, f)
    # with open(freq_path, 'wb') as f:
    #     pickle.dump(freq_vector, f)
    # with open(vocab_path, 'wb') as f:
    #     pickle.dump(opcode_vocab, f)

    logger.info("Preprocessing complete.")
    return embedding, indices