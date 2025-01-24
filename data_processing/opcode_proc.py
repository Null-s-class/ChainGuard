from __future__ import absolute_import, division, print_function

import logging
import re
import torch
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

cpu_cont = 16
logger = logging.getLogger(__name__)

def clean_opcode(opcode_str):
    #opcode_str = re.sub(r'0x[a-fA-F0-9]+', '', opcode_str)
    opcode_str = opcode_str.replace('\n', ' ')
    #codes = re.findall(r'[A-Z]+', opcode_str)
    return opcode_str

def process_opcode(opcode, max_length= 512):
    Vocab_of_size = opcode['opcode'].nunique()

    length = opcode['opcode'].str.split().apply(len)
    avg_length = int(length.mean())

    logger.info(f'Average of length {avg_length} --> {max_length}')

    tokenizer = Tokenizer(num_words = Vocab_of_size)
    tokenizer.fit_on_texts(opcode['opcode']) #xay dung tu dien 
    sequences = tokenizer.texts_to_sequences(opcode['opcode']) #chuyen sequence ve mang interger
    opcode_matrix = pad_sequences(sequences,maxlen=max_length) #padding ve cung 1 size 
    opcode_tensor = torch.tensor(opcode_matrix) #chuyen ve tensor 
    index_list = opcode['index'].tolist()
    result_dict = {index_list[i]: opcode_tensor[i] for i in range(len(index_list))}
    return result_dict
