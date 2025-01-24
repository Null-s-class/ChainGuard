from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import numpy as np
import torch

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from tqdm import tqdm
from model_edit_by_Null import Model
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from data_processing.dataset_class import TextDataset

cpu_cont = 16
DEF_SEED = 42

DEF_N_LABELS = 10 # 10 labels

DEF_OUTPUT_DIR = 'saved_models'
DEF_CONFIG_NAME = 'microsoft/graphcodebert-base'
DEF_MODEL_NAME_OR_PATH = 'microsoft/graphcodebert-base'
DEF_TOKENIZER_NAME = 'microsoft/graphcodebert-base'
DEF_DO_TRAIN = True
DEF_TRAIN_DATA_FILE = 'Data/Dataset/data.txt'
DEF_EVAL_DATA_FILE = 'Data/Dataset/Valid.txt'
DEF_TEST_DATA_FILE = 'Data/Dtaset/Test.txt'
DEF_EPOCH = 1
DEF_CODE_LENGTH = 512
DEF_DATA_FLOW_LENGTH = 128
DEF_TRAIN_BATCH_SIZE = 4
DEF_EVAL_BATCH_SIZE = 16
DEF_LEARNING_RATE = 2e-5
DEF_MAX_GRAD_NORM = 1.0

DRY_RUN_MODE = False
DRY_RUN_DATA = 0.1 # only used 10% of the original data to test 

DEF_MODEL_VER = '1_Null'
DEF_MODEL_CHECKPOINT_DIR= 'checkpoint-best-f1'
os.makedirs(DEF_MODEL_CHECKPOINT_DIR, exist_ok=True)



logger = logging.getLogger(__name__)

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def train(args, model, tokenizer):
    """
    Model Training Function with Optimized Components
    """
    # Data Preparation
    args.hidden_size = model.config.hidden_size
    print(f"MODEL HIDDEN: {args.hidden_size}\n")
    train_dataset = TextDataset(tokenizer, args, args.train_data_file, DRY_RUN_MODE = DRY_RUN_MODE, DRY_RUN_DATA = DRY_RUN_DATA)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=args.train_batch_size, 
        num_workers=16
    )
    
    # Optimization Configuration
    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 5
    args.warmup_steps = args.max_steps // 5
    
    # Optimizer Configuration
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate*0.1, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )
    
    # Multi-GPU Setup
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        print(f'Using {args.n_gpu} GPUs')
    else:
        model = model.to(args.device)

    # Logging Training Details
    logger.info(f"***** Running Training *****")
    logger.info(f"  Total Examples: {len(train_dataset)}")
    logger.info(f"  Total Epochs: {args.epochs}")
    logger.info(f"  Batch Size per GPU: {args.train_batch_size // max(args.n_gpu, 1)}")
    logger.info(f"  Total Optimization Steps: {args.max_steps}")
    
    # Training Loop with Optimization
    global_step, best_f1 = 0, 0
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
        train_loss, train_steps = 0, 0
        
        for step, batch in enumerate(progress_bar):
            # Batch Preparation
            inputs = [x.to(args.device) for x in batch]
            inputs_ids, position_idx, attn_mask, bytecode_embedding, opcode_tensor, labels = inputs
            
            # Model Training
            model.train()
            loss, _ = model(inputs_ids, position_idx, attn_mask, bytecode_embedding, opcode_tensor, labels)
            
            # Loss Scaling for Multi-GPU and Gradient Accumulation
            loss = loss.mean() if args.n_gpu > 1 else loss
            loss = loss / args.gradient_accumulation_steps
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Optimization Step
            train_loss += loss.item()
            train_steps += 1
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Periodic Evaluation and Model Saving
                if (global_step == 0 or  args.save_steps == 0 or global_step % args.save_steps == 0):
                    results = evaluate(args, model, tokenizer, eval_when_training=True)
                    
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        save_best_model(args, model, best_f1)
                
            progress_bar.set_description(f"Epoch {epoch} Loss: {train_loss/train_steps:.4f}")


def save_best_model(args, model, best_f1):
    """Helper function to save the best model checkpoint"""
    checkpoint_prefix = 'checkpoint-best-f1'
    output_dir = os.path.join(args.output_dir, checkpoint_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    model_to_save = model.module if hasattr(model, 'module') else model
    model_path = os.path.join(output_dir, 'model.bin')
    
    torch.save(model_to_save.state_dict(), model_path)
    logger.info(f"Best Model (F1: {best_f1:.4f}) saved to {model_path}")


def evaluate(args, model, tokenizer, eval_when_training=False):
    """Model Evaluation Function with Detailed Metrics"""
    # Dataset and Dataloader Setup
    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=args.eval_batch_size, 
        num_workers=4
    )
    
    # Multi-GPU Handling
    if args.n_gpu > 1 and not eval_when_training:
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(args.device)
    
    # Evaluation Logging
    logger.info("***** Running Evaluation *****")
    logger.info(f"  Total Examples: {len(eval_dataset)}")
    
    # Prediction Collection
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs = [x.to(args.device) for x in batch]
            inputs_ids, position_idx, attn_mask, bytecode_embedding, opcode_tensor, labels = inputs
            
            _, logit = model(inputs_ids, position_idx, attn_mask, bytecode_embedding, opcode_tensor, labels)
            
            all_logits.append(logit.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Performance Metrics Calculation
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    metrics = {
        "recall": recall_score(labels, logits, average='weighted'),
        "precision": precision_score(labels, logits, average='weighted'),
        "f1": f1_score(labels, logits, average='weighted'),
        "accuracy": accuracy_score(labels, logits)
    }
    
    # Logging Results
    logger.info("***** Evaluation Results *****")
    for key, value in metrics.items():
        logger.info(f"  {key.capitalize()}: {value:.4f}")
    
    return {f"eval_{k}": v for k, v in metrics.items()}


def test(args, model, tokenizer, eval_when_training=False):
    """Model Test Function with Detailed Metrics"""
    # Dataset and Dataloader Setup
    test_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, 
        sampler=test_sampler, 
        batch_size=args.test_batch_size, 
        num_workers=4
    )
    
    # Multi-GPU Handling
    if args.n_gpu > 1 :
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(args.device)
    # Test Logging
    logger.info("***** Running Testing *****")
    logger.info(f"  Total Examples: {len(test_dataset)}")
    
    # Prediction Collection
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            inputs = [x.to(args.device) for x in batch]
            inputs_ids, position_idx, attn_mask, bytecode_embedding, opcode_tensor, labels = inputs
            
            _, logit = model(inputs_ids, position_idx, attn_mask, bytecode_embedding, opcode_tensor, labels)
            
            all_logits.append(logit.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Performance Metrics Calculation
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    metrics = {
        "recall": recall_score(labels, logits, average='weighted'),
        "precision": precision_score(labels, logits, average='weighted'),
        "f1": f1_score(labels, logits, average='weighted'),
        "accuracy": accuracy_score(labels, logits)
    }
    
    # Logging Results
    logger.info("***** Test Results *****")
    for key, value in metrics.items():
        logger.info(f"  {key.capitalize()}: {value:.4f}")
    
    return {f"test_{k}": v for k, v in metrics.items()}



def main():
    print("EX: python run_edit_by_Null.py --epoch 1 --do_train")
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=DEF_TRAIN_DATA_FILE, type=str, 
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=DEF_OUTPUT_DIR, type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=DEF_EVAL_DATA_FILE, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=DEF_TEST_DATA_FILE, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_name_or_path", default=DEF_MODEL_NAME_OR_PATH, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default=DEF_CONFIG_NAME, type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default=DEF_TOKENIZER_NAME, type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=DEF_CODE_LENGTH, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=DEF_DATA_FLOW_LENGTH, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    parser.add_argument("--do_train", default=DEF_DO_TRAIN, action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--train_batch_size", default=DEF_TRAIN_BATCH_SIZE, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=DEF_EVAL_BATCH_SIZE, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=DEF_LEARNING_RATE, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=DEF_MAX_GRAD_NORM, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=DEF_SEED,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=DEF_EPOCH,
                        help="training epochs")

    args = parser.parse_args()


    ###############################################
    #################### SETUP ####################
    ###############################################
    # Set model ver
    model_ver = DEF_MODEL_VER
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.device = device

    # Setup CPU
    # device = torch.device("cpu")
    # args.n_gpu = 0  # No GPUs are being used
    # args.device = device
    
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu)
    # Set seed
    set_seed(args.seed, args.n_gpu)

    # Set RobertaConfig
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    print(f'\n\n CONFIG \n ############################ \n {config} \n\n############################\n')

    
    
    # Set model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, clean_up_tokenization_spaces = False)# su dung tokenizer tu pretrainmodel
    #model = RobertaModel.from_pretrained(args.model_name_or_path, config=config)  # Dùng encoder từ mô hình pre-trained
    model_pretrain = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config) #encoder ma hoa model ma hoa
    #return loss and logits co shape la 1 tensor
    model = Model(model_pretrain, config,tokenizer, args, num_classes = DEF_N_LABELS) 
    
    ###############################################
    #################### EO_SETUP #################
    ###############################################

    ###############################################
    #################### RUN  #####################
    ###############################################
    logger.info("#####################")
    logger.info("Training/evaluation parameters:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("#####################")

    # Training
    if args.do_train:
        model.to(args.device)
        train(args, model, tokenizer)

    # Evaluation
    results = {}

    if args.do_eval:
        checkpoint_prefix = f'checkpoint-best-f1/model_{model_ver}.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, weights_only = True))
        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-f1/model_{model_ver}.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, weights_only = True))
        model.to(args.device)
        test(args, model, tokenizer)

    return results


if __name__ == "__main__":
    results = main()
    logger.info(f"   Result: \n\n###########################\n {results} \n###########################")
