import argparse
import logging
import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AutoConfig,AutoTokenizer,AutoModel, AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
from model_edit_by_Null import Model
from data_processing.dataset_class import TextDataset
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import torch.cuda.amp as amp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEF_SEED = 42
DEF_OUTPUT_DIR = 'saved_models'
DEF_CONFIG_NAME = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
DEF_MODEL_NAME_OR_PATH = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
DEF_TOKENIZER_NAME = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'


DEF_TRAIN_DATA_FILE = 'Data/Dataset/train.txt'
DEF_EVAL_DATA_FILE = 'Data/Dataset/valid.txt'
DEF_TEST_DATA_FILE = 'Data/Dataset/test.txt'
DEF_EPOCH = 3
DEF_CODE_LENGTH = 128
DEF_DATA_FLOW_LENGTH = 32
DEF_HIDDEN_SIZE = 256
DEF_TRAIN_BATCH_SIZE = 4
DEF_EVAL_BATCH_SIZE = 8
DEF_LEARNING_RATE = 2e-5
DEF_MAX_GRAD_NORM = 1.0
DRY_RUN_MODE = True
DRY_RUN_DATA_SAMPLES = 500
RUN_W_ONLY_SC = True # False = RUN_W_SC_OPCODE_BYTE 

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def train(args, model, tokenizer, train_dataloader):

    # Training configuration
    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 5
    args.warmup_steps = args.max_steps // 5
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, 
                                                num_training_steps=args.max_steps)
    
    # Move model to device and enable optimizations
    model.to(args.device)
    scaler = amp.GradScaler()              # Mixed precision training
    
    logger.info(f"Training: {len(train_dataloader)} examples, {args.epochs} epochs")
    global_step, best_f1 = 0, 0
    
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
        train_loss = 0
        for step, batch in enumerate(progress_bar):
            model.train()
            inputs = [x.to(args.device) for x in batch]
            
            with amp.autocast():
                loss, _ = model(*inputs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            train_loss += loss.item()
            
            if global_step % args.save_steps == 0:
                results = evaluate(args, model, tokenizer)  # Assume evaluate() is defined
                if results['eval_f1'] > best_f1:
                    best_f1 = results['eval_f1']
                    save_best_model(args, model, best_f1)  # Assume save_best_model() is defined
            
            progress_bar.set_description(f"Epoch {epoch} Loss: {train_loss/(step+1):.4f}")


def save_best_model(args, model, best_f1):
    output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1')
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.bin')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Best Model (F1: {best_f1:.4f}) saved to {model_path}")

def evaluate(args, model, tokenizer):
    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file, 
                               DRY_RUN_MODE=DRY_RUN_MODE, DRY_RUN_DATA_SAMPLES=DRY_RUN_DATA_SAMPLES)
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), 
                                 batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
    
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs = [x.to(args.device) for x in batch]
            with amp.autocast():
                logits = model(*inputs[:-1])  # Exclude labels
            all_logits.append(logits.cpu().numpy())
            all_labels.append(inputs[-1].cpu().numpy())
    
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metrics = {
        "recall": recall_score(labels, logits, average='weighted'),
        "precision": precision_score(labels, logits, average='weighted'),
        "f1": f1_score(labels, logits, average='weighted'),
        "accuracy": accuracy_score(labels, logits)
    }
    logger.info("***** Evaluation Results *****")
    for k, v in metrics.items():
        logger.info(f"  {k.capitalize()}: {v:.4f}")
    return {f"eval_{k}": v for k, v in metrics}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", default=DEF_TRAIN_DATA_FILE, type=str)
    parser.add_argument("--output_dir", default=DEF_OUTPUT_DIR, type=str)
    parser.add_argument("--eval_data_file", default=DEF_EVAL_DATA_FILE, type=str)
    parser.add_argument("--test_data_file", default=DEF_TEST_DATA_FILE, type=str)
    
    parser.add_argument("--model_name_or_path", default=DEF_MODEL_NAME_OR_PATH, type=str)
    parser.add_argument("--config_name", default=DEF_CONFIG_NAME, type=str)
    parser.add_argument("--tokenizer_name", default=DEF_TOKENIZER_NAME, type=str)

    parser.add_argument("--code_length", default=DEF_CODE_LENGTH, type=int)
    parser.add_argument("--data_flow_length", default=DEF_DATA_FLOW_LENGTH, type=int)

    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_prepare", action='store_true')

    parser.add_argument("--train_batch_size", default=DEF_TRAIN_BATCH_SIZE, type=int)
    parser.add_argument("--eval_batch_size", default=DEF_EVAL_BATCH_SIZE, type=int)
    parser.add_argument("--learning_rate", default=DEF_LEARNING_RATE, type=float)
    parser.add_argument("--max_grad_norm", default=DEF_MAX_GRAD_NORM, type=float)
    parser.add_argument("--epochs", default=DEF_EPOCH, type=int)
    parser.add_argument("--seed", default=DEF_SEED, type=int)
    parser.add_argument("--dry_run_mode", action='store_true', default=DRY_RUN_MODE)
    parser.add_argument("--dry_run_samples", default=DRY_RUN_DATA_SAMPLES, type=int)
    parser.add_argument("--hidden_size", default=DEF_HIDDEN_SIZE, type=int)

    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)
    
    args.run_w_only_sc = RUN_W_ONLY_SC

    config = AutoConfig.from_pretrained(args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, pad_token='<|endoftext|>') 
    logger.info('\n\n\n ============================= BEGIN ===============================\n\n\n')
    if args.do_prepare:
        # Preprocess opcode and bytecode (load paths only)
        train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file, 
                                    DRY_RUN_MODE=DRY_RUN_MODE, DRY_RUN_DATA_SAMPLES=DRY_RUN_DATA_SAMPLES)

        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), 
                                    batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
        logger.info("\nDone.")

        logger.info("\n\n\n =============================TEST DATASET returns ===============================\n\n\n")
        NUM_SAMPLES = 6
        # Display NUM_SAMPLES random samples from the dataset to test
        indices = random.sample(range(len(train_dataset)), NUM_SAMPLES)
        for i in indices:
            sample = train_dataset[i]
            logger.info(f"Sample {i}:")
            logger.info(f"Source code input shape: {sample[0].shape}")
            logger.info(f"Data flow input shape: {sample[1].shape}")
            logger.info(f"Label: {sample[-1]}")
            
            # Print first 50 tokens of source code
            decoded_text = tokenizer.decode(sample[0][:50])
            logger.info(f"Source code preview: {decoded_text}...")

        sys.exit(0)

        
        
    model = Model(config, tokenizer, args)
    if args.do_train:
                # Initialize dataset and dataloader
        train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file, 
                                    DRY_RUN_MODE=DRY_RUN_MODE, DRY_RUN_DATA_SAMPLES=DRY_RUN_DATA_SAMPLES)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), 
                                    batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
        train(args, model, tokenizer, train_dataloader)
    if args.do_eval:
        z = 'v1'
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f'checkpoint-best-f1/model_{z}.bin')))
        evaluate(args, model, tokenizer)

if __name__ == "__main__":
    main()