import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification,AutoModel

class Model(nn.Module):
    def __init__(self, config, tokenizer, args, num_classes=10):
        super().__init__()
        self.encoder = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
        self.tokenizer = tokenizer
        self.args = args
        self.isRunningWithOnlySourceCode = args.run_w_only_sc


        if not self.isRunningWithOnlySourceCode:
            # Opcode processing
            self.opcode_embedding = nn.Embedding(1001, 256)  # Vocabulary size of opcodes
            self.opcode_rnn = nn.GRU(256, 256, batch_first=True)
            
            # Bytecode processing
            self.bytecode_proj = nn.Linear(768, 768)  # Project bytecode to match encoder dims
        

        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        
        # Fusion and classification
        self.fusion_dense = nn.Linear(768 * 1, 512)  # Combine source, bytecode, opcode

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, input_ids, position_idx, attn_mask, bytecode_embedding = None, opcode_tensor = None, labels=None):
        # Encode source code
        outputs = self.encoder(input_ids=input_ids, attention_mask=attn_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :].unsqueeze(0)  # [1, batch_size, 768]
        
        if not self.isRunningWithOnlySourceCode:
            # Encode opcode sequence
            opcode_emb = self.opcode_embedding(opcode_tensor)  # [batch_size, seq_len, 256]
            _, opcode_hidden = self.opcode_rnn(opcode_emb)     # [1, batch_size, 256]
            opcode_rep = opcode_hidden[-1].unsqueeze(0)        # [1, batch_size, 256]
            opcode_rep = nn.Linear(256, 768)(opcode_rep)       # Project to 768 dims
            
            #Project bytecode embedding
            bytecode_rep = self.bytecode_proj(bytecode_embedding).unsqueeze(0)  # [1, batch_size, 768]
            
            #Cross-attention between modalities
            query = cls_embedding
            key_value = torch.cat([bytecode_rep, opcode_rep], dim=0)  # [2, batch_size, 768]
            attended, _ = self.cross_attention(query, key_value, key_value)  # [1, batch_size, 768]
            attended = attended.squeeze(0)  # [batch_size, 768]
            
            # Fuse all features
            combined = torch.cat([attended, bytecode_embedding, opcode_rep.squeeze(0)], dim=1)  # [batch_size, 768 * 3]
        else:
            combined = cls_embedding.squeeze(0)

        fused = self.fusion_dense(combined)
        fused = torch.tanh(fused)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        
        
        # Compute loss if labels are provided
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return loss, torch.sigmoid(logits) > 0.5
        return torch.sigmoid(logits)