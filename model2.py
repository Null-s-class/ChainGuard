import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig

class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        
        # Improved classification head
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 4)
        
        # Additional layers
        self.opcode_dense = nn.Linear(200, config.hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=12)
    
    def forward(self, inputs_ids, position_idx, attn_mask, bytecode_embedding, opcode_tensor, labels=None):       
        # Embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        inputs_embeddings[:,:20,:] += bytecode_embedding
        
        # Improved opcode handling
        opcode_tensor = opcode_tensor.float()
        opcode_transformed = self.opcode_dense(opcode_tensor)
        opcode_transformed = opcode_transformed.unsqueeze(1).expand(-1, inputs_embeddings.size(1), -1)
        
        # Apply attention
        inputs_embeddings, _ = self.attention(inputs_embeddings, opcode_transformed, opcode_transformed)
        print('Shape of inputs embeeing', inputs_embeddings.shape)
        # Encode
        outputs = self.encoder(
            inputs_embeds=inputs_embeddings,
            attention_mask=attn_mask,
            position_ids=position_idx,
            token_type_ids=position_idx.eq(-1).long()
        )[0]
        
        # Improved classification head
        logits = self.out_proj(self.dropout(F.gelu(self.dense(outputs[:,0,:]))))
        prob = torch.sigmoid(logits)
        
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return loss, prob
        else:
            return prob