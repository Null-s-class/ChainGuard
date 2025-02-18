import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 10) # 10 classes

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])cl
        #x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args , num_classes=10):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.hidden_size = config.hidden_size
        self.args = args
        self.opcode_dense = nn.Linear(1,self.hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8)
        self.model_info_show = False
    
    @staticmethod
    def expand_tensor_with_padding(tensor, new_size):
        batch_size, old_size, _ = tensor.shape
        pad_size = new_size - old_size
        padded_tensor = F.pad(tensor, (0, pad_size, 0, pad_size), "constant", 0)
        return padded_tensor

    @staticmethod
    def expand_tensor_with_positionidx(tensor, new_size):
        batch_size, old_size = tensor.shape
        pad_size = new_size - old_size
        padded_tensor = F.pad(tensor, (0, pad_size), "constant", 0)
        return padded_tensor

    def forward(self, inputs_ids, position_idx, attn_mask, bytecode_embedding, opcode_tensor, labels=None):
        """
        Forward pass for the model with enhanced embedding processing and classification.
        
        Args:
            inputs_ids (torch.Tensor): Input token IDs
            position_idx (torch.Tensor): Position indices for tokens
            attn_mask (torch.Tensor): Attention mask
            bytecode_embedding (torch.Tensor): Bytecode embedding
            opcode_tensor (torch.Tensor): Opcode tensor
            labels (torch.Tensor, optional): Ground truth labels
        
        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: 
            Loss and prediction if labels are provided, otherwise prediction probabilities
        """
        # Process node and token embeddings
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)

        # Compute average embeddings for nodes
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        # Transform and incorporate opcode embeddings
        opcode_tensor = opcode_tensor.float()
        opcode_transformed = opcode_tensor.unsqueeze(-1)  # [batch_size, 512, 1]
        opcode_transformed = self.opcode_dense(opcode_transformed)  # [batch_size, 512, 768]

        # Concatenate embeddings
        inputs_embeddings = torch.cat([inputs_embeddings, bytecode_embedding, opcode_transformed], dim=1)

        # Expand attention mask and position indices
        attn_mask = self.expand_tensor_with_padding(attn_mask, inputs_embeddings.size(dim=1))
        position_idx = self.expand_tensor_with_positionidx(position_idx, inputs_embeddings.size(dim=1))

        if (self.model_info_show == False): # show only once 
            self.model_info_show = True
            logger.info(f"Shape of inputs_embedding: {inputs_embeddings.shape}\n")
            logger.info(f"Shape of input_ids: {inputs_ids.shape}\n")
            logger.info(f"Shape of position_idx: {position_idx.shape}\n")
            logger.info(f"Shape of attn_mask: {attn_mask.shape}\n")
            logger.info(f"Shape of bytecode embedding: {bytecode_embedding.shape}\n")
            logger.info(f"Shape of opcode: {opcode_tensor.shape}\n")
            # logger.info(f"Opcode received from extract: {opcode_tensor}\n")
            # logger.info(f"Bytecode received from extract: {bytecode_embedding}\n")
            # logger.info(f"Input embedding: {inputs_embeddings}\n")
            # logger.info(f"Position_idx: {position_idx}\n")
            # logger.info(f"Attn_mask: {attn_mask}\n")

        # Encode with RoBERTa
        outputs = self.encoder.roberta(
            inputs_embeds=inputs_embeddings,
            attention_mask=attn_mask,
            position_ids=position_idx,
            token_type_ids=position_idx.eq(-1).long()
        )[0]

        # Classify and compute probabilities
        logits = self.classifier(outputs)
        prob = torch.sigmoid(logits)
        prediction = (prob > 0.5).int()

        # Compute loss if labels are provided
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return loss, prediction
        
        return prob