from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEmbeddingType(Enum):
    Sinusoidal=1
    Absolute=2
    
class SurpriseCalibrationTransformer(nn.Module):
    def __init__(
            self, 
            in_features, 
            context_length=10,
            embedding_dim=32,
            num_heads=8,
            num_layers=4,
            dropout=0.2,
            pos_embedding_type=PositionEmbeddingType.Absolute
        ):
        super().__init__()

        if pos_embedding_type==PositionEmbeddingType.Absolute:
            self.pos_embedding = nn.Embedding(num_embeddings=context_length, embedding_dim=embedding_dim)
        elif pos_embedding_type==PositionEmbeddingType.Sinusoidal:
            pe = self.sinusoidal_pe(seq_len=context_length, embedding_dim=embedding_dim) # (CL, embedding_dim)
            self.register_buffer("pos_embedding", pe)
        self.pos_embedding_type = pos_embedding_type
        
        self.embedding = nn.Linear(in_features=in_features, out_features=embedding_dim)
        
        self.register_buffer('causal_mask',torch.triu(torch.ones((context_length, context_length)) * float('-inf'), diagonal=1))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            dim_feedforward=4*embedding_dim,
            nhead=num_heads, 
            activation=F.gelu,
            batch_first=True,
            dropout=dropout,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.lm_head = nn.Linear(embedding_dim, in_features, bias=False)
        
    def forward(self, inputs: torch.TensorType):
        B,T,C = inputs.shape
        
        if self.pos_embedding_type==PositionEmbeddingType.Absolute:
            positions = torch.arange(T, device=inputs.device)
            pos_embedding = self.pos_embedding(positions)
        elif self.pos_embedding_type==PositionEmbeddingType.Sinusoidal:
            pos_embedding = self.pos_embedding[:T, :]
        
        inputs = self.embedding(inputs) + pos_embedding
        out = self.transformer_encoder(inputs, mask=self.causal_mask[:T, :T], is_causal=True)
        out = self.lm_head(out)
            
        return out
        
    def sinusoidal_pe(self, seq_len: int = None, positions: torch.TensorType = None, embedding_dim: int = None):
        assert seq_len or positions, "Need either the sequence length or positions tensor to compute the sinusoidal embeddings"
        
        if seq_len and not positions:
            positions = torch.arange(seq_len).unsqueeze(-1) # (T, 1)        
        
        i = torch.arange(embedding_dim//2)
        exponent = 2*i/embedding_dim
        denominator = torch.pow(100, exponent)
        
        theta = positions/denominator
        pe = torch.zeros(seq_len, embedding_dim)
                
        pe[:, 0::2] = torch.sin(theta)
        pe[:, 1::2] = torch.cos(theta)
        
        return pe
        
        