import torch
import torch.nn as nn
import torch.nn.functional as F

class CalibrationTransformer(nn.Module):
    def __init__(self, in_features, context_length=10, embedding_dim=128, num_heads=8, num_layers=4):
        super().__init__()

        self.pos_embedding = nn.Embedding(num_embeddings=context_length, embedding_dim=embedding_dim)
        self.embedding = nn.Linear(in_features=in_features, out_features=embedding_dim)
        
        self.register_buffer('causal_mask',torch.triu(torch.ones((context_length, context_length)) * float('-inf'), diagonal=1))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.lm_head = nn.Linear(embedding_dim, 1, bias=False)
        
    def forward(self, inputs: torch.TensorType):
        B,T,C = inputs.shape
        positions = torch.arange(T, device=inputs.device)
        inputs = self.embedding(inputs) + self.pos_embedding(positions)
        out = self.transformer_encoder(inputs, mask=self.causal_mask[:T, :T], is_causal=True)
        out = self.lm_head(out)
        out = 0.2 + 1.8*F.sigmoid(out)
        
        return out
        