from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEmbeddingType(Enum):
    SINUSOIDAL=0
    ABSOLUTE=1
    
class CalibratorOutputType(Enum):
    TEMPERATURE=0
    CALIBRATED_PROBABILITY=1
    
class CalibrationTransformer(nn.Module):
    def __init__(
            self, 
            in_features, 
            context_length=10,
            embedding_dim=32,
            num_heads=8,
            num_layers=4,
            dropout=0.2,
            pos_embedding_type=PositionEmbeddingType.SINUSOIDAL,
            output_type = CalibratorOutputType.CALIBRATED_PROBABILITY
        ):
        super().__init__()

        if pos_embedding_type==PositionEmbeddingType.ABSOLUTE:
            self.pos_embedding = nn.Embedding(num_embeddings=context_length, embedding_dim=embedding_dim)
        elif pos_embedding_type==PositionEmbeddingType.SINUSOIDAL:
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
        
        self.lm_head = nn.Linear(embedding_dim, 1, bias=False)
        self.output_type = output_type
        
    def forward(self, inputs: torch.TensorType):
        B,T,C = inputs.shape
        
        if self.pos_embedding_type==PositionEmbeddingType.ABSOLUTE:
            positions = torch.arange(T, device=inputs.device)
            pos_embedding = self.pos_embedding(positions)
        elif self.pos_embedding_type==PositionEmbeddingType.SINUSOIDAL:
            pos_embedding = self.pos_embedding[:T, :]
        
        inputs = self.embedding(inputs) + pos_embedding
        out = self.transformer_encoder(inputs, mask=self.causal_mask[:T, :T], is_causal=True)
        out = self.lm_head(out)
        
        if self.output_type==CalibratorOutputType.CALIBRATED_PROBABILITY:
            out = F.sigmoid(out)
        elif self.output_type==CalibratorOutputType.TEMPERATURE:
            out = 0.2 + 1.8*F.sigmoid(out)
        else:
            raise NotImplementedError(f"CalibratorOutputType `{self.output_type}` not available, please pick one from {CalibratorOutputType._member_names_}")
        
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
    
_IDX_GT_PROB_MSE = 0
_IDX_TOP1 = 1
_IDX_CORRECTNESS = 2
_IDX_GT_CLASS_PROB = 3
_IDX_TOP2 = 4

_TOP_DEFAULT = 0.5  # default for top1/top2 history mean at t=0
_OUTCOME_DEFAULTS = {  # defaults for correctness/gt_class_prob/gt_prob_mse when window is empty
    "correctness": 0.5,
    "gt_class_prob": 0.5,
    "gt_prob_mse": 0.25,
}


def _prefix_mean_exclusive(x: torch.Tensor) -> torch.Tensor:
    """Mean of x[:, 0:t] for each row t, i.e. strictly-prior-rows mean.
    Row 0 output is undefined (div-by-zero avoided via clamp); caller must
    overwrite row 0 with an appropriate default."""
    T = x.shape[1]
    cumsum = torch.cumsum(x, dim=1)
    prefix_sum = cumsum - x  # sum of rows 0..t-1
    counts = torch.arange(1, T + 1, device=x.device).view(1, T) - 1
    counts = counts.clamp(min=1)
    return prefix_sum / counts


def _window_mean_1_to_t(x: torch.Tensor) -> torch.Tensor:
    """Mean of x[:, 1:t+1] for each row t, i.e. rows 1..t inclusive.
    At t=0 the window is empty; caller must overwrite row 0 with a default."""
    T = x.shape[1]
    cumsum = torch.cumsum(x, dim=1)  # cumsum[:, t] = sum of rows 0..t
    window_sum = cumsum - x[:, [0]]  # sum of rows 1..t = cumsum[t] - row 0
    counts = torch.arange(1, T + 1, device=x.device).view(1, T) - 1  # number of rows in 1..t
    counts = counts.clamp(min=1)
    return window_sum / counts


def causal_summary_stats(inputs: torch.TensorType) -> torch.Tensor:
    """
    Builds the 7-dim causal summary-statistic feature used by the MLP and
    logistic-regression baselines.

    inputs: (B, T, 5) raw feature tensor, columns as documented above.
    returns: (B, T, 7)
    """
    B, T, C = inputs.shape
    assert C == 5, f"causal_summary_stats expects 5 input features, got {C}"

    top1 = inputs[..., _IDX_TOP1]                    # (B, T)
    top2 = inputs[..., _IDX_TOP2]                     # (B, T)
    correctness = inputs[..., _IDX_CORRECTNESS]       # (B, T)
    gt_class_prob = inputs[..., _IDX_GT_CLASS_PROB]   # (B, T)
    gt_prob_mse = inputs[..., _IDX_GT_PROB_MSE]       # (B, T)

    # --- own-prediction history: mean over rows 0..t-1 ---
    top1_hist = _prefix_mean_exclusive(top1)
    top2_hist = _prefix_mean_exclusive(top2)
    top1_hist[:, 0] = _TOP_DEFAULT
    top2_hist[:, 0] = _TOP_DEFAULT

    # --- observed-outcome history: mean over rows 1..t ---
    correctness_hist = _window_mean_1_to_t(correctness)
    gt_class_prob_hist = _window_mean_1_to_t(gt_class_prob)
    gt_prob_mse_hist = _window_mean_1_to_t(gt_prob_mse)
    correctness_hist[:, 0] = _OUTCOME_DEFAULTS["correctness"]
    gt_class_prob_hist[:, 0] = _OUTCOME_DEFAULTS["gt_class_prob"]
    gt_prob_mse_hist[:, 0] = _OUTCOME_DEFAULTS["gt_prob_mse"]

    out = torch.stack(
        [
            top1_hist,
            top2_hist,
            correctness_hist,
            gt_class_prob_hist,
            gt_prob_mse_hist,
            top1,   # current, unshifted
            top2,   # current, unshifted
        ],
        dim=-1,
    )  # (B, T, 7)

    return out
    
def apply_output_activation(out: torch.TensorType, output_type: CalibratorOutputType):
    if output_type==CalibratorOutputType.CALIBRATED_PROBABILITY:
        out = F.sigmoid(out)
    elif output_type==CalibratorOutputType.TEMPERATURE:
        out = 0.2 + 1.8*F.sigmoid(out)
    else:
        raise NotImplementedError(f"CalibratorOutputType `{output_type}` not available, please pick one from {CalibratorOutputType._member_names_}")
    return out

class CalibrationMLP(nn.Module):
    def __init__(
            self, 
            hidden_dim=64,
            num_layers=5,
            dropout=0.2,
            output_type = CalibratorOutputType.CALIBRATED_PROBABILITY
        ):        
        super().__init__()
        summary_dim = 7  # fixed by causal_summary_stats
        self.layers = nn.ModuleList([nn.Linear(in_features=summary_dim, out_features=hidden_dim)])
        self.layers.append(nn.Dropout(dropout))
        
        for layer in range(num_layers-1):
            self.layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            self.layers.append(nn.Dropout(dropout))
            
        self.layers.append(nn.Linear(in_features=hidden_dim, out_features=1))
        self.output_type = output_type
            
    def forward(self, inputs: torch.TensorType):
        x = causal_summary_stats(inputs) # (B, T, 7)
        
        for layer in self.layers:
            x = layer(x)
        
        out = apply_output_activation(x, self.output_type)
        
        return out
    
class CalibrationRecurrent(nn.Module):
    def __init__(
            self,
            in_features,
            cell_type='lstm', 
            hidden_dim=256,
            num_layers=2,
            dropout=0.2,
            output_type = CalibratorOutputType.CALIBRATED_PROBABILITY
        ):
        super().__init__()
        
        cell_type = cell_type.lower()
        assert cell_type in ('rnn', 'lstm'), f"cell_type must be 'rnn' or 'lstm', got `{cell_type}`"
        self.cell_type = cell_type
        
        self.embedding = nn.Linear(in_features=in_features, out_features=hidden_dim)
        
        recurrent_cls = nn.LSTM if cell_type=='lstm' else nn.RNN
        self.recurrent = recurrent_cls(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0.0,
        )
        
        self.head = nn.Linear(hidden_dim, 1)
        self.output_type = output_type
        
    def forward(self, inputs: torch.TensorType):
        x = self.embedding(inputs) # (B, T, hidden_dim)
        x, _ = self.recurrent(x) # (B, T, hidden_dim), causal by construction
        out = self.head(x) # (B, T, 1)
        
        out = apply_output_activation(out, self.output_type)
        
        return out
 
class CalibrationRNN(CalibrationRecurrent):
    def __init__(self, in_features, hidden_dim=386, num_layers=2, dropout=0.2,
                 output_type=CalibratorOutputType.CALIBRATED_PROBABILITY):
        super().__init__(in_features, cell_type='rnn', hidden_dim=hidden_dim,
                        num_layers=num_layers, dropout=dropout, output_type=output_type)
 
class CalibrationLSTM(CalibrationRecurrent):
    def __init__(self, in_features, hidden_dim=273, num_layers=1, dropout=0.2,
                 output_type=CalibratorOutputType.CALIBRATED_PROBABILITY):
        super().__init__(in_features, cell_type='lstm', hidden_dim=hidden_dim,
                        num_layers=num_layers, dropout=dropout, output_type=output_type)
 
class CalibrationLogisticRegressor(nn.Module):
    def __init__(
            self,
            output_type = CalibratorOutputType.CALIBRATED_PROBABILITY
        ):
        super().__init__()

        summary_dim = 7  # fixed by causal_summary_stats
        self.head = nn.Linear(in_features=summary_dim, out_features=1)
        self.output_type = output_type
 
    def forward(self, inputs: torch.TensorType):
        x = causal_summary_stats(inputs) # (B, T, 7)
        x = self.head(x)
 
        out = apply_output_activation(x, self.output_type)
 
        return out