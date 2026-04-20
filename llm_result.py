
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x)

@dataclass
class Logprobs:
    probs: List[np.ndarray]
    logits: List[np.ndarray]
    hidden_features: List[np.ndarray] = None

    def to_numpy(self) -> None:
        self.probs = [to_numpy(prob) for prob in self.probs]
        self.logits = [to_numpy(logit) for logit in self.logits]
        if self.hidden_features is not None:
            self.hidden_features = [to_numpy(hidden) for hidden in self.hidden_features]

@dataclass
class LlmResult:
    logprobs: List[Logprobs]
    in_context_logprobs: List[Logprobs] = None

    def to_numpy(self) -> None:
        for logprob in self.logprobs:
            logprob.to_numpy()
        if self.in_context_logprobs is not None:
            for ic_logprob in self.in_context_logprobs:
                ic_logprob.to_numpy()
    
    
        