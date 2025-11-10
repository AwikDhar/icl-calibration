import numpy as np
import torch
from losses import brier_score, smooth_ece
from copy import deepcopy

# metrics over a single/bunch of datasets(post averaging using the dunder methods)
class Metrics():
    def __init__(self, logits: torch.tensor, calibrated_logits: torch.tensor, labels: torch.tensor, shots_start: int):
        B, T, num_classes = logits.shape
        logits = logits[:,shots_start:,:].reshape(B*(T-shots_start), num_classes).detach().cpu()
        calibrated_logits = calibrated_logits[:,shots_start:,:].reshape(B*(T-shots_start), num_classes).detach().cpu()
        labels = labels[:,shots_start:].reshape(B*(T-shots_start)).detach().cpu()
        
        self.ece = smooth_ece(logits, labels).item()
        self.calibrated_ece = smooth_ece(calibrated_logits, labels).item()
        
        self.brier_score = brier_score(logits, labels).item()
        self.calibrated_brier_score = brier_score(calibrated_logits, labels).item()

    @classmethod
    def zeros(cls):
        obj = cls.__new__()
        obj.ece = 0.0
        obj.calibrated_ece = 0.0
        obj.brier_score = 0.0
        obj.calibrated_brier_score = 0.0
        
        return obj
    
    def __add__(self, other):
        result = deepcopy(self)

        result.ece += other.ece
        result.calibrated_ece += other.calibrated_ece

        result.brier_score += other.brier_score
        result.calibrated_brier_score += other.calibrated_brier_score

        return result
        
    def __truediv__(self, divisor):
        result = deepcopy(self)

        result.ece /= divisor
        result.calibrated_ece /= divisor

        result.brier_score /= divisor
        result.calibrated_brier_score /= divisor

        return result
