from collections import namedtuple
import relplot
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

CalibrationError = namedtuple('CalibrationError', ['ece', 'mce'])

def compute_binned_ce(logits: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray, n_bins: int=10):
    """Get calibration errors by binning confidences

    Args:
        logits (torch.Tensor | np.ndarray): prediction logits
        labels (torch.Tensor | np.ndarray): GT labels

    Returns:
        BinnedCalibrationError(namedtuple): the ECE and MCE 
    """
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)
            
    probs = F.softmax(logits, dim=-1)
    
    confidences, predictions = torch.max(probs, -1)
    accuracies = predictions.eq(labels)
    
    ece = 0
    mce = 0 
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            bin_confidence_gap = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
            
            ece += (bin_confidence_gap* prop_in_bin).item()
            if bin_confidence_gap.item()>mce:
                mce = bin_confidence_gap.item()
                mce_bin_prop = prop_in_bin.item()

    # print(f"{mce_bin_prop*100}% of predictions in the MCE bin")
    return CalibrationError(ece, mce)
    
def brier_score(logits, labels):
    probs = F.softmax(logits, dim=-1)
    labels_one_hot = F.one_hot(labels, num_classes=probs.shape[-1]).float()
    
    loss = torch.mean(torch.sum((probs-labels_one_hot)**2, dim=-1))
    
    return loss

# def brier_score(logits, labels):
#     # prediction probs
#     probs = F.softmax(logits, dim=-1)
#     # ground truth probs
#     gt_probs = probs[torch.arange(len(labels)), labels]
    
#     loss = torch.mean((1.0-gt_probs)**2)
#     return loss

def smooth_ece(logits, labels, prepare_rel_diag=False, plot_confidence_band=False):
    conf, acc = relplot.multiclass_logits_to_confidences(logits, labels) # reduce to binary setting
    ece = relplot.smECE(f=conf, y=acc) # compute smECE of confidence calibration
    if prepare_rel_diag:
        # fig, ax = relplot.rel_diagram(f=conf, y=acc)
        diagram = relplot.prepare_rel_diagram(f=conf, y=acc, plot_confidence_band=plot_confidence_band)
        return ece, diagram
    return ece

class BrierLoss(nn.Module):
    def __init__(self, shots_start: int):
        super().__init__()
        self.shots_start = shots_start
        
    def forward(self, logits, labels):
        B, T, num_classes = logits.shape
        
        logits = logits[:,self.shots_start:,:].reshape(B*(T-self.shots_start), num_classes)
        labels = labels[:,self.shots_start:].reshape(B*(T-self.shots_start))
        
        return brier_score(logits, labels)