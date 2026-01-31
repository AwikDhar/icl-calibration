import numpy as np
import torch
from losses import brier_score, smooth_ece, compute_binned_ce
from copy import deepcopy

import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(timeout_sec):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(timeout_sec)
    try:
        yield
    finally:
        signal.alarm(0)
        
class ClassificationMetrics():
    def __init__(self, logits: torch.Tensor, calibrated_logits: torch.Tensor, 
                 labels: torch.Tensor, binned: bool=False):
        
        logits = self._to_tensor(logits)
        calibrated_logits = self._to_tensor(calibrated_logits)
        labels = self._to_tensor(labels)
        
        self.calibration_metrics = CalibrationMetrics(logits, calibrated_logits, labels, shots_start=None, binned=binned)
        
        self.accuracy            =            (logits.argmax(dim=-1)==labels).float().mean()
        self.calibrated_accuracy = (calibrated_logits.argmax(dim=-1)==labels).float().mean()
        
        self.mean_conf            =            logits.softmax(dim=-1).max(dim=-1).values.mean()
        self.mean_calibrated_conf = calibrated_logits.softmax(dim=-1).max(dim=-1).values.mean()

    @classmethod
    def _to_tensor(self, array: np.ndarray | torch.Tensor):
        if isinstance(array, torch.Tensor):
            return array
        elif isinstance(array, np.ndarray):
            return torch.from_numpy(array)
        else:
            return torch.tensor(array)
    
# metrics over a single/bunch of datasets(post averaging using the dunder methods)
class CalibrationMetrics():
    def __init__(self, logits: torch.Tensor, calibrated_logits: torch.Tensor, labels: torch.Tensor, shots_start: int, 
                 prepare_rel_diag: bool = False, plot_confidence_band: bool =False, binned: bool=False):
       
        if logits.ndim==3:
            # Reshape tensors: flatten batch and time dimensions after shots_start
            B, T, num_classes = logits.shape
            N = B * (T - shots_start)  # Total samples
        
            logits = logits[:, shots_start:, :].reshape(N, num_classes)
            calibrated_logits = calibrated_logits[:, shots_start:, :].reshape(N, num_classes)
            labels = labels[:, shots_start:].reshape(N)
        
        elif logits.ndim!=2:
            raise ValueError(f"Invalid ndims: {logits.ndims}, only 2 (B, num_classes) or 3 (B, shots, num_classes) allowed")
        
        logits = logits.detach().cpu()
        calibrated_logits = calibrated_logits.detach().cpu()
        labels = labels.detach().cpu()
        
        self.ece, self.mce, self.rel_diag = self._compute_ece(logits, labels, prepare_rel_diag, plot_confidence_band, binned)
        self.calibrated_ece, self.calibrated_mce, self.calibrated_rel_diag = self._compute_ece(calibrated_logits, labels, prepare_rel_diag, plot_confidence_band, binned)            
        
        self.brier_score            = brier_score(logits,            labels).item()
        self.calibrated_brier_score = brier_score(calibrated_logits, labels).item()

    def _compute_ece(self, logits, labels, prepare_rel_diag=False, plot_confidence_band=False, binned=False):
        binned_ce = compute_binned_ce(logits, labels)
        
        if binned:
            return binned_ce.ece, binned_ce.mce, None
        
        timeout_sec=10
        try:
            with time_limit(timeout_sec):
                out = smooth_ece(logits, labels, prepare_rel_diag, plot_confidence_band)
                if prepare_rel_diag:
                    return out[0].item(), binned_ce.mce, out[1]
                return out.item(), binned_ce.mce, None
        except TimeoutException:
            import logging
            logging.error(f"Smooth ECE calculation timed out after {timeout_sec} seconds, falling back to binned ECE")
            return binned_ce.ece, binned_ce.mce, None
    
    @classmethod
    def zeros(cls):
        obj = cls.__new__(cls)
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
