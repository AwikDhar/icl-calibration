import argparse
import json
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn.functional as F

from calibration.data_utils import get_batch, load_datasets

def get_shotwise_static_temperatures(data, shots_start, batch_size=None):  
    inputs, logits, labels = get_batch(data, batch_size=batch_size) # len(eval),T,C | len(eval),T,num_classes | len(eval),T
    
    B,T,num_classes = logits.shape
    tuned_temps = {}
    
    for shot in range(shots_start, T):
        shot_logits, shot_labels = logits[:, shot, :], labels[:, shot]
        tuned_temps[shot] = tune_temp(shot_logits, shot_labels)
        
    return tuned_temps

def tune_temp(logits, labels, lower=0.01, upper=5.0, eps=0.00001):
    # if logits.device!='cpu':
    #     logits = logits.detach().cpu()
    #     labels = labels.detach().cpu()

    while upper - lower > eps:
        t_guess = torch.tensor([0.5 * (lower + upper)], device=logits.device, requires_grad=True)
        loss = F.cross_entropy(logits * t_guess, labels)
        grad = torch.autograd.grad(loss, t_guess)[0]
        
        if  grad> 0:
            upper = 0.5 * (lower + upper)
        else:
            lower = 0.5 * (lower + upper)
        
    t = min([lower, 0.5 * (lower + upper), upper], key=lambda x: float(F.cross_entropy(logits * x, labels)))
    return t
        
def get_equivalent_temp(logits, calibrated_pred_probs, num_iters=500):
    """Binary search for T such that the temperature scaled prediction prob is the same as the calibrated prob given by calibrator"""
    B, T_seq, C = logits.shape
    target_log_prob = torch.log(calibrated_pred_probs)  # B, T_seq, 1
    
    T_low = torch.ones(B, T_seq, 1, device=logits.device) * 0.01
    T_high = torch.ones(B, T_seq, 1, device=logits.device) * 100.0
    
    for _ in range(num_iters):
        T_mid = (T_low + T_high) / 2
        scaled_logits = logits * T_mid
        log_probs = scaled_logits - scaled_logits.logsumexp(dim=-1, keepdim=True)
        current_max_log_prob = log_probs.max(dim=-1, keepdim=True).values
        
        # If current max prob is too high, T is not low enough (need to flatten more)
        T_high = torch.where(current_max_log_prob > target_log_prob, T_mid, T_high)
        T_low = torch.where(current_max_log_prob <= target_log_prob, T_mid, T_low)
    
    return (T_low + T_high) / 2
 
def get_equivalent_temp_gd(logits, calibrated_pred_probs, num_iters=500, lr=0.1):
    """Gradient descent to find T such that temperature scaled prediction prob matches calibrated prob"""
    B, T_seq, C = logits.shape
    target_log_prob = torch.log(calibrated_pred_probs)  # B, T_seq, 1
    
    # Initialize T (require gradients)
    T = torch.ones(B, T_seq, 1, device=logits.device, requires_grad=True)
    
    # Use an optimizer
    optimizer = torch.optim.AdamW([T], lr=lr)
    
    for _ in range(num_iters):
        optimizer.zero_grad()
        
        # Apply temperature scaling (your convention: multiply)
        scaled_logits = logits * T
        log_probs = scaled_logits - scaled_logits.logsumexp(dim=-1, keepdim=True)
        current_max_log_prob = log_probs.max(dim=-1, keepdim=True).values
        
        # Loss: MSE between current and target log probs
        loss = ((current_max_log_prob - target_log_prob) ** 2).mean()
        
        loss.backward()
        optimizer.step()
        
        # Clamp T to reasonable range
        with torch.no_grad():
            T.clamp_(0.01, 100.0)
    
    return T.detach()
 
def tune_temp_combined(logits_list, labels_list, lower=0.01, upper=5.0, eps=0.01):
    """
    Tune a single temperature across multiple datasets with different num_classes.
    
    Args:
        logits_list: List of tensors, each of shape (B_i, num_classes_i)
        labels_list: List of tensors, each of shape (B_i,)
    """
    # Move to CPU if needed
    # logits_list = [logits.detach().cpu() if logits.device != 'cpu' else logits for logits in logits_list]
    # labels_list = [labels.detach().cpu() if labels.device != 'cpu' else labels for labels in labels_list]

    while upper - lower > eps:
        t_guess = torch.tensor([0.5 * (lower + upper)], device=logits_list[0].device, requires_grad=True)
        
        # Compute combined loss across all datasets
        total_loss = 0
        for logits, labels in zip(logits_list, labels_list):
            total_loss += F.cross_entropy(logits * t_guess, labels)
        
        grad = torch.autograd.grad(total_loss, t_guess)[0]
        
        if grad > 0:
            upper = 0.5 * (lower + upper)
        else:
            lower = 0.5 * (lower + upper)
    
    # Final selection: evaluate all three candidates
    def eval_loss(t):
        total = 0
        for logits, labels in zip(logits_list, labels_list):
            total += float(F.cross_entropy(logits * t, labels))
        return total
    
    t = min([lower, 0.5 * (lower + upper), upper], key=eval_loss)
    return t


def get_combined_shotwise_static_temperatures(data, shots_start):  
    """
    Learn a common temperature across all datasets for each shot position.
    
    Args:
        data: Dict mapping dataset_key -> {'inputs': ..., 'logits': ..., 'labels': ...}
        shots_start: First shot position to start calibration from
    """
    # First, collect all data organized by shot position
    shot_data = {}  # shot -> {'logits_list': [...], 'labels_list': [...]}
    
    for dataset_key in data.keys():
        inputs, logits, labels = get_batch(data[dataset_key], batch_size=None)
        B, T, num_classes = logits.shape
        
        for shot in range(shots_start, T):
            if shot not in shot_data:
                shot_data[shot] = {'logits_list': [], 'labels_list': []}
            
            shot_logits = logits[:, shot, :]  # (B, num_classes)
            shot_labels = labels[:, shot]      # (B,)
            
            shot_data[shot]['logits_list'].append(shot_logits)
            shot_data[shot]['labels_list'].append(shot_labels)
    
    # Now tune temperature for each shot using combined loss
    tuned_temps = {}
    for shot, data_dict in shot_data.items():
        tuned_temps[shot] = tune_temp_combined(
            data_dict['logits_list'], 
            data_dict['labels_list']
        )
    
    return tuned_temps

def tune_temp_for_sequence_gd(logits_seq: torch.TensorType, labels_seq: torch.TensorType, 
                           temp_init: torch.TensorType = None, iterations: int = 500, lr: float = 0.01):
    """Train a temp for each sequence of predictions for each sequence in batch

    Args:
        logits_seq (torch.TensorType): (B, k-shots, num_classes) 
        labels_seq (torch.TensorType): (B, k-shots)
    
    Returns:
        temperatures (torch.TensorType): (B,) optimized temperatures
    """    
    B, K, num_classes = logits_seq.shape
    if temp_init is None:
        temperatures = torch.ones(B, device=logits_seq.device, requires_grad=True)
    else:
        temperatures = temp_init.clone().detach().requires_grad_(True)
            
    optimizer = torch.optim.AdamW([temperatures], lr=lr) 
    
    for _ in range(iterations):
        optimizer.zero_grad()
        
        calibrated_logits = logits_seq * temperatures.view(B, 1, 1)  # (B, K, C)
        
        loss = F.cross_entropy(
            calibrated_logits.reshape(B*K, num_classes), 
            labels_seq.reshape(B*K),
        )
        
        loss.backward()
        optimizer.step()
                # Clamp T to reasonable range
        with torch.no_grad():
            temperatures.clamp_(0.01, 100.0)
            
        if temperatures.grad is not None and torch.norm(temperatures.grad) < 0.001:
            break
    
    return temperatures.detach()        

def tune_temp_for_sequence(logits_seq: torch.TensorType, labels_seq: torch.TensorType,
                           temp_init: torch.TensorType = None, 
                           temp_min: float = 0.01, temp_max: float = 100.0,
                           max_iterations: int = 500, tolerance: float = 1e-4):
    """Find optimal temperature for each sequence using binary search on the loss landscape.
    
    Args:
        logits_seq (torch.TensorType): (B, k-shots, num_classes) 
        labels_seq (torch.TensorType): (B, k-shots)
        temp_init (torch.TensorType): (B,) initial temperatures (optional)
        temp_min (float): minimum temperature bound
        temp_max (float): maximum temperature bound
        max_iterations (int): maximum binary search iterations
        tolerance (float): convergence tolerance
    
    Returns:
        temperatures (torch.TensorType): (B,) optimized temperatures
    """
    B, K, num_classes = logits_seq.shape
    device = logits_seq.device
    
    if temp_init is None:
        temperatures = torch.ones(B, device=device)
    else:
        temperatures = temp_init.clone()
    
    # Initialize search bounds for each sequence
    temp_low = torch.full((B,), temp_min, device=device)
    temp_high = torch.full((B,), temp_max, device=device)
    
    def compute_loss(temps):
        """Helper to compute loss for given temperatures"""
        calibrated_logits = logits_seq * temps.view(B, 1, 1)
        loss_per_sample = F.cross_entropy(
            calibrated_logits.reshape(B*K, num_classes),
            labels_seq.reshape(B*K),
            reduction='none'
        ).reshape(B, K).mean(dim=1)  # (B,)
        return loss_per_sample
    
    # Use ternary search for each batch element
    for iteration in range(max_iterations):
        # Check convergence
        if (temp_high - temp_low).max() < tolerance:
            break
        
        # Ternary search: evaluate at two interior points
        mid1 = temp_low + (temp_high - temp_low) / 3
        mid2 = temp_high - (temp_high - temp_low) / 3
        
        loss_mid1 = compute_loss(mid1)
        loss_mid2 = compute_loss(mid2)
        
        # Update bounds based on which midpoint has lower loss
        # If loss_mid1 < loss_mid2, optimum is in [low, mid2]
        # If loss_mid2 < loss_mid1, optimum is in [mid1, high]
        mask_left = loss_mid1 < loss_mid2
        temp_high = torch.where(mask_left, mid2, temp_high)
        temp_low = torch.where(mask_left, temp_low, mid1)
    
    # Final temperature is midpoint of converged range
    temperatures = (temp_low + temp_high) / 2
    
    with torch.no_grad():
        temperatures.clamp_(0.01, 100.0)
        
    return temperatures

def _fit_predict_calibrated_prob(conf_hist: np.ndarray, correct_hist: np.ndarray,
                                  conf_query: float, method: str,
                                  min_fit_points: int = 2) -> float:
    has_enough_points = len(conf_hist) >= min_fit_points
    has_both_classes = has_enough_points and (len(np.unique(correct_hist)) > 1)
 
    if not has_both_classes:
        return conf_query
 
    if method == 'isotonic_cal':
        regressor = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        regressor.fit(conf_hist, correct_hist)
        return float(regressor.predict([conf_query])[0])
 
    elif method == 'beta_cal':
        eps = 1e-6
        def beta_features(p):
            p = np.clip(p, eps, 1 - eps)
            return np.stack([np.log(p), np.log(1 - p)], axis=-1)
 
        regressor = LogisticRegression()
        regressor.fit(beta_features(conf_hist), correct_hist)
        return float(regressor.predict_proba(beta_features(np.array([conf_query])))[0, 1])
 
    
def get_shotwise_scalar_calibrated_temperatures(logits: torch.Tensor, labels: torch.Tensor,
                                                 shots_start: int, method: str,
                                                 equivalent_temp_iters: int = 500):
    B, T, num_classes = logits.shape
    device = logits.device
 
    probs = F.softmax(logits, dim=-1)
    confidences = probs.max(dim=-1).values  # (B, T)
    correctness = (logits.argmax(dim=-1) == labels).float()  # (B, T)
 
    confidences_np = confidences.detach().cpu().numpy().astype(np.float64)
    correctness_np = correctness.detach().cpu().numpy().astype(np.float64)
 
    tuned_temps = {}
 
    for shot in range(shots_start, T):
        calibrated_probs = np.empty(B, dtype=np.float64)
 
        for b in range(B):
            conf_hist = confidences_np[b, :shot]
            correct_hist = correctness_np[b, :shot]
            conf_query = confidences_np[b, shot]
 
            calibrated_probs[b] = _fit_predict_calibrated_prob(
                conf_hist, correct_hist, conf_query, method
            )
 
        calibrated_probs_t = torch.tensor(calibrated_probs, device=device, dtype=logits.dtype).view(B, 1, 1)
        query_logits = logits[:, [shot], :]  # (B, 1, num_classes)
 
        equivalent_temp = get_equivalent_temp(query_logits, calibrated_probs_t, num_iters=equivalent_temp_iters)  # (B, 1, 1)
        tuned_temps[shot] = equivalent_temp.view(B)
 
    return tuned_temps

def get_shotwise_dynamic_temperatures(data, shots_start, method: str = 'temp_scaling'):
    logits, labels = data['logits'], data['labels'] # len(eval),T,num_classes | len(eval),T
 
    if method == 'temp_scaling':
        B,T,num_classes = logits.shape
        tuned_temps = {}
        temp_init = None
 
        for shot in range(shots_start, T):
            logits_seq, labels_seq = logits[:, :shot, :], labels[:, :shot]
            tuned_temps[shot] = tune_temp_for_sequence(logits_seq, labels_seq, temp_init)
 
            temp_init = tuned_temps[shot]
 
        return tuned_temps
 
    elif method in ('beta_cal', 'isotonic_cal'):
        return get_shotwise_scalar_calibrated_temperatures(logits, labels, shots_start, method)
 
    else:
        raise NotImplementedError(
            f"method `{method}` not available, please pick one from ['temp_scaling', 'beta_cal', 'isotonic_cal']"
        )

def main(llms, 
         datasets,
         feature_type,
         shots_start,
         sampling_strategy = None,
         gpu_id=0):
    
    device = f'cuda:{gpu_id}'
    
    data = load_datasets(llms, datasets, device, feature_type, splits=('train',), 
                        sampling_strategy=sampling_strategy, temp_augment=False, label_augment=False)
    print("Loaded the datasets")

    # Restructure data to be flat dict of all llm-dataset combinations
    combined_data = {}
    for llm in llms:
        for dataset in datasets:
            key = f"{llm}_{dataset}"
            combined_data[key] = data[llm][dataset]['train']
        
    shotwise_static_temps = get_combined_shotwise_static_temperatures(combined_data, shots_start)
    
    shotwise_static_temps = {k: float(v) for k, v in shotwise_static_temps.items()}
    
    with open("calibration/trained_temperature.json", "w") as file:
        json.dump(shotwise_static_temps, file)
    
    print(f"Saved temperatures for shots {shots_start} onwards to calibration/trained_temperature.json")
    
if __name__ == '__main__':
    from utils import convert_to_list
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--llms', dest='llms', action='store', required=True, help='name of llms to evaluate the calibrator on')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of datasets to eval the calibrator on')    
    parser.add_argument('--feature_type', dest='feature_type', action='store', required=False, default="", help='the type of input features that make up the dataset')
    parser.add_argument('--shots_start', dest='shots_start', action='store', required=False, type=int, default=10, help='which shot # onwards we will do calibration for training and eval')
    parser.add_argument('--sampling_strategy', dest='sampling_strategy', action='store', required=False, default=None, help='what sampling strategy data to select(entropy vs similarity) (default: None - means select all)')
    parser.add_argument('--gpu_id', dest='gpu_id', action='store', default=0, required=False, help='Which CUDA gpu to run model on', type=int)

    args = parser.parse_args()
    args = vars(args)

    args['datasets'] = convert_to_list(args['datasets'])
    args['llms'] = convert_to_list(args['llms'])
    
    sampling_strategy = args.get('sampling_strategy')
    if sampling_strategy:
        args['sampling_strategy'] = sampling_strategy
    
    main(**args)