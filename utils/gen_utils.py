from copy import deepcopy
from enum import Enum
import os
from pathlib import Path
import pickle
from typing import Callable, Dict
from llm_framework import LlmFramework
import numpy as np
import torch
import random
from sampling_strategies import SamplingStrategy

ROOT_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT_DIR/"saved_results_final"

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print(f"Created {SAVE_DIR} for saving results")

class OverWriteType(Enum):
    FULL=0
    "Overwrite the saved reults with current results completely"
    LOAD=1
    "Load existing results, overwrite current results, save full results"
    SKIP=2
    "Skip seeds with saved results"

def normalize_distribution(p, epsilon=1e-9):
    """
    Normalize array to valid probability distribution.
    """
    p = np.asarray(p) + epsilon
    return p / np.sum(p)


def kl_divergence(p, q, epsilon=1e-9):
    """
    Compute KL divergence D_KL(P || Q).
    """
    p = normalize_distribution(p, epsilon)
    q = normalize_distribution(q, epsilon)
    
    return np.sum(p * np.log(p / q))


def js_divergence(p, q, epsilon=1e-9):
    """
    Compute Jensen-Shannon divergence between two probability distributions.
    """
    p = normalize_distribution(p, epsilon)
    q = normalize_distribution(q, epsilon)
    
    m = 0.5 * (p + q)
    
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def get_llm_framework(model_name: str):    
    if 'gemini' in model_name:
        llm_framework = LlmFramework.GOOGLE
    else:
        llm_framework = LlmFramework.VLLM

    return llm_framework

def load_pickle(params):
    # load saved results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")
    return data

def load_results(params_list):
    # load saved results from model
    result_tree = dict()
    for params in params_list:
        saved_result = load_pickle(params)
        match params['sampling_strategy']:
            case SamplingStrategy.ENTROPY:
                exp_setting = params['entropy_level']
            case SamplingStrategy.SIMILARITY:
                exp_setting = 'similarity_sampling'

        keys = [params['dataset'], params['model'], exp_setting, params['num_shots']]
        
        node = result_tree # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        node[params['seed']] = saved_result['accuracies']
    print_results(result_tree)
    
def print_results(tree, calibration_methods=[], log:Callable = print):
    calibration_methods = [method.name for method in calibration_methods]
    
    # print out all results
    root = deepcopy(tree)

    for dataset in root.keys():
        log(f"\n\nDataset: {dataset}")
        models_node = root[dataset]
        for model in models_node.keys():
            log(f"\nModel: {model}")
            entropy_node = models_node[model]
            for entropy_level in entropy_node.keys():
                log(f"\nEntropy level: {entropy_level}")
                num_shots_node = entropy_node[entropy_level]
                for num_shots in num_shots_node.keys():
                    seeds_results = num_shots_node[num_shots]
                    metrics = np.array(list(seeds_results.values()))
                    # print(dataset, model, entropy_level, num_shots_node, metrics); exit()
                    if len(metrics) == 0:
                        continue
                    metrics_mean = np.mean(metrics, axis=0)
                    metrics_low = np.min(metrics, axis=0)
                    metrics_high = np.max(metrics, axis=0)
                    metrics_std = np.std(metrics, axis=0)

                    # Determine metric names based on num_shots key
                    if isinstance(num_shots, str):
                        if 'ece' in num_shots:
                            names = ['Original ECE'] + [f'{method} ECE' for method in calibration_methods]
                        if 'mce' in num_shots:
                            names = ['Original MCE'] + [f'{method} MCE' for method in calibration_methods]
                        elif 'entropy' in num_shots:
                            names = ['Entropy'] + [f'{method} entropy' for method in calibration_methods]
                        elif 'conf' in num_shots:
                            names = ['Confidence'] + [f'{method} confidence' for method in calibration_methods]
                    else:
                        names = ['Original Accuracy'] + [f'{method} Accuracy' for method in calibration_methods]
                        log(f"\n{num_shots}-shot, {entropy_level} ICL entropy, {len(metrics)} seeds")
                    
                    # for aligned | char
                    max_len = max(len(name) for name in names)
                    names = [name + ' '*(max_len-len(name)) for name in names]

                    for i, (m, l, h, s) in enumerate(zip(metrics_mean, metrics_low, metrics_high, metrics_std)):
                        log(f"{names[i]} | Mean: {m:.4f}, Low: {l:.4f}, High: {h:.4f}, Std: {s:.4f}")
                    print()

    

def convert_to_list(items, cvt_func=None):
    if cvt_func:
        return [cvt_func(s.strip()) for s in items.split(",")]
    else:
        return [s.strip() for s in items.split(",")]

def get_saved_results_file_name(params: Dict):
        sampling_strategy = params['entropy_level'].name if params['sampling_strategy']==SamplingStrategy.ENTROPY else SamplingStrategy.SIMILARITY.name
        file_name = (f"{SAVE_DIR}/{params['model'].replace('/','_').replace('-FP8','')}/{params['dataset']}/" # In case it's an HF model
                 f"{sampling_strategy}/{params['num_shots']}_shot/{params['seed']}_seed.pkl") 
        
        return file_name
        
def fix_seed(seed):
    """For deterministic training."""
    random.seed(seed)
    np.random.seed(seed)

def setup_vllm_env_settings():    
    seed = 42 
    
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)  

    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_BATCH_INVARIANT'] = '1'
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"