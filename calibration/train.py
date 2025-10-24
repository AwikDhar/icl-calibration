from collections import deque
import json
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from calibration.model import CalibrationTransformer
from losses import ECELoss
from time import time
from utils import convert_to_list

def recalculate_features(item: Dict):
    # print(item['inputs'])
    item['inputs'][0][2] = -1
    temp = torch.empty(1).uniform_(0.5,2).item()
    item['logits'] /= temp
    
    num_classes = item['logits'].shape[-1]
    probs = F.softmax(item['logits'], dim=-1)
        
    preds = torch.argmax(probs, dim=-1)
    pred_probs = probs[torch.arange(len(preds)), preds]
    shifted_gt_probs = torch.concatenate((
        torch.tensor([-1]), 
        probs[torch.arange(len(preds)-1), item['labels'][:-1]]
    )) # exclude last position and shift

    # normalized_entropies = np.array([
    #     probs[sent_idx]@np.log(probs[sent_idx].T) 
    #     for sent_idx in range(len(sentences))
    # ]) / np.log(num_classes)
    normalized_entropies = -(probs@torch.log(probs.T + 1e-9)).diagonal() / torch.log(torch.tensor(num_classes)) # (B, num_classes) × (num_classes, B) --> (B, B) --> diagonal elements

    item['inputs'][:,0] = pred_probs
    item['inputs'][:,2] = shifted_gt_probs
    item['inputs'][:,3] = normalized_entropies
    # print(item['inputs'])
    # exit()
 
def load_datasets(model: str, datasets: list, device: str, feature_type: str, splits = ('train', 'test'), sampling_strategy = None, apply_features: bool = True):
    """Preload all datasets into GPU memory for fast training (with optional feature recalculation)."""
    data = {}

    for dataset in datasets:
        data[dataset] = {}
        for split in splits:
            path = f"calibration/datasets/{model.replace('/','_')}/{dataset}/{feature_type}/{split}.json"
            with open(path) as file:
                split_data = json.load(file)
                if sampling_strategy:
                    split_data = [item for item in split_data if item['sampling_strategy']==sampling_strategy.upper()] 

            # convert each sample into tensors and optionally recalc features
            for idx in range(len(split_data)):
                split_data[idx]["inputs"] = torch.tensor(split_data[idx]["inputs"], dtype=torch.float32)
                split_data[idx]["logits"] = torch.tensor(split_data[idx]["logits"], dtype=torch.float32)
                split_data[idx]["labels"] = torch.tensor(split_data[idx]["labels"], dtype=torch.long)

                # if apply_features:
                #     recalculate_features(split_data[idx])

            # batchify entire dataset
            inputs_all = torch.stack([item["inputs"] for item in split_data])
            logits_all = torch.stack([item["logits"] for item in split_data])
            labels_all = torch.stack([item["labels"] for item in split_data])

            # push to GPU
            data[dataset][split] = {
                "inputs": inputs_all.to(device, non_blocking=True),
                "logits": logits_all.to(device, non_blocking=True),
                "labels": labels_all.to(device, non_blocking=True),
            }

    return data
   
def main(model, datasets, iterations, lr, eval_iter, batch_size, temp_augment, feature_type, gpu_id):
    device = f'cuda:{gpu_id}'
    
    data = load_datasets(model, datasets, device, feature_type, apply_features=temp_augment)
                
    model_dir = f"./calibration/models/{model.replace('/','_')}/{'_'.join(datasets)}"
    model_path = f'{model_dir}/calibrator'  
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f"calibration/models/transformer_config.json", 'r') as file:
        config = json.load(file)
        
    sample_dataset = datasets[0]
    T, C = data[sample_dataset]['train']['inputs'][0].shape
    calibrator = CalibrationTransformer(
        in_features=C, 
        context_length=config['context_length'], 
        embedding_dim=config['embedding_dim'], 
        num_heads=config['num_heads'], 
        num_layers=config['num_layers']
    ).to(device)
    
    calibrator = torch.compile(calibrator, fullgraph=True, dynamic=False, mode="max-autotune")
    
    train(calibrator, data, iterations, lr, eval_iter, batch_size, device, model_path)

def train(model: nn.Module, 
          data: Dict, 
          iterations: int, 
          lr: float,
          eval_iter: int,
          batch_size: int,
          device: str,
          model_path: str):
    
    best_eval_calibrated_ece = torch.inf
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # scaler = torch.amp.GradScaler()
    
    ece_loss = ECELoss(n_bins=30)
    datasets = data.keys()
    
    grad_history = deque(maxlen=20000)
    
    model.train()
    for iter in tqdm(range(iterations), desc='Training calibrator'):
        optimizer.zero_grad(set_to_none=True)
        
        total_loss = 0
        train_ece = 0
        train_calibrated_ece = 0
        
        for dataset in datasets:
            inputs, logits, labels = get_batch(data[dataset]['train'], batch_size, device) # B,T,C

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                temperatures = model(inputs) # B,T,1
                calibrated_logits = logits*temperatures # B,T,num_classes

                B,T,num_classes = calibrated_logits.shape
                loss = F.cross_entropy(calibrated_logits.view(B*T, num_classes), labels.view(B*T))
                total_loss += loss
                            
        total_loss /= len(datasets)
        train_ece /= len(datasets)
        train_calibrated_ece /= len(datasets)
        
        total_loss.backward()
        # scaler.scale(total_loss).backward()    
        # scaler.unscale_(optimizer)
        
        obs_grad_norm = _get_grad_norm(model)
        grad_history.append(obs_grad_norm)
        clip_value = np.percentile(grad_history, 10)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        # scaler.step(optimizer)
        # scaler.update()
        optimizer.step()
        
        if (iter+1)%100==0:
            print(f'Step: {iter+1} | Train loss: {loss.item() : .2f}')
        
        if (iter+1)%eval_iter==0:
            train_ece += ece_loss(logits, labels)
            train_calibrated_ece += ece_loss(calibrated_logits, labels)
            
            eval_loss, eval_ece, eval_calibrated_ece = eval(model, data, batch_size, device, ece_loss)
            print('|------EVAL------|')
            print(f'Step: {iter+1} | Eval loss: {eval_loss.item() : .2f}\nTrain ECE: {train_ece.item() : .4f}, Train calibrated ECE: {train_calibrated_ece.item() : .4f}, Eval ECE: {eval_ece.item() : .4f}, Eval calibrated ECE: {eval_calibrated_ece.item() : .4f}\n')
            if eval_calibrated_ece<best_eval_calibrated_ece:
                if hasattr(model, '_orig_mod'):
                    torch.save(model._orig_mod.state_dict(), model_path)
                else:
                    torch.save(model.state_dict(), model_path)
                print("Saved improved model\n")
                
                best_eval_calibrated_ece = eval_calibrated_ece
                
def eval(model, data, batch_size, device, ece_loss):
    total_loss = 0
    ece = 0
    calibrated_ece = 0
    
    datasets = data.keys()
    
    model.eval()
    with torch.inference_mode():
        for dataset in datasets:
            inputs, logits, labels = get_batch(data[dataset]['test'], None, device) # B,T,C | B,T,num_classes | B,T
        
            temperatures = model(inputs) # B,T,1
            calibrated_logits = logits*temperatures # B,T,num_classes
            
            B,T,num_classes = calibrated_logits.shape
            loss = F.cross_entropy(calibrated_logits.view(B*T, num_classes), labels.view(B*T))        
            total_loss += loss
            
            ece += ece_loss(logits, labels)
            calibrated_ece += ece_loss(calibrated_logits, labels)
        
        total_loss /= len(datasets)
        ece /= len(datasets)
        calibrated_ece /= len(datasets)
        
        return total_loss, ece, calibrated_ece
    
def get_batch(data: Dict, batch_size: Optional[int] = None, device: str = "cuda:0"):
    N = data["inputs"].shape[0] # dataset size
    if batch_size is None:
        return (
        data["inputs"],
        data["logits"],
        data["labels"],
    )
        
    idxs = torch.randint(N, (batch_size,), device=device)
    return (
        data["inputs"][idxs],
        data["logits"][idxs],
        data["labels"][idxs],
    )

def _get_grad_norm(model: nn.Module):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 

def args_check(args: Dict):
    assert args['iterations'] > 0, "iterations must be positive"
    assert args['lr'] > 0, "lr must be positive"
    assert args['batch_size'] > 0, "batch_size must be positive"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--model', dest='model', action='store', required=True, help='name of model to train the calibrator for')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset to train the calibrator for')
    parser.add_argument('--feature_type', dest='feature_type', action='store', required=False, default="class_agnostic", help='the type of input features that make up the dataset')
    parser.add_argument('--iterations', dest='iterations', action='store', required=False, type=int, default=20000 , help='number of iterations for training')
    parser.add_argument('--lr', dest='lr', action='store', required=False, type=float, default=1e-4, help='learning rate')
    parser.add_argument('--eval_iter', dest='eval_iter', action='store', required=False, type=int, default=200, help='number of iterations after which to do eval since start/last eval')
    parser.add_argument('--batch_size', dest='batch_size', action='store', required=False, type=int, default=32,
                        help='batch size for model training')
    parser.add_argument('--temp_augment', dest='temp_augment', action='store', required=False, type=bool, default=True,
                        help="Whether or not to randomly temperature scale data logits(and affect features) for robust training")
    
    parser.add_argument('--gpu_id', dest='gpu_id', action='store', default=0, required=False, help='Which CUDA gpu to run model on', type=int)
    
    args = parser.parse_args()
    args = vars(args)
            
    args['datasets'] = convert_to_list(args['datasets'])

    args_check(args)
    main(**args)