from collections import deque
import json 
import random
from typing import Dict, List

import numpy as np
from tqdm import tqdm
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from calibration.model import CalibrationTransformer, PositionEmbeddingType
from calibration.temperature import get_equivalent_temp
from calibration.data_utils import load_datasets, get_batch

from losses import BrierLoss
from utils import convert_to_list
from metrics import Metrics
   
def main(llms: List[str], 
         datasets: List[str],
         shots_start: int,
         datasets_dropout: float,
         temp_augment: bool,
         label_augment: bool,
         feature_type: str,
         sampling_strategy:str,
         iterations: int,
         lr: float,
         eval_iter: int,
         batch_size: int,
         model_name: str,
         resume_saved_ckpt: bool,
         gpu_id: int,
    ):
    """_summary_

    Args:
        models (nn.Module): LLM(s) to calibrate 
        datasets (List[str]): calibration datasets
        datasets_dropout (float): what fraction of datasets to drop out(remaining will be used for training on a given iter)
        iterations (int): number of training iterations
        lr (float): learning rate
        eval_iter (int): what iteration intervals to do eval on
        batch_size (int): training batch size
        temp_augment (bool): whether to apply random temperature scaling to data before training
        feature_type (str): the type of features for the transformer . Class agnostic is the only option for now
        resume_saved_ckpt (bool): whether to resume training from saved best ckpt
        gpu_id (int): GPU id to keep all GPU operations on
        shots_start (int): what shots onwards to consider loss for backprop(earlier ones will be ignored for loss) 
    """    
    
    device = f'cuda:{gpu_id}'
    
    data = load_datasets(llms, datasets, device, feature_type, temp_augment=temp_augment, label_augment=label_augment, sampling_strategy=sampling_strategy)
                
    model_dir = f"./calibration/models/"+ (llms[0].replace('/','_') if len(llms)==1 else 'llm_agnostic')
    if sampling_strategy:
        model_dir += f"/{sampling_strategy}"
    model_path = f'{model_dir}/{model_name}'  
    metrics_path = f'{model_dir}/metrics.json'
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f"calibration/models/transformer_config.json", 'r') as file:
        config = json.load(file)
        
    sample_model, sample_dataset = llms[0], datasets[0]
    T, C = data[sample_model][sample_dataset]['train']['inputs'][0].shape
    calibrator = CalibrationTransformer(
        in_features=C, 
        context_length=T, 
        embedding_dim=config['embedding_dim'], 
        num_heads=config['num_heads'], 
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        pos_embedding_type=PositionEmbeddingType.Sinusoidal
    ).to(device)
    
    print(calibrator)
    print(f"{sum(p.numel() for p in calibrator.parameters())/10**6: .2f} M parameters")
    
    if resume_saved_ckpt:
        state_dict = torch.load(model_path, weights_only=True)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                cleaned_state_dict[key.replace('_orig_mod.', '')] = value
            else:
                cleaned_state_dict[key] = value
                
        calibrator.load_state_dict(cleaned_state_dict)    
                    
    calibrator = torch.compile(calibrator, fullgraph=True, dynamic=False, mode="max-autotune")
    
    # print(model_path); exit()
    train(
        calibrator, 
        data,
        datasets_dropout,
        iterations,
        lr,
        eval_iter,
        batch_size,
        device,
        model_path,
        metrics_path,
        resume_saved_ckpt, 
        shots_start
    )

def train(model: nn.Module, 
          data: Dict, 
          datasets_dropout: float,
          iterations: int, 
          lr: float,
          eval_iter: int,
          batch_size: int,
          device: str,
          model_path: str,
          metrics_path: str,
          resume_saved_ckpt: bool,
          shots_start: int):
    """_summary_

    Args:
        model (nn.Module): the calibrator
        data (Dict): dataset dict
        datasets_dropout (float): _description_
        iterations (int): number of training iterations
        lr (float): learning rate
        eval_iter (int): _description_
        batch_size (int): training abtch size
        device (str): torch device
        model_path (str): model save path
        metrics_path (str): path to save ckpt's eval metrics
        resume_saved_ckpt (bool): whether to resume training from saved best ckpt
        gpu_id (int): GPU id to keep all GPU operations on
        shots_start (int): what shots onwards to consider loss for backprop(earlier ones will be ignored for loss calculation) 
    """    
    if resume_saved_ckpt and os.path.exists(metrics_path):
        with open(metrics_path, 'r') as file:
            metrics = json.load(file)
            best_eval_calibrated_ece = metrics['best_eval_calibrated_ece']
    else:     
        best_eval_calibrated_ece = torch.inf
    
    ckpt_eval_brier_score = torch.inf
    ckpt_eval_ce_loss = torch.inf   
    best_eval_loss = torch.inf
    
    improved = False
    
    # temp_lambda = 0.5
    
    optimizer = get_optimizer(model)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # scaler = torch.amp.GradScaler()
    
    gamma = 1.0 # Focal loss hyperparameter
    brier_loss = BrierLoss(shots_start)
    
    datasets = []
    
    for llm in data:
        for dataset in data[llm]:
            datasets.append((llm, dataset))
    
    grad_history = deque(maxlen=40000)
    
    for iter in tqdm(range(iterations), desc='Training calibrator'):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss = 0
        train_metrics = None
        
        active_count = round( (1-datasets_dropout)*len(datasets) )
        active_datasets = random.sample(datasets, active_count)
        
        for (llm, dataset) in active_datasets:
            inputs, logits, labels = get_batch(data[llm][dataset]['train'], batch_size, device) # B,T,C
            correctness_labels = (logits.argmax(dim=-1)==labels).float()
            B,T,num_classes = logits.shape
                        
            # with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            pred_probs, pred_classes = logits.max(dim=-1)  # (B, T, num_classes), (B, T)
            
            calibrated_pred_probs = model(inputs) # B,T,1
            # Focal loss
            # calibrated_probs_flat = calibrated_pred_probs[:, shots_start:, :].reshape(-1)
            # correctness_flat = correctness_labels[:, shots_start:].reshape(-1)

            # pt = correctness_flat * calibrated_probs_flat + (1 - correctness_flat) * (1 - calibrated_probs_flat)
            # loss = ( - ((1-pt)**gamma) * torch.log(pt + 1e-8)).mean() 
            
            # Brier loss
            # loss = F.mse_loss(
            #     calibrated_pred_probs[:,shots_start:, :].reshape(B*(T-shots_start)), 
            #     correctness_labels[:,shots_start:].reshape(B*(T-shots_start))
            #     )
            loss = F.binary_cross_entropy(
                calibrated_pred_probs[:,shots_start:, :].reshape(B*(T-shots_start)), 
                correctness_labels[:,shots_start:].reshape(B*(T-shots_start)),
            )  
                        
            # shotwise losses
            # ce_loss = F.binary_cross_entropy(
            #     calibrated_pred_probs[:,shots_start:].reshape(B*(T-shots_start)), 
            #     correctness_labels[:,shots_start:].reshape(B*(T-shots_start)),
            #     reduction='none'
            # )  

            # Linearly increasing weights that sum to 1
            # num_positions = T - shots_start
            # shot_weights = torch.linspace(1, num_positions, num_positions, device=device) # increasing weight to higher shots to promote ICL
            # shot_weights = shot_weights / shot_weights.sum()  

            # ce_loss = (shotwise_losses * shot_weights.unsqueeze(0))#.mean() # weighted sum of shot losses 
            
            calibrated_pred_probs = torch.clamp(calibrated_pred_probs, min=1/num_classes + 0.01, max = 0.99)
            remaining_prob_mass = 1.0 - calibrated_pred_probs # Shape: B, T, 1
            remaining_prob_per_class = remaining_prob_mass / (num_classes - 1) 
            
            # 4. Concatenate: [P'_max | P'_other, P'_other, ...]
            # Initialize all probabilities to the uniform remaining probability
            calibrated_probs = remaining_prob_per_class.repeat(1, 1, num_classes) # Shape: B, T, num_classes
            
            # Set the predicted class to the calibrated probability
            calibrated_probs.scatter_(dim=-1, index=pred_classes.unsqueeze(-1), 
                                    src=calibrated_pred_probs)
            
            calibrated_logits = torch.log(calibrated_probs + 1e-5)
            # # calibrated_logits = logits*temperatures # B,T,num_classes

            # pt = torch.exp(-ce_loss)
            # loss = (((1-pt)**gamma) * ce_loss).mean()#.reshape(B, T-shots_start)
            # loss = (shotwise_focal_loss * shot_weights.unsqueeze(0)).mean() # weighted sum of shot losses 
            
            total_loss += loss #+ temp_lambda * temp_regularization_loss
            # total_loss += brier_loss(calibrated_logits, labels) #+ temp_lambda*temp_regularizar_loss
                            
            if (iter+1)%eval_iter==0:
                with torch.inference_mode():
                    cur_metrics = Metrics(logits, calibrated_logits, labels, shots_start)
                    if not train_metrics:
                        train_metrics = cur_metrics
                    else:
                        train_metrics += cur_metrics                        
                 
        total_loss /= active_count
        
        total_loss.backward()
        # scaler.scale(total_loss).backward()    
        # scaler.unscale_(optimizer)
        
        obs_grad_norm = _get_grad_norm(model)
        grad_history.append(obs_grad_norm)
        clip_value = np.percentile(grad_history, 20)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        # scaler.step(optimizer)
        # scaler.update()
        optimizer.step()
        
        if (iter+1)%200==0:
            print(f'Step: {iter+1} | Train loss: {total_loss.item() : .4f}')
        
        if (iter+1)%eval_iter==0:
            train_metrics /= active_count
            
            eval_loss, eval_metrics = eval(model, data, shots_start, None, device)
            print('|------EVAL------|')
            print(f'Step: {iter+1} | Eval loss: {eval_loss.item() : .4f}\nTrain ECE: {train_metrics.ece : .4f}, Train calibrated ECE: {train_metrics.calibrated_ece : .4f}\n' 
                  +f'Eval ECE: {eval_metrics.ece : .4f}, Eval calibrated ECE: {eval_metrics.calibrated_ece : .4f}\n'
                  +f'Eval brier: {eval_metrics.brier_score : .4f}, Eval calibrated brier: {eval_metrics.calibrated_brier_score : .4f}\n'
            )           
            if eval_metrics.calibrated_ece<best_eval_calibrated_ece:
                if hasattr(model, '_orig_mod'):
                    torch.save(model._orig_mod.state_dict(), model_path)
                else:
                    torch.save(model.state_dict(), model_path)
                
                improved = True
                print("Saved improved model\n")
                best_eval_calibrated_ece = eval_metrics.calibrated_ece
                ckpt_eval_brier_score = eval_metrics.calibrated_brier_score
                
            if eval_loss<best_eval_loss:
                best_eval_loss = eval_loss.item()

    print(f"Best eval calibrated ECE: {best_eval_calibrated_ece}, best CE loss: {best_eval_loss}")
      
    if improved:     
        with open(metrics_path, 'w') as file:
            metrics = {
                "best_eval_loss": best_eval_loss,
                "best_eval_calibrated_ece":best_eval_calibrated_ece,
                "ckpt_eval_brier_score" : ckpt_eval_brier_score
            }
            json.dump(metrics, file)
      
def eval(model, data, shots_start, batch_size, device):
    total_loss = 0
    metrics = None
    batch_size = None # ensure entire dataset eval for reliable checkpoints. change if needed
      
    num_datasets = 0
      
    model.eval()
    with torch.inference_mode():
        for llm in data:
            for dataset in data[llm]:
                inputs, logits, labels = get_batch(data[llm][dataset]['test'], batch_size, device) # B,T,C | B,T,num_classes | B,T
                correctness_labels = (logits.argmax(dim=-1)==labels).float()
                num_classes = logits.shape[-1]
                
                pred_classes = logits.argmax(dim=-1)  # B, T
                
                calibrated_pred_probs = model(inputs) # B,T,1
                calibrated_pred_probs = torch.clamp(calibrated_pred_probs, min=1/num_classes + 0.01, max = 0.99)
                
                temperatures = get_equivalent_temp(logits, calibrated_pred_probs)  # B, T, 1
                calibrated_logits = logits*temperatures

                B,T,num_classes = calibrated_logits.shape
                # loss = F.cross_entropy(calibrated_logits[:,shots_start:,:].reshape(B*(T-shots_start), num_classes), 
                #                         labels[:,shots_start:].reshape(B*(T-shots_start)))
                loss = F.binary_cross_entropy(
                    calibrated_pred_probs[:,shots_start:].reshape(B*(T-shots_start)), 
                    correctness_labels[:,shots_start:].reshape(B*(T-shots_start)))
                
                total_loss += loss
                
                cur_metrics = Metrics(logits, calibrated_logits, labels, shots_start)
                if not metrics:
                    metrics = cur_metrics
                else:
                    metrics += cur_metrics
                    
                num_datasets += 1
        
        total_loss /= num_datasets
        metrics /= num_datasets
        
        return total_loss, metrics
    
def _get_grad_norm(model: nn.Module):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 

def get_optimizer(model, learning_rate=1e-3, weight_decay_attn=0.01, weight_decay_mlp=0.05):
    """
    Get an optimizer with model layer/module specific weight decay
    """
    
    attn_params = []
    mlp_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'self_attn' in name:
            attn_params.append(param)
        elif 'linear1' in name or 'linear2' in name:
            mlp_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {'params': attn_params, 'weight_decay': weight_decay_attn},
        {'params': mlp_params, 'weight_decay': weight_decay_mlp},
        {'params': other_params, 'weight_decay': 0.00},
    ]
    
    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)
    
    return optimizer

def args_check(args: Dict):
    assert args['iterations'] > 0, "iterations must be positive"
    assert args['lr'] > 0, "lr must be positive"
    assert args['batch_size'] > 0, "batch_size must be positive"

def set_torch_env():
    torch._dynamo.config.cache_size_limit = 32 
    torch.set_float32_matmul_precision('high') # better performance as per warning during torch.compile
    
if __name__ == '__main__':
    set_torch_env()
    
    parser = argparse.ArgumentParser()
    # core arguments
    parser.add_argument('--llms', dest='llms', action='store', required=True, help='name of llm(s) to train the calibrator for')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset to train the calibrator for')
    parser.add_argument('--datasets_dropout', dest='datasets_dropout', action='store', required=False, default=0.8, help='fraction of datasets randomly dropped out each training iteration')
    parser.add_argument('--feature_type', dest='feature_type', action='store', required=False, default="", help='the type of input features that make up the dataset')
    parser.add_argument('--sampling_strategy', dest='sampling_strategy', action='store', required=False, default=None, help='what sampling strategy data to select(entropy vs similarity) (default: None - means select all)')
    parser.add_argument('--shots_start', dest='shots_start', action='store', required=False, type=int, default=2, help='which shot # onwards we will do calibration for training and eval')
    parser.add_argument('--temp_augment', dest='temp_augment', action='store_const', const=True, default=False,
                        help="Whether or not to randomly temperature scale data logits(and affect features) for robust training")
    parser.add_argument('--label_augment', dest='label_augment', action='store_const', const=True, default=False,
                        help="Whether or not to randomly sample synthetic labels from a temp scaled prob distribution for accuracy variation/robust training")
    
    # general training args
    parser.add_argument('--iterations', dest='iterations', action='store', required=False, type=int, default=20000 , help='number of iterations for training')
    parser.add_argument('--lr', dest='lr', action='store', required=False, type=float, default=1e-4, help='learning rate')
    parser.add_argument('--eval_iter', dest='eval_iter', action='store', required=False, type=int, default=400, help='number of iterations after which to do eval since start/last eval')
    parser.add_argument('--batch_size', dest='batch_size', action='store', required=False, type=int, default=32,
                        help='batch size for model training')
    # other args
    parser.add_argument('--model_name', dest='model_name', action='store', required=False, default="calibrator", help='custom name for the model(calibrator), used for saving the checkpoints')    
    parser.add_argument('--resume_saved_ckpt', dest='resume_saved_ckpt', action='store_const', const=True, default=False, help='whether to resume training from saved model')
    parser.add_argument('--gpu_id', dest='gpu_id', action='store', default=0, required=False, help='Which CUDA gpu to run model on', type=int)
    
    args = parser.parse_args()
    args = vars(args)
            
    args['datasets'] = convert_to_list(args['datasets'])
    args['llms'] = convert_to_list(args['llms'])

    args_check(args)
    main(**args)