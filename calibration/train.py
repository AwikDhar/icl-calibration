from collections import deque
import json 
import pprint
import random
from typing import Dict, List

import numpy as np
from tqdm import tqdm
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from calibration.model import (
    CalibrationTransformer, PositionEmbeddingType, CalibratorOutputType, 
    CalibrationLSTM, CalibrationRNN, 
    CalibrationMLP, CalibrationLogisticRegressor
)
from calibration.temperature import get_equivalent_temp, get_equivalent_temp_gd
from calibration.data_utils import fix_seed, load_datasets, get_batch, reshuffle_embeddings

from losses import BrierLoss
from utils.gen_utils import convert_to_list
# from utils.run_utils import fix_seed
from metrics import CalibrationMetrics
   
import logging

def setup_logger():
    logFormatter = logging.Formatter(
        "{asctime} - {levelname} - {message}", 
        style="{",
        datefmt="%Y-%m-%d %H:%M"
    )
    logger = logging.getLogger(__name__)

    fileHandler = logging.FileHandler("./cal_data_gen.log")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)

    return logger

def main(llms: List[str], 
         datasets: List[str],
         unseen_datasets: List[str],
         shots_start: int,
         shots_end: int,
         datasets_dropout: float,
         temp_augment: bool,
         label_augment: bool,
         volume_fraction: float,
         calibrator_type: str,
         calibrator_output_type: CalibratorOutputType,
         feature_type: str,
         sampling_strategies:List[str],
         iterations: int,
         lr: float,
         eval_iter: int,
         batch_size: int,
         ablation_method: str,
         num_seeds: int,
         seed_start: int,
         seed_train_ds_count: int | None,
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
    
    fix_seed(seed=42)
    data = load_datasets(llms, datasets, device, feature_type, splits=('train',), temp_augment=temp_augment, label_augment=label_augment, volume_fraction=volume_fraction, sampling_strategies=sampling_strategies, purpose="calibrator training")
    unseen_data = load_datasets(llms, unseen_datasets, device, feature_type, shots_end=shots_end, splits=('test',), temp_augment=False, label_augment=False, sampling_strategies=sampling_strategies, purpose="calibrator validation")
          
    for llm in llms:
        for dataset in unseen_datasets:
            if dataset not in data[llm]:
                data[llm][dataset] = unseen_data[llm][dataset]
            else:
                for sampling_strategy in sampling_strategies:
                    data[llm][dataset][sampling_strategy]['test'] = unseen_data[llm][dataset][sampling_strategy]['test']
        
    sample_model, sample_dataset, sample_strategy = llms[0], datasets[0], sampling_strategies[0]
    T, C = data[sample_model][sample_dataset][sample_strategy]['train']['inputs'][0].shape
    
    seed_train_ds_count = seed_train_ds_count or len(datasets)
    
    # seeds>1 means deterministic training and seed specific checkpoints
    for seed in range(seed_start, num_seeds):
        if num_seeds>1:
            fix_seed(seed)
        
        seed_train_ds = random.sample(datasets, seed_train_ds_count)
        # print(seed_train_ds_count); exit()
        model_dir = f"./calibration/models/"+ (llms[0].replace('/','_') if len(llms)==1 else 'llm_agnostic')
        if ablation_method:
            model_dir += f"/ablations/{ablation_method}"
        if len(sampling_strategies)==1:
            model_dir += f"/{sampling_strategies[0].lower()}"
        if num_seeds>1:
            model_dir += f"/{seed}_seed"
            
        model_path = f'{model_dir}/{model_name}'  
        metrics_path = f'{model_dir}/metrics.json'
        os.makedirs(model_dir, exist_ok=True)
        
        if os.path.exists(model_path):
            logging.info(f"Skipping seed {seed}, already done before.")
            continue
        
        calibrator = initialise_calibrator(calibrator_type, C, calibrator_output_type, device)
        
        print(calibrator)
        print(f"{sum(p.numel() for p in calibrator.parameters())/10**6: .2f} M parameters")
        # exit()
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
            calibrator_output_type,
            data,
            seed_train_ds,
            datasets_dropout,
            iterations,
            lr,
            eval_iter,
            batch_size,
            model_path,
            metrics_path,
            resume_saved_ckpt, 
            shots_start
        )

def train(model: nn.Module, 
          calibrator_output_type: CalibratorOutputType,
          data: Dict, 
          seed_train_ds: List[str],
          datasets_dropout: float,
          iterations: int, 
          lr: float,
          eval_iter: int,
          batch_size: int,
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
            best_eval_calibrated_brier = metrics['best_eval_calibrated_brier']
    else:     
        best_eval_calibrated_ece = torch.inf
        best_eval_calibrated_brier = torch.inf
  
    best_eval_loss = torch.inf    
    improved = False
    
    # temp_lambda = 0.5
    
    optimizer = get_optimizer(model)
    
    # gamma = 1.0 # Focal loss hyperparameter
    # brier_loss = BrierLoss(shots_start)
    
    datasets = []
    
    for llm in data:
        for dataset in seed_train_ds:
            for sampling_strategy in data[llm][dataset]:
                if 'train' not in data[llm][dataset][sampling_strategy]:
                    continue
                datasets.append((llm, dataset, sampling_strategy))
    
    grad_history = deque(maxlen=40000)
    
    for iter in tqdm(range(iterations), desc='Training calibrator'):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss = 0
        train_metrics = None
        
        active_count = round( (1-datasets_dropout)*len(datasets) )
        active_datasets = random.sample(datasets, active_count)
        
        for (llm, dataset, sampling_strategy) in active_datasets:
            inputs, logits, labels = get_batch(data[llm][dataset][sampling_strategy]['train'], batch_size) # B,T,C
                                    
            outputs = model(inputs) # B,T,1
            
            calibrated_logits, loss = get_calibrated_logits_loss(logits, outputs, calibrator_output_type, labels, shots_start)

            # Focal loss
            # calibrated_probs_flat = calibrated_pred_probs[:, shots_start:, :].reshape(-1)
            # correctness_flat = correctness_labels[:, shots_start:].reshape(-1)

            # pt = correctness_flat * calibrated_probs_flat + (1 - correctness_flat) * (1 - calibrated_probs_flat)
            # loss = ( - ((1-pt)**gamma) * torch.log(pt + 1e-8)).mean() 

            # Linearly increasing weights that sum to 1
            # num_positions = T - shots_start
            # shot_weights = torch.linspace(0.5, 1, num_positions, device=device) # increasing weight to higher shots to promote ICL
            # shot_weights = shot_weights / shot_weights.sum()  

            # shotwise_losses = loss.reshape(B, (T-shots_start))
            # loss = (shotwise_losses * shot_weights.unsqueeze(0)).sum(dim=1).mean() # weighted sum of shot losses 
            
            # pt = torch.exp(-ce_loss)
            # loss = (((1-pt)**gamma) * ce_loss).mean()#.reshape(B, T-shots_start)
            # loss = (shotwise_focal_loss * shot_weights.unsqueeze(0)).mean() # weighted sum of shot losses 
            
            total_loss += loss #+ temp_lambda * temp_regularization_loss
            # total_loss += brier_loss(calibrated_logits, labels) #+ temp_lambda*temp_regularizar_loss
                            
            if (iter+1)%eval_iter==0:
                with torch.inference_mode():
                    # fast approximation of ECE with binned=True, training dataset ECEs are not that important
                    cur_metrics = CalibrationMetrics(logits, calibrated_logits, labels, shots_start, binned=True)  
                    if not train_metrics:
                        train_metrics = cur_metrics
                    else:
                        train_metrics += cur_metrics                        
                 
        total_loss /= active_count
        
        total_loss.backward()
        
        obs_grad_norm = _get_grad_norm(model)
        grad_history.append(obs_grad_norm)
        clip_value = np.percentile(grad_history, 20)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        if (iter+1)%200==0:
            print(f'Step: {iter+1} | Train loss: {total_loss.item() : .4f}')
        
        if (iter+1)%eval_iter==0:
            train_metrics /= active_count
            
            print('|------EVAL------|')
            eval_loss, eval_metrics = eval(model, calibrator_output_type, data, shots_start, batch_size=None)
            print(f'Step: {iter+1} | Eval loss: {eval_loss.item() : .4f}\nTrain ECE: {train_metrics.ece : .4f}, Train calibrated ECE: {train_metrics.calibrated_ece : .4f}\n' 
                  +f'Eval ECE: {eval_metrics.ece : .4f}, Eval calibrated ECE: {eval_metrics.calibrated_ece : .4f}\n'
                  +f'Eval brier: {eval_metrics.brier_score : .4f}, Eval calibrated brier: {eval_metrics.calibrated_brier_score : .4f}\n'
            )           
            if eval_metrics.calibrated_ece<best_eval_calibrated_ece and eval_metrics.calibrated_brier_score<best_eval_calibrated_brier:
                if hasattr(model, '_orig_mod'):
                    torch.save(model._orig_mod.state_dict(), model_path)
                else:
                    torch.save(model.state_dict(), model_path)
                
                improved = True
                print(f"Saved improved model to {model_path}\n")
                best_eval_calibrated_ece = eval_metrics.calibrated_ece
                best_eval_calibrated_brier = eval_metrics.calibrated_brier_score
                
            if eval_loss<best_eval_loss:
                best_eval_loss = eval_loss.item()
                
        # if (iter+1)%5000:
        #     reshuffle_embeddings(data)

    print(f"\nBest CE loss: {best_eval_loss} \nBest eval calibrated ECE: {best_eval_calibrated_ece} \nBest eval calibrated brier: {best_eval_calibrated_brier}")
      
    if improved:     
        with open(metrics_path, 'w') as file:
            metrics = {
                "best_eval_loss": best_eval_loss,
                "best_eval_calibrated_ece":best_eval_calibrated_ece,
                "best_eval_calibrated_brier" : best_eval_calibrated_brier
            }
            json.dump(metrics, file)
      
def eval(model, calibrator_output_type, data, shots_start, batch_size):
    total_loss = 0
    metrics = None
    batch_size = None # ensure entire dataset eval for reliable checkpoints. change if needed
      
    num_datasets = 0
      
    model.eval()
    with torch.inference_mode():
        for llm in data:
            for dataset in data[llm]:
                for sampling_strategy in data[llm][dataset]:
                    if 'test' not in data[llm][dataset][sampling_strategy]:
                        continue
                    # print(dataset)
                    inputs, logits, labels = get_batch(data[llm][dataset][sampling_strategy]['test'], batch_size) # B,T,C | B,T,num_classes | B,T
                    # correctness_labels = (logits.argmax(dim=-1)==labels).float()
                    # num_classes = logits.shape[-1]
                                    
                    outputs = model(inputs) # B,T,1
                    calibrated_logits, loss = get_calibrated_logits_loss(logits, outputs, calibrator_output_type, labels, shots_start, num_iters=30) # defaulted to 10 till 15-05
                    
                    total_loss += loss
                    
                    B,T,num_classes = calibrated_logits.shape
                    
                    # Compute metrics per shot and average them
                    cur_metrics = None
                    for shot in range(shots_start, T):
                        shot_metrics = CalibrationMetrics(
                            logits[:, [shot], :], calibrated_logits[:, [shot], :], labels[:, [shot]], 
                            shots_start=0, binned=True
                        )
                        if cur_metrics is None:
                            cur_metrics = shot_metrics
                        else:
                            cur_metrics += shot_metrics
                    
                    # Average over shots
                    num_shots = T - shots_start
                    cur_metrics /= num_shots
                    
                    if not metrics:
                        metrics = cur_metrics
                    else:
                        metrics += cur_metrics
                        
                    num_datasets += 1
        
        total_loss /= num_datasets
        metrics /= num_datasets
        
        return total_loss, metrics
   
def get_calibrated_logits_loss(logits, outputs, calibrator_output_type, labels, shots_start, num_iters=15):
    if calibrator_output_type is CalibratorOutputType.CALIBRATED_PROBABILITY:
        return get_correctness_calibrated_logits_loss(logits, outputs, labels, shots_start, num_iters)

    if calibrator_output_type is CalibratorOutputType.TEMPERATURE:
        return get_temperature_calibrated_logits_loss(logits, outputs, labels, shots_start)

def get_correctness_calibrated_logits_loss(logits, calibrated_pred_probs, labels, shots_start, num_iters):
    correctness_labels = (logits.argmax(dim=-1)==labels).float()        
    # pred_probs, pred_classes = logits.max(dim=-1)  # (B, T, num_classes), (B, T)
    
    B,T,num_classes = logits.shape
    
    # Brier loss
    # loss = F.mse_loss(
    #     calibrated_pred_probs[:,shots_start:, :].reshape(B*(T-shots_start)), 
    #     correctness_labels[:,shots_start:].reshape(B*(T-shots_start)),
    #     # reduction='none'
    #     )
    loss = F.binary_cross_entropy(
        calibrated_pred_probs[:,shots_start:, :].reshape(B*(T-shots_start)), 
        correctness_labels[:,shots_start:].reshape(B*(T-shots_start)),
        # reduction='none'
    )
    
    # calibrated_pred_probs = torch.clamp(calibrated_pred_probs, min=1/num_classes + 0.01, max = 0.99)
    # remaining_prob_mass = 1.0 - calibrated_pred_probs # Shape: B, T, 1
    # remaining_prob_per_class = remaining_prob_mass / (num_classes - 1) 
    
    # # 4. Concatenate: [P'_max | P'_other, P'_other, ...]
    # # Initialize all probabilities to the uniform remaining probability
    # calibrated_probs = remaining_prob_per_class.repeat(1, 1, num_classes) # Shape: B, T, num_classes
    
    # # Set the predicted class to the calibrated probability
    # calibrated_probs.scatter_(dim=-1, index=pred_classes.unsqueeze(-1), 
    #                         src=calibrated_pred_probs)
    
    # calibrated_logits = torch.log(calibrated_probs + 1e-5)
    temperatures = get_equivalent_temp(logits, calibrated_pred_probs, num_iters=num_iters)  # B, T, 1
    # temperatures = get_equivalent_temp_gd(logits, calibrated_pred_probs)  # B, T, 1
    # temperatures = torch.nan_to_num(temperatures, nan=1.0)
    
    calibrated_logits = logits*temperatures
        
    return calibrated_logits, loss

def get_temperature_calibrated_logits_loss(logits, temperatures, labels, shots_start):
    B,T,num_classes = logits.shape

    calibrated_logits = logits*temperatures

    loss = F.cross_entropy(
        calibrated_logits[:,shots_start:, :].reshape(B*(T-shots_start), num_classes), 
        labels[:,shots_start:].reshape(B*(T-shots_start)),
        # reduction='none'
    )
    
    return calibrated_logits, loss
        
def _get_grad_norm(model: nn.Module):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 

def get_optimizer(model, learning_rate=1e-3, weight_decay_attn=0.00, weight_decay_mlp=0.05):
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
        {'params': other_params, 'weight_decay': 0.02},
    ]
    
    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)
    
    return optimizer

def initialise_calibrator(calibrator_type: str, in_features: int, calibrator_output_type: CalibratorOutputType, device: str) -> nn.Module:
    """Builds the calibrator model for the given architecture type.
    ...
    """
    calibrator_type = calibrator_type.lower()

    if calibrator_type=='transformer':
        with open(f"calibration/models/transformer_config.json", 'r') as file:
            config = json.load(file)

        calibrator = CalibrationTransformer(
            in_features=in_features, 
            context_length=config['context_length'], 
            embedding_dim=config['embedding_dim'], 
            num_heads=config['num_heads'], 
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            pos_embedding_type=PositionEmbeddingType.SINUSOIDAL,
            output_type=calibrator_output_type,
        )
    elif calibrator_type=='mlp':
        calibrator = CalibrationMLP(output_type=calibrator_output_type)
    elif calibrator_type=='rnn':
        calibrator = CalibrationRNN(in_features=in_features, output_type=calibrator_output_type)
    elif calibrator_type=='lstm':
        calibrator = CalibrationLSTM(in_features=in_features, output_type=calibrator_output_type)
    elif calibrator_type=='logistic':
        calibrator = CalibrationLogisticRegressor(output_type=calibrator_output_type)
    else:
        raise NotImplementedError(f"calibrator_type `{calibrator_type}` not available, please pick one from ['transformer', 'mlp', 'rnn', 'lstm', 'logistic']")

    return calibrator.to(device)

def args_check(args: Dict):
    assert args['iterations'] > 0, "iterations must be positive"
    assert args['lr'] > 0, "lr must be positive"
    assert args['batch_size'] > 0, "batch_size must be positive"

def set_torch_env():
    torch._dynamo.config.cache_size_limit = 64 
    torch.set_float32_matmul_precision('high') # better performance as per warning during torch.compile
    torch._dynamo.config.allow_rnn=True
    
if __name__ == '__main__':
    logger = setup_logger()
    set_torch_env()
    
    parser = argparse.ArgumentParser()
    # core arguments
    parser.add_argument('--llms', dest='llms', action='store', required=True, help='name of llm(s) to train the calibrator for')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset to train the calibrator for')
    parser.add_argument('--unseen_datasets', action='store', required=True, help='name of dataset to eval the calibrator on')
    parser.add_argument('--shots_end', dest='shots_end', action='store', required=False, type=int, default=None, help='Till which shot # we will do calibration eval')
    parser.add_argument('--datasets_dropout', dest='datasets_dropout', action='store', required=False, default=0.8, help='fraction of datasets randomly dropped out each training iteration')
    parser.add_argument('--feature_type', dest='feature_type', action='store', required=False, default="", help='the type of input features that make up the dataset')
    parser.add_argument('--sampling_strategies', action='store', required=False, default=None, help='what sampling strategy data to select(entropy vs similarity) (default: None - means select all)')
    parser.add_argument('--shots_start', dest='shots_start', action='store', required=True, type=int, default=None, help='which shot # onwards we will do calibration for training and eval')
    parser.add_argument('--temp_augment', dest='temp_augment', action='store_const', const=True, default=False,
                        help="Whether or not to randomly temperature scale data logits(and affect features) for robust training")
    parser.add_argument('--label_augment', dest='label_augment', action='store_const', const=True, default=False,
                        help="Whether or not to randomly sample synthetic labels from a temp scaled prob distribution for accuracy variation/robust training")
    parser.add_argument('--volume_fraction', action='store', default=1, type=float,
                        help="What fraction of the dataset to load and use for training")
    
    
    # general training args
    parser.add_argument('--iterations', dest='iterations', action='store', required=False, type=int, default=20000 , help='number of iterations for training')
    parser.add_argument('--lr', dest='lr', action='store', required=False, type=float, default=1e-4, help='learning rate')
    parser.add_argument('--eval_iter', dest='eval_iter', action='store', required=False, type=int, default=400, help='number of iterations after which to do eval since start/last eval')
    parser.add_argument('--batch_size', dest='batch_size', action='store', required=False, type=int, default=32,
                        help='batch size for model training')
    
    # ablation related
    parser.add_argument('--calibrator_type', action='store', required=False, default='transformer',
                        choices=['transformer', 'mlp', 'rnn', 'lstm', 'logistic'], help='which calibrator architecture to train')
    parser.add_argument('--calibrator_output_type', action='store', required=False, default="calibrated_probability", help='What the transformer calibrator outputs(temperature/calibrated_probability)')
    parser.add_argument('--ablation_method', action='store', required=False, default=None, help='Ablation method name, if performing ablation')
    parser.add_argument('--num_seeds', action='store', required=False, default=1, type=int, help='Number of seeds to train calibrators for')
    parser.add_argument('--seed_start', action='store', required=False, default=0, type=int, help='Which seed # to start training from')
    parser.add_argument('--seed_train_ds_count', action='store', required=False, default=None, type=int, help='Number of datasets to train given seed with')
    
    # other args
    parser.add_argument('--model_name', dest='model_name', action='store', required=False, default="calibrator", help='custom name for the model(calibrator), used for saving the checkpoints')    
    parser.add_argument('--resume_saved_ckpt', dest='resume_saved_ckpt', action='store_const', const=True, default=False, help='whether to resume training from saved model')
    parser.add_argument('--gpu_id', dest='gpu_id', action='store', default=0, required=False, help='Which CUDA gpu to run model on', type=int)
    
    args = parser.parse_args()
    args = vars(args)
    args['llms'] = convert_to_list(args['llms'])
    args['datasets'] = convert_to_list(args['datasets'])
    if args.get('unseen_datasets'):
        args['unseen_datasets'] = convert_to_list(args['unseen_datasets'])
    else:
        logger.warning("Using the training datasets as the unseen/validation datasets")
        args['unseen_datasets'] = args['datasets']
        
    args['calibrator_output_type'] = CalibratorOutputType[args['calibrator_output_type'].upper()]
    if args.get('sampling_strategies'): 
        args['sampling_strategies'] = convert_to_list(args['sampling_strategies'], lambda s: s.upper())
    else:
        args['sampling_strategies'] = ['ENTROPY', 'SIMILARITY']
        # args['sampling_strategies'] = ['SIMILARITY', 'ENTROPY']
        
    args_check(args)
    pprint.pprint(args)
         
    main(**args)