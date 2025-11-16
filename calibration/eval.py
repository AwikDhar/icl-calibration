import argparse
import json
import os
from typing import Dict

import torch
import torch.nn.functional as F
import numpy as np

from calibration.model import CalibrationTransformer
from metrics import Metrics
from calibration_plot_data import CalibrationPlotData
from calibration.train import get_batch, load_datasets, convert_to_list
from utils import convert_to_list
from plot_results.plot_tc import plot_calibration

def main(models, datasets, feature_type, shots_start, sampling_strategy = None, gpu_id=0, model_path=None, llm_agnostic=False, save_path=None):
    device = f'cuda:{gpu_id}'
    
    data = load_datasets(models, datasets, device, feature_type, splits=('test',), sampling_strategy=sampling_strategy, temp_augment=True)
    
    with open(f"calibration/models/transformer_config.json", 'r') as file:
        config = json.load(file)
    
    if model_path is None:
        if llm_agnostic or len(models)>1:
            model_dir = f"./calibration/models/llm_agnostic"
        else:
            model_dir = f"./calibration/models/{models[0].replace('/','_')}"
            
        model_path = f'{model_dir}/calibrator'

    sample_model, sample_dataset = models[0], datasets[0]
    T, C = data[sample_model][sample_dataset]['test']['inputs'][0].shape
    calibrator = CalibrationTransformer(
    in_features=C, 
        context_length=config['context_length'], 
        embedding_dim=config['embedding_dim'], 
        num_heads=config['num_heads'], 
        num_layers=config['num_layers']
    ).to(device)
    # print(calibrator)

    state_dict = torch.load(model_path, weights_only=True)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            cleaned_state_dict[key.replace('_orig_mod.', '')] = value
        else:
            cleaned_state_dict[key] = value

    calibrator.load_state_dict(cleaned_state_dict)
          
    for llm in models:
        for dataset in datasets:
            calibration_data = eval(calibrator, data, llm, dataset, shots_start, device)
            
            plot_calibration(calibration_data, llm, dataset, feature_type, llm_agnostic, sampling_strategy, save_path=save_path)
    
def eval(model, data, llm, dataset, shots_start, device):
    calibration_data = CalibrationPlotData()
    
    model.eval()
    with torch.no_grad():
        inputs, logits, labels = get_batch(data[llm][dataset]['test'], batch_size=None, device=device) # len(eval),T,C | len(eval),T,num_classes | len(eval),T
        
        temperatures = model(inputs) # B,T,1
        temperatures = torch.nan_to_num(temperatures, nan=1.0)

        calibrated_logits = logits*temperatures # B,T,num_classes
        
        B,T,num_classes = calibrated_logits.shape
        # loss = F.cross_entropy(calibrated_logits.view(B*T, num_classes), labels.view(B*T))        
        
        # print(calibrated_logits.shape, logits.shape, labels.shape, temperatures.shape)
        # exit()
        probs, preds = F.softmax(logits, dim=-1).max(dim=-1)
        calibrated_probs, calibrated_preds = F.softmax(calibrated_logits, dim=-1).max(dim=-1)

        print(f"|------Dataset: {dataset}------|")
        for shot in range(shots_start, T):
            eval_metrics = Metrics(logits[:, [shot], :], calibrated_logits[:, [shot], :], labels[:, [shot]], 0)
     
            calibration_data.ece_shots_map['original'][shot] = eval_metrics.ece
            calibration_data.ece_shots_map['calibrated'][shot] = eval_metrics.calibrated_ece
            
            calibration_data.brier_shots_map['original'][shot] = eval_metrics.brier_score
            calibration_data.brier_shots_map['calibrated'][shot] = eval_metrics.calibrated_brier_score
            
            calibration_data.reldiag_shots_map['original'][shot] = eval_metrics.rel_diag
            calibration_data.reldiag_shots_map['calibrated'][shot] = eval_metrics.calibrated_rel_diag
            
            calibration_data.temp_shots_map[shot] = temperatures[:,shot,:].flatten().cpu().numpy()

            calibration_data.accuracies.append((labels==preds).cpu()[:, shot].sum()/len(preds))
            calibration_data.conf_shots_map['original'][shot] = np.ma.masked_invalid(probs[:, shot].cpu()).mean()
            calibration_data.conf_shots_map['calibrated'][shot] = np.ma.masked_invalid(calibrated_probs[:, shot].cpu()).mean()
            
            print(f"{shot} shot accuracy {calibration_data.accuracies[shot-shots_start]:.4f}, \
                    mean prob {calibration_data.conf_shots_map['original'][shot]:.4f}   \
                    mean calibrated prob {calibration_data.conf_shots_map['calibrated'][shot]:.4f}")
        # exit()
        # print(torch.isnan(temperatures).any()) ;exit()
        # print(ece_loss(calibrated_logits, labels)); exit()
        
        return calibration_data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--models', dest='models', action='store', required=True, help='name of models to eval the calibrator on')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of datasets to eval the calibrator on')    
    parser.add_argument('--feature_type', dest='feature_type', action='store', required=False, default="class_agnostic", help='the type of input features that make up the dataset')
    parser.add_argument('--shots_start', dest='shots_start', action='store', required=False, type=int, default=2, help='which shot # onwards we will do calibration for training and eval')
    parser.add_argument('--sampling_strategy', dest='sampling_strategy', action='store', required=False, default=None, help='what sampling strategy data to select(entropy vs similarity) (default: None - means select all)')
    parser.add_argument('--gpu_id', dest='gpu_id', action='store', default=0, required=False, help='Which CUDA gpu to run model on', type=int)
    parser.add_argument('--model_path', dest='model_path', action='store', default=None, required=False, help='Path of the model to be loaded ')
    parser.add_argument('--llm_agnostic', dest='llm_agnostic', action='store_const', const=True, default=False, help='whether to use the llm agnostic calibrator')
    parser.add_argument('--save_path', dest='save_path', action='store', default=None, required=False, help='What path to save the calibration plot to ')
    
    args = parser.parse_args()
    args = vars(args)

    args['datasets'] = convert_to_list(args['datasets'])
    args['models'] = convert_to_list(args['models'])
    
    sampling_strategy = args.get('sampling_strategy')
    if sampling_strategy:
        args['sampling_strategy'] = sampling_strategy
        
    main(**args)