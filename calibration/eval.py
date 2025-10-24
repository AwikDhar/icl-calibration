import argparse
import json
from typing import Dict
from calibration.train import get_batch, load_datasets, convert_to_list
from utils import convert_to_list
import os

import torch
import torch.nn.functional as F
from calibration.model import CalibrationTransformer
from losses import ECELoss
import numpy as np

from plot_results.plot_tc import plot_calibration

def main(model, datasets, feature_type, sampling_strategy = None, gpu_id=0, model_path=None, save_path=None):
    ece_loss = ECELoss(n_bins=30)
    device = f'cuda:{gpu_id}'
    
    data = load_datasets(model, datasets, device, feature_type, splits=('test',), sampling_strategy=sampling_strategy, apply_features=False)
    
    with open(f"calibration/models/transformer_config.json", 'r') as file:
        config = json.load(file)
    
    calibrator = None
    
    for dataset in datasets:
        if model_path is None:
            model_dir = f"./calibration/models/{model.replace('/','_')}/{dataset}"
            model_path = f'{model_dir}/calibrator'
            os.makedirs(model_dir, exist_ok=True)
    
        if calibrator is None:
            T, C = data[dataset]['test']['inputs'][0].shape
            # print(data['train'][0]['inputs'].shape, data['train'][0]['logits'].shape, data['train'][0]['labels'].shape)
            # exit()
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

        ece_shots_map, temp_shots_map, conf_shots_map, accuracies = eval(calibrator, data, dataset, device, ece_loss)
        
        plot_calibration(ece_shots_map, temp_shots_map, conf_shots_map, accuracies, 
                         model, dataset, feature_type, sampling_strategy, save_path=save_path)
    
def eval(model, eval_data, dataset, device, ece_loss):
    ece_shots_map = {'calibrated':{}, 'original':{}}
    temp_shots_map = {}
    conf_shots_map = {'calibrated':{}, 'original':{}}
    accuracies = []
    
    model.eval()
    with torch.no_grad():
        inputs, logits, labels = get_batch(eval_data[dataset]['test'], batch_size=None, device=device) # len(eval),T,C | len(eval),T,num_classes | len(eval),T
        
        temperatures = model(inputs) # B,T,1
        temperatures = torch.nan_to_num(temperatures, nan=1.0)

        calibrated_logits = logits*temperatures # B,T,num_classes
        
        B,T,num_classes = calibrated_logits.shape
        # loss = F.cross_entropy(calibrated_logits.view(B*T, num_classes), labels.view(B*T))        
        
        # print(calibrated_logits.shape, logits.shape, labels.shape, temperatures.shape)
        # exit()
        probs, preds = F.softmax(logits, dim=-1).max(dim=-1)
        calibrated_probs, calibrated_preds = F.softmax(calibrated_logits, dim=-1).max(dim=-1)

        for shot in range(T):
            ece = ece_loss(logits[:,shot, :], labels[:,shot])
            calibrated_ece = ece_loss(calibrated_logits[:,shot, :], labels[:,shot])
            
            ece_shots_map['original'][shot] = ece.item()
            ece_shots_map['calibrated'][shot] = calibrated_ece.item()
            
            temp_shots_map[shot] = temperatures[:,shot,:].flatten().cpu().numpy()

            accuracies.append((labels==preds).cpu()[:, shot].sum()/len(preds))
            conf_shots_map['original'][shot] = np.ma.masked_invalid(probs[:, shot].cpu()).mean()
            conf_shots_map['calibrated'][shot] = np.ma.masked_invalid(calibrated_probs[:, shot].cpu()).mean()
            
            print(f"{shot} shot accuracy {accuracies[shot]:.4f}, \
                    mean prob {conf_shots_map['original'][shot]:.4f}   \
                    mean calibrated prob {conf_shots_map['calibrated'][shot]:.4f}")
        
        # exit()
        # print(torch.isnan(temperatures).any()) ;exit()
        # print(ece_loss(calibrated_logits, labels)); exit()
        return ece_shots_map, temp_shots_map, conf_shots_map, accuracies
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', dest='model', action='store', required=True, help='name of model to ecal the calibrator on')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of datasets to eval the calibrator on')    
    parser.add_argument('--feature_type', dest='feature_type', action='store', required=False, default="class_agnostic", help='the type of input features that make up the dataset')
    parser.add_argument('--sampling_strategy', dest='sampling_strategy', action='store', required=False, default=None, help='what sampling strategy data to select(entropy vs similarity) (default: None - means select all)')
    parser.add_argument('--gpu_id', dest='gpu_id', action='store', default=0, required=False, help='Which CUDA gpu to run model on', type=int)
    parser.add_argument('--model_path', dest='model_path', action='store', default=None, required=False, help='Path of the model to be loaded ')
    parser.add_argument('--save_path', dest='save_path', action='store', default=None, required=False, help='What path to save the calibration plot to ')
    
    args = parser.parse_args()
    args = vars(args)
    # print(args)
    args['datasets'] = convert_to_list(args['datasets'])
    
    sampling_strategy = args.get('sampling_strategy')
    if sampling_strategy:
        args['sampling_strategy'] = sampling_strategy
        
    main(**args)