import json
import pickle
import os
import glob
# import datasets
# from data_utils_old import load_commonsense_qa
import numpy as np
import pandas as pd
import torch
from calibration.model import CalibrationTransformer

# print(torch.cuda.get_device_name(0), torch.cuda.get_device_name(1), torch.cuda.get_device_name(2), torch.cuda.get_device_name(3), torch.cuda.device_count()); exit()
# calibrator = CalibrationTransformer(in_features=11)
# print(f"{sum(p.numel() for p in calibrator.parameters())/10**6: .2f} M parameters"); exit()
# model = 'Qwen/Qwen3-8B'# # 'openai/gpt-oss-20b' 'Qwen/Qwen3-8B'
# model = 'openai/gpt-oss-20b'
model = 'meta-llama/Llama-3.1-8B-Instruct'
datasets = ['sst5', 'snli', 'trec', 'rte', 'agnews', 'goemotions', 'dbpedia_l2', 'toxic_chat', 'newsgroups']

# split = 'test'
split = 'train'

if split == 'test':
    datasets += ['commonsense_qa', 'qqp', 'banking77', 'metatool', 'wildguard', 'when2call', 'wikitoxic', 'amazon_counterfactual', 'massive_intent', 'dbpedia_l1']
# datasets = ['metatool', 'dbpedia_l1' 'amazon_counterfactual', 'massive_intent', 'wildguard', 'commonsense_qa', 'qqp']
# print(os)
# icl-calibration/calibration/datasets/meta-llama_Llama-3.1-8B-Instruct/commonsense_qa
print(f"|----{model}----|")
for idx, dataset in enumerate(datasets):
    print(f'\n{idx+1}. '+dataset + ': ')
    # for split in ('train', ):
    try:
        with open(f"calibration/datasets/{model.replace('/','_')}/{dataset}/{split}.json") as file:
            data = json.load(file)
            # if len(data)>1000:
            #     data = data[1000:]
            sim_count = len([True for item in data if item['sampling_strategy']=='ENTROPY'])
            ratio = sim_count/len(data)
            print(f' {split} size: {len(data)}')
            if abs(ratio-0.5)>0.1:
                print(f' Imbalanced {split} split: {sim_count/len(data)}')
    except Exception as e:
        print(type(e).__name__, e)
    # print('-----------')
        # exit()
# with open(f"calibration/datasets/{model.replace('/','_')}/{dataset}/{split}.json", 'w') as file:
#     json.dump(data, file, indent=2)
            
# with open(f"data/when2call/test.json", 'r') as file:
#     data = json.load(file)
#     print(data[0]['sentence'])

# print(np.exp( b ))
# print(np.exp(a))
# print(torch.exp(torch.tensor([83, 86, 87, 88.75, 89], dtype=torch.float64)))
# with open("calibration.log", "r") as f:
#     content = f.read()
#     print(content[-2000:])
# with open("./raw_logits/sst2_meta-llamasst2_meta-llama_Llama-3.2-3B_0shot_100_subsample_seed0.pkl", 'rb') as f:
# with open("./raw_logits_high_bs/sst5_meta-llama_Llama-3.1-8B_0shot_rand_entropy_level_seed4.pkl", 'rb') as f:
#     data = pickle.load(f)
#     print(data['all_labels_prob_mass'])
    # print(data, data.keys(), data['all_label_probs'], data['params'], data['accuracies'][0], data['eces'][0])
#     print(data['raw_logits'].shape, data['all_label_probs'].shape)

# with open("calibration/datasets/meta-llama_Llama-3.1-8B/sst5/test.json", 'r') as file:
#     data = json.load(file)

# with open("calibration/datasets/meta-llama_Llama-3.1-8B/sst5/test.json", 'w') as file:    
#     for idx in range(len(data)):
#         orig = np.array(data[idx]['inputs'])
#         for i in range(len(orig)-1):
#             orig[i][-(8-i):] = 0
#         data[idx]['inputs'] = orig.tolist()
        
#     json.dump(data, file, indent=2)
    
# with open("calibration/datasets/meta-llama_Llama-3.1-8B/sst5/train.json", 'r') as file:
#     data = json.load(file)
