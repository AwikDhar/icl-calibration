import json
import pickle
import os
import glob
import datasets
# from data_utils_old import load_commonsense_qa
import numpy as np
import pandas as pd
import torch
import datasets

model = 'meta-llama/Llama-3.1-8B-Instruct'
dataset = 'qqp/class_agnostic'
split = 'test'
# print(os)
# icl-calibration/calibration/datasets/meta-llama_Llama-3.1-8B-Instruct/commonsense_qa
# with open(f"calibration/datasets/{model.replace('/','_')}/{dataset}/{split}.json") as file:
#     data = json.load(file)
#     for idx in range(len(data)):
#         # print(data[idx]['inputs'][0])
#         data[idx]['inputs'][0][2] = -1
#         # print(data[idx]['inputs'][0])
#         # exit()
# with open(f"calibration/datasets/{model.replace('/','_')}/{dataset}/{split}.json", 'w') as file:
#     json.dump(data, file, indent=2)
            
# with open(f"data/when2call/test.json", 'r') as file:
#     data = json.load(file)
#     print(data[0]['sentence'])
            
a = torch.tensor([ 89.0])
b = torch.logsumexp(a, dim=0).cpu().numpy()
print(b)

print(np.exp( a ))
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
