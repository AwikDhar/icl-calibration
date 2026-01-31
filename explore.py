import json
import pickle
import os
import glob
from time import time
# import datasets
# from data_utils_old import load_commonsense_qa
import numpy as np
import orjson
import msgspec
import pandas as pd
import torch
from calibration.model import CalibrationTransformer
from calibration_methods import CalibrationMethods

# T, embedding_dim = 8, 64
# positions = torch.arange(T).unsqueeze(-1)  # [5, 1]: [[0], [1], [2], [3], [4]]
# denominator = torch.rand((embedding_dim//2,))
# print((positions/denominator).shape); exit()
# max_temp = 1 + torch.clamp(torch.tanh(4 * (torch.tensor(0.9) - 0.5)), min=0)
# temp = torch.empty(1, 1, 1).uniform_(1, 1).mul_(max_temp - 1).add_(1)[0][0]
# print(max_temp, temp); exit()
# models = ['meta-llama/Llama-3.1-8B-Instruct', 'Qwen/Qwen3-8B', 'openai/gpt-oss-20b']
# datasets = ['sst5', 'snli', 'trec', 'rte', 'agnews', 'goemotions', 'dbpedia_l2', 'toxic_chat', 'newsgroups', 'commonsense_qa', 'qqp', 'banking77', 'metatool', 'wildguard', 'when2call', 'wikitoxic', 'amazon_counterfactual', 'massive_intent', 'dbpedia_l1']
# models = ['meta-llama/Llama-3.1-8B-Instruct']
# datasets = ['yelp_reviews']
# splits = ('train','test')

# for dataset in datasets:
#     for split in splits:
#         path = f"data/{dataset}/{split}.json"
#         with open(path, 'r') as file:
#             split_data = msgspec.json.decode(file.read())
#         with open(path, 'w') as file:
#             json.dump(split_data, file)

# for llm in models:
#         for dataset in datasets:
#             for split in splits:
#                 path = f"calibration/datasets/{llm.replace('/','_')}/{dataset}/{split}.json"
#                 with open(path, 'r') as file:
#                     split_data = msgspec.json.decode(file.read())
#                 with open(path, 'w') as file:
#                     json.dump(split_data, file)
# s = time()
# with open("./calibration/datasets/openai_gpt-oss-20b/agnews/train.json", 'r') as file:
#     data = msgspec.json.decode(file.read())
#     # print(data[0]['inputs'][10][:30])
# print(f"{time()-s: .2f}")

# s = time()
# with open("/mnt/nas/awikdhar/calibration/icl-calibration/data/yahoo_answers/train.json", 'r') as file:
#     data = json.load(file)
#     print(len(data)); exit()
    # print(data[0]['inputs'][10][:30])
# print(f"{time()-s: .2f}")

# s = time()
# with open("./calibration/datasets/openai_gpt-oss-20b/agnews/train.json", 'r') as file:
#     data = orjson.loads(file.read())
#     # print(data[0]['inputs'][10][:30])
# print(f"{time()-s: .2f}")

# with open("./calibration/datasets/openai_gpt-oss-20b/agnews/test.json", 'w') as file:
#     json.dump(data, file)

# exit()
# print(torch.rand(()).item())
# print(torch.cuda.get_device_name(0), torch.cuda.get_device_name(1), torch.cuda.get_device_name(2), torch.cuda.get_device_name(3), torch.cuda.device_count()); exit()
# calibrator = CalibrationTransformer(in_features=173, embedding_dim=64, num_layers=12)
# print(calibrator)
# breakpoint()
# print(f"{sum(p.numel() for p in calibrator.parameters())/10**6: .2f} M parameters"); exit()
# model = 'Qwen/Qwen3-8B'
# model = 'openai/gpt-oss-20b'
model = 'meta-llama/Llama-3.1-8B-Instruct'
# model = "gemini-2.5-flash"
datasets = ['sst5', 'snli', 'trec', 'rte', 'agnews', 'goemotions', 'dbpedia_l2', 'toxic_chat', 'newsgroups', 'yahoo_answers']

# split = 'test'
split = 'train'

# if 1 or split == 'test':
#     datasets += ['commonsense_qa', 'qqp', 'banking77', 'metatool', 'wildguard', 'when2call', 'wikitoxic', 'amazon_counterfactual', 'massive_intent', 'dbpedia_l1', 'yelp_reviews']
#     datasets += ['qqp', 'commonsense_qa', 'wildguard', 'when2call', 'wikitoxic', 'amazon_counterfactual']
# datasets = ['metatool', 'dbpedia_l1' 'amazon_counterfactual', 'massive_intent', 'wildguard', 'commonsense_qa', 'qqp']
# print(os)
# icl-calibration/calibration/datasets/meta-llama_Llama-3.1-8B-Instruct/commonsense_qa
print(f"|----{model}----|")
for idx, dataset in enumerate(datasets[::-1]):
    print(f'\n{idx+1}. '+dataset + ': ')
    # for split in ('train', ):
    try:
        with open(f"calibration/datasets/{model.replace('/','_')}/{dataset}/{split}.json") as file:
            data = msgspec.json.decode(file.read())
            # print(data[-1]['logits'])
            
            # exit()
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
        exit()
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
# with open("./saved_results/Qwen_Qwen3-Next-80B-A3B-Instruct/amazon_counterfactual/SIMILARITY/6_shot/1_seed.pkl", 'rb') as f:
#     data = pickle.load(f)
#     print(data.keys())
#     print(data['metrics'][CalibrationMethods.TF].__dict__)
#     print(data['metrics'][CalibrationMethods.TF].calibration_metrics.__dict__)
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
