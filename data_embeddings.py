import argparse
from typing import List, Dict
import os
import json
from time import time

from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from data_utils.create_dataset import load_dataset
from utils import ROOT_DIR

def check_overwriting_creating_dataset(dataset:str):
    if os.path.isfile(f"{ROOT_DIR}/data/{dataset}/train.json"):
        user_inp = input(f"{dataset} already exists! Overwrite? ( y/n )")
        return user_inp.strip().lower()=='y'
    return True

def create_dataset_with_embeddings(model:str, dataset:str, dim = None)->None:
    """
    Create a standardized dict dataset with sentences, their embeddings and labels

    Args:
        model (str): Sentence transformer model
        dataset (str): The dataset whose sentence embeddings are to be generated
        dim (int): embedding dim
    """    
    start = time()

    dir_path = f"{ROOT_DIR}/data/{dataset}"
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    # if not check_overwriting_creating_dataset(dataset):
    #     print('cancelled')
    #     return
        
    params = {'dataset':dataset} # the function requires a params dict for many operations, we only need the dataset
    train_sentences, train_labels, test_sentences, test_labels = load_dataset(params) # entire dataset
    
    batch_size = 256
    model = SentenceTransformer(f'{model}', device='cuda:0', truncate_dim=dim, cache_folder=os.environ['HF_HOME'], model_kwargs={'torch_dtype':torch.bfloat16})
    with torch.inference_mode():
        train_sentences_embeddings = model.encode(train_sentences, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True) 
        test_sentences_embeddings =  model.encode(test_sentences, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    def get_dataset_dict(sentences, labels, embeddings):
        return [
            {
                "sentence":sentence,
                "label":label,
                "sentence_embedding":np.float32(embedding).tolist()
            }
            for sentence, label, embedding in zip(sentences, labels, embeddings)
        ]

    train_data = get_dataset_dict(train_sentences, train_labels, train_sentences_embeddings)
    test_data = get_dataset_dict(test_sentences, test_labels, test_sentences_embeddings)

    data_dir = f'{ROOT_DIR}/data/{dataset}'
    os.makedirs(data_dir, exist_ok=True)
    
    with open(f"{data_dir}/train.json", 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(f"{data_dir}/test.json", 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"{time()-start: .2f}s to create the dataset {dataset} with {dim} dimensions")

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('--model', dest='model', action='store', default='google/embeddinggemma-300m', required=False, help='name of model(s), e.g., GPT2-XL')
    argparser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    argparser.add_argument('--dim', dest='dim', action='store', required=True, default=None, help='embedding dim if using Matryoshka trained model', type=int)
    argparser.add_argument('--gpu_id', dest='gpu_id', action='store', required=False, default=0, type=int)
    
    args = vars(argparser.parse_args())
    args['datasets'] = [s.strip() for s in args['datasets'].split(",")]
    os.environ['CUDA_VISIBLE_DEVICES']=str(args['gpu_id'])
    
    for dataset in args['datasets']:
        create_dataset_with_embeddings(args['model'], dataset, args['dim'])
