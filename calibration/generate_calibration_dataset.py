import argparse
import json
import os
import random
from data_utils import load_dataset_with_embeddings, set_prompt_params, IclDataset, IclDatasetSplit
from sampling_strategies import SamplingStrategy
from calibration.calibrator_input_generators import *
from utils import *
from tqdm import tqdm 
import numpy as np
from time import time
import logging
import multiprocessing as mp

logFormatter = logging.Formatter(
    "{asctime} - {levelname} - {message}", 
    style="{",
    datefmt="%Y-%m-%d %H:%M"
)
logger = logging.getLogger()

fileHandler = logging.FileHandler("./cal_data_gen.log")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

logger.setLevel(logging.INFO)

SAVE_DIR_TMP = 'calibration/datasets'
os.makedirs(SAVE_DIR_TMP, exist_ok=True)

def save_dataset(params, generated_dataset):
    save_path = os.path.join(SAVE_DIR_TMP, params['model'].replace('/','_'), params['dataset'], "class_agnostic") # '/' in HF model id
    os.makedirs(save_path, exist_ok=True)
    
    for split in ('train', 'test'):
        data = generated_dataset[split]
        if not data:
            logger.warning("Empty split data, skipping saving/overwriting.")
            continue
        
        file_name = os.path.join(save_path, f'{split}.json')
        if os.path.isfile(file_name):
            action = 'appending' if params['append_data'] else 'overwriting'
            logger.warning(f"WARNING! {action} existing saved files")
        
            if params['append_data']:
                with open(file_name, 'r') as file:  
                    prev_data = json.load(file)
                data = prev_data + data                           
        
        with open(file_name, 'w') as file:  
            json.dump(data, file, indent=2)

        logger.info(f"Saved to {file_name}")
        
def main(models, datasets, num_shots, train_size, test_size, append_data, api_num_log_prob, approx, bs, **kwargs):
    """
    Run experiment or load past results, print accuracy
    """
    default_params = {
        'train_size': train_size,
        'test_size': test_size,
        'append_data': append_data,
        'api_num_log_prob': api_num_log_prob,
        'approx': approx,
        'bs': bs
    }

    # list of all experiment parameters to run
    all_params = []
    calibration_datasets = {}
    for model in models:
        for dataset in datasets:
            p = deepcopy(default_params)
            p['model'] = model
            p['dataset'] = dataset
            p['num_shots'] = num_shots
            p['train_size'] = train_size
            p['test_size'] = test_size
            p.update(kwargs)
            p['generated_dataset_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot"
            calibration_datasets[p['generated_dataset_name']] = {'train':[], 'test':[]}
            all_params.append(p)

    start = time()
    datasets_dict = {}
    for dataset in datasets:
        train_split = IclDatasetSplit(*load_dataset_with_embeddings(dataset, 'train'))
        datasets_dict[dataset] = IclDataset(train=train_split, test=None)

    # with

    generate_calibration_datasets(all_params, datasets_dict, calibration_datasets)
    time_taken = time()-start
    logger.info("Time taken to finish: %d hours, %d mins", time_taken//3600, (time_taken%3600)//60)

def generate_calibration_datasets(params_list: List[Dict], datasets: Dict, calibration_datasets: Dict):#, min_confidence_limits: np.ndarray:
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """

    for params in tqdm(params_list, total=len(params_list), desc="Processing experiments"):
        file_name = os.path.join(SAVE_DIR_TMP, f"{params['generated_dataset_name'].replace('/','_')}.pkl")
        if os.path.isfile(file_name):
            logging.info("Skipping experiment, already done before.")
            continue
        logger.info(params)
        logger.info("\Dataset to be generated: %s", params['generated_dataset_name'])
        
        ### load data
        start = time()
        dataset: IclDataset = datasets[params['dataset']]
        all_sentences, all_labels, all_embeddings = dataset.train.sentences, dataset.train.labels, dataset.train.embeddings 
        split_size = int(0.8*len(all_sentences))
        all_train_sentences, all_train_labels, all_train_embeddings = all_sentences[:split_size], all_labels[:split_size], all_embeddings[:split_size]
        all_test_sentences, all_test_labels, all_test_embeddings = all_sentences[split_size:], all_labels[split_size:], all_embeddings[split_size:]

        logger.info("Train sizes: %d sentences, %d labels, %d embeddings | Test sizes: %d sentences, %d labels, %d embeddings",
                len(all_train_sentences), len(all_train_labels), len(all_train_embeddings),
                len(all_test_sentences), len(all_test_labels), len(all_test_embeddings)
        )

        # for prompt construction
        set_prompt_params(params)   

        num_shots = params['num_shots'] 

        icl_data = {
            'train': [all_train_sentences, all_train_embeddings, all_train_labels],
            'test': [all_test_sentences, all_test_embeddings, all_test_labels]
        }

        data_generation_fn = generate_data_class_agnostic
        for split in ('train', 'test'):
            all_sentences, all_embeddings, all_labels = icl_data[split]
            for iter in tqdm(range(params[f'{split}_size']), desc=f'Generating {split} split'):
                sampling_strategy = random.choices(
                    [SamplingStrategy.ENTROPY, SamplingStrategy.SIMILARITY],
                    [0.5, 0.5],
                    # [0, 1],
                    # [1, 0],
                    k=1
                    )[0]
                if sampling_strategy==SamplingStrategy.ENTROPY:        
                    selected_sentences, selected_labels, selected_idxs = random_sampling(all_sentences, all_labels, num_shots+1, "rand") 
                    selected_embeddings = np.array([all_embeddings[idx] for idx in selected_idxs])
                    data = data_generation_fn(params, selected_sentences, selected_embeddings, selected_labels)
                elif sampling_strategy==SamplingStrategy.SIMILARITY:
                    selected_sentences, selected_labels, selected_idxs = random_sampling(all_sentences, all_labels, 1, "rand") 
                    selected_embeddings = np.array([all_embeddings[idx] for idx in selected_idxs]) # only 1 idx here
                    sampled_data = similarity_sampling(all_sentences, all_embeddings, all_labels, 
                                                       test_embeddings=selected_embeddings, num_shots=num_shots+1, shuffle=True, return_embeddings=True)
                    # print(sampled_data.sentences, selected_sentences[0])
                    # exit()
                    # print(sampled_data.embeddings.shape, all_embeddings.shape)
                    # exit()
                    data = data_generation_fn(params, sampled_data.sentences[0], sampled_data.embeddings[0], sampled_data.labels[0])
                data['sampling_strategy'] = sampling_strategy.name
                calibration_datasets[params['generated_dataset_name']][split].append(data)

        logger.info("Time taken to create %s dataset: %d sec", params['generated_dataset_name'], round(time()-start))
        save_dataset(params, calibration_datasets[params['generated_dataset_name']])

def args_check(args: Dict):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # mp.set_start_method('spawn', force=True)
    # setup_single_threading()
    setup_vllm_env_settings()
    # required arguments
    parser.add_argument('--model', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_shots', dest='num_shots', action='store', required=True, help='num training examples to use', type=int)
    parser.add_argument('--train_size', dest='train_size', action='store', required=True, type=int,
                            default=10000, help='how big of a trian dataset you want to generate')
    parser.add_argument('--test_size', dest='test_size', action='store', required=True, type=int,
                            default=2000, help='how big of a test dataset you want to generate')
    parser.add_argument('--append_data', dest='append_data', action='store_const', required=False, const=True, default=False,
                            help='append to existing dataset if True, overwrite otherwise')
    parser.add_argument('--api_num_log_prob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=100, help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    # flags
    parser.add_argument('--approx', dest='approx', action='store_const', const=True, default=True,
                        help='whether to set token prob to zero if not in top 100')
    
    parser.add_argument('--gpu_id', dest='gpu_id', action='store', default=0, required=False, help='Which CUDA gpu to run model on', type=int)
    
    args = parser.parse_args()
    args = vars(args)
    # print(args)
    

    # simple processing
    def convert_to_list(items, cvt_func=None):
        if cvt_func:
            return [cvt_func(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]

    args['models'] = convert_to_list(args['models'])
    args['datasets'] = convert_to_list(args['datasets'])
    
    args_check(args)
    main(**args)