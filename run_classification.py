import argparse
import os
import pprint
import shutil
from tqdm import tqdm 
import numpy as np
from time import time, sleep
from typing import Callable, List, Dict
# import multiprocessing as mp

from data_utils import load_dataset_with_embeddings, set_prompt_params, IclDataset, IclDatasetSplit
from utils.gen_utils import *
from utils.sampling_utils import *
from metrics import ClassificationMetrics
from losses import compute_binned_ce, smooth_ece

from calibration_methods import CalibrationMethods
from sampling_strategies import SamplingStrategy
from calibration.generate_calibration_dataset import get_autoregressive_results, get_shifted_features

# SAVE_DIR_TMP = ROOT_DIR/"saved_results_final"
# os.makedirs(SAVE_DIR_TMP, exist_ok=True)
    
def save_pickle_tmp(params, data):
    # save results from model
    file_name = get_saved_results_file_name(params)
    
    if os.path.isfile(file_name):
        logger.warning("WARNING! overwriting existing saved files")
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    
    logger.info(f"Saved to {file_name}")

def main(models, datasets, all_shots, num_seeds, seed_start, subsample_test_set, api_num_log_prob, entropy_levels, use_saved_results, **kwargs):
    """
    Run experiment or load past results, print accuracy
    """
    default_params = {
        'subsample_test_set': subsample_test_set,
        'api_num_log_prob': api_num_log_prob,
    }

    sampling_strategy = kwargs['sampling_strategy']
    # print(sampling_strategy)
    # list of all experiment parameters to run
    all_params = []
    for model in models:
        for dataset in datasets:
            if sampling_strategy==SamplingStrategy.ENTROPY:
                for entropy_level in entropy_levels:
                    for num_shots in all_shots:
                        if entropy_level!=EntropyLevels.RANDOM and num_shots==0:
                            continue
                        for seed in range(seed_start, num_seeds):
                            p = deepcopy(default_params)
                            p['model'] = model
                            p['dataset'] = dataset
                            p['num_shots'] = num_shots
                            p['entropy_level'] = entropy_level
                            p['seed'] = seed
                            p.update(kwargs)
                            p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{p['entropy_level']}_entropy_level_seed{p['seed']}"
                            all_params.append(p)
            elif sampling_strategy==SamplingStrategy.SIMILARITY:
                for num_shots in all_shots:
                    if num_shots==0:
                        continue
                    for seed in range(seed_start, num_seeds):
                        p = deepcopy(default_params)
                        p['model'] = model
                        p['dataset'] = dataset
                        p['num_shots'] = num_shots
                        p['seed'] = seed
                        p.update(kwargs)
                        p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_similarity_sampling_seed{p['seed']}"
                        all_params.append(p)

    calibration_methods = all_params[0].get('calibration')
    if calibration_methods is not None:
        for params in all_params:
            params['expr_name'] = params['expr_name'].replace('_seed', f'_{"_".join([method.name for method in calibration_methods])}_seed')

    # query the model and save the responses
    if use_saved_results:
        load_results(all_params)
    else:
        start = time()
        datasets_dict = {}
        for dataset in datasets:
            train_split = IclDatasetSplit(*load_dataset_with_embeddings(dataset, 'train'))
            test_split = IclDatasetSplit(*load_dataset_with_embeddings(dataset, 'test'))
            datasets_dict[dataset] = IclDataset(train=train_split, test=test_split)

        # min_confidence_limits = np.linspace(1e-3, 0.01, max(all_shots)+1) # increase lower limit of mean of minimum confidence per prediction, as shots increase, to 2%
        # print("Root handlers:", logging.getLogger().handlers); exit()

        run_and_save_results(all_params, datasets_dict)
        time_taken = time()-start
        logger.info("Time taken to finish: %d hours, %d mins", time_taken//3600, (time_taken%3600)//60)

def run_and_save_results(params_list: List[Dict], datasets: Dict):#, min_confidence_limits: np.ndarray):
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """
    result_tree = dict()

    for _, params in tqdm(enumerate(params_list), total=len(params_list), desc="Processing experiments", smoothing=0.8):
        fix_seed(params['seed'])
        
        file_name = get_saved_results_file_name(params)
        if os.path.isfile(file_name) and params['overwrite_type'] is OverWriteType.SKIP:
            logger.info(f"Skipping experiment shot {params['num_shots']} seed {params['seed']}, already done before.")
            continue
        logger.info(params)
        # pprint.pprint(params)
        logger.info("\nExperiment name: %s", params['expr_name'])
        logger.info("\nSave dir: %s", get_saved_results_file_name(params))
        
        ### load data
        start = time()
        dataset: IclDataset = datasets[params['dataset']]
        all_train_sentences, all_train_labels, all_train_embeddings = dataset.train.sentences, dataset.train.labels, dataset.train.embeddings 
        all_test_sentences, all_test_labels, all_test_embeddings = dataset.test.sentences, dataset.test.labels, dataset.test.embeddings
        
        logger.info("Train sizes: %d sentences, %d labels, %d embeddings | Test sizes: %d sentences, %d labels, %d embeddings",
                len(all_train_sentences), len(all_train_labels), len(all_train_embeddings),
                len(all_test_sentences), len(all_test_labels), len(all_test_embeddings)
        )

        ### sample few-shot training examples
        num_shots = params['num_shots'] 
        calibration_methods = params.get('calibration')
        has_calibration_set = set(calibration_methods).intersection({CalibrationMethods.GC, CalibrationMethods.BC, CalibrationMethods.BTF, CalibrationMethods.PTF})             
        # print(logger.handlers); exit()

        train_sentences, train_labels, train_embeddings = [], [], [] 
            
        if params['sampling_strategy']==SamplingStrategy.ENTROPY:
            if params['entropy_level']==EntropyLevels.RANDOM_SHARED: 
                params['prompt_shared'] = True
                
            test_sentences, test_labels, test_embeddings = get_test_data(all_test_sentences, all_test_labels, all_test_embeddings, params['subsample_test_set'])
            
            for i in range(len(test_sentences)):
                selected_sentences, selected_labels, selected_idxs = random_sampling(all_train_sentences, all_train_labels,
                                                                      num_shots, params['entropy_level'], test_labels[i]) 
                                   
                if has_calibration_set:
                    calib_set_sentences, calib_set_labels = [], [] 

                    remaining_sentences  = [sentence  for i, sentence  in enumerate(all_train_sentences)  if i not in set(selected_idxs)]
                    remaining_labels     = [label     for i, label     in enumerate(all_train_labels)     if i not in set(selected_idxs)]
                    remaining_embeddings = [embedding for i, embedding in enumerate(all_train_embeddings) if i not in set(selected_idxs)]
                   
                    calib_set_sentences, calib_set_labels, calib_set_idxs = random_sampling(remaining_sentences, remaining_labels, params['calibration_set_size'], EntropyLevels.RANDOM)
                    calib_set_embeddings = [remaining_embeddings[idx] for idx in calib_set_idxs]
                    
                if CalibrationMethods.TF in calibration_methods:
                    train_embeddings.append([all_train_embeddings[idx] for idx in selected_idxs])
                    # print(train_embeddings, selected_idxs); exit()            
                
                # sharing the training/in-context examples is expected for GC, BC which require separate calibration set
                if params['prompt_shared'] or has_calibration_set:
                    train_sentences = [selected_sentences for i in range(len(test_sentences))]
                    train_embeddings = np.vstack([train_embeddings for i in range(len(test_sentences))])
                    train_labels = [selected_labels for i in range(len(test_sentences))]
                    break
                
                train_sentences.append(selected_sentences)
                train_labels.append(selected_labels)

        elif params['sampling_strategy']==SamplingStrategy.SIMILARITY:
            shuffle_examples = random.sample([0,1], 1)[0]  # 50% odds of the in-context examples being sorted by similarity
            
            if params['subsample_test_set']>=len(all_test_sentences) and not shuffle_examples and params['seed']>0:
                logger.warning("Found less test inputs in dataset than requested, saving seed 0 results to avoid repeated experiments")
                model_name = params['expr_name'].replace('/','_') # In case it's an HF model
                file_name = os.path.join(SAVE_DIR, f"{model_name}.pkl")
                shutil.copy(file_name.replace(f"seed{params['seed']}", 'seed0'), file_name)
                continue
            
            test_sentences, test_labels, test_embeddings = get_test_data(all_test_sentences, all_test_labels, all_test_embeddings, params['subsample_test_set'])

            sampled_data = similarity_sampling(all_train_sentences, all_train_embeddings, all_train_labels, 
                                               test_embeddings, num_shots, test_sentences, 
                                               shuffle=shuffle_examples, return_embeddings=True)
            train_sentences, train_embeddings, train_labels = sampled_data.sentences, sampled_data.embeddings, sampled_data.labels
            
        logger.info("Time taken to load %s dataset: %d sec", params['dataset'], round(time()-start))

        # for prompt construction
        set_prompt_params(params)   

        logger.info(f"getting raw resp for {len(test_sentences)} test sentences with {num_shots} ICL examples.")
        
        # if all the labels are distinguishable by their first token, only 1 llm call per query prediction will be made, we can get IC example predictions conveniently
        single_depth_label_trie = len(LabelsTrie(params['label_dict']).root.children) == len(params['label_dict'])
        in_context_logprobs = single_depth_label_trie and get_llm_framework(params['model'])==LlmFramework.HF

        llm_result = get_results(params, train_sentences, train_labels, test_sentences, in_context_logprobs=in_context_logprobs)
        probs = np.array([llm_result.logprobs[i].probs[0] for i in range(len(test_sentences))])
        logits = np.array([llm_result.logprobs[i].logits[0] for i in range(len(test_sentences))])
        
        acc_original, conf_ori = eval_accuracy(probs, test_labels)
        ece_original = smooth_ece(logits, test_labels)
        mce_original = compute_binned_ce(logits, test_labels).mce
        
        accuracies = [acc_original]
        eces = [ece_original.item()]
        mces = [mce_original]
        confs = [conf_ori]

        metrics = {CalibrationMethods.UNCALIBRATED: ClassificationMetrics(logits, logits, test_labels)}
        calibrated_logits = {}
        if calibration_methods is not None:
            if set(calibration_methods).intersection({CalibrationMethods.TF, CalibrationMethods.ICT}):
                if in_context_logprobs:
                    train_logits = np.array([llm_result.in_context_logprobs[test_idx].logits for test_idx in range(len(test_sentences))])
                else:
                    train_probs, train_logits = get_autoregregressive_train_results(params, train_sentences, train_labels)
                    logger.info("Generated autoregressive predictions over in context examples")

                combined_logits = np.concatenate([train_logits, np.expand_dims(logits, axis=1)], axis=1)
                combined_labels = np.array([train_labels[i] + [-1] for i in range(len(test_sentences))]) # dummy test label for methods that expect k+1 labels

            if CalibrationMethods.TF in calibration_methods:
                combined_sentences = [train_sentences[test_idx] + [test_sentences[test_idx]] for test_idx in range(len(test_sentences))]
                calibrated_logits[CalibrationMethods.TF] = get_tc_logits(params, combined_sentences, train_labels, train_logits, logits)
                # calibrated_logits[CalibrationMethods.TF] = get_tc_logits(params, combined_sentences, train_labels, train_logits, logits, 'google/embeddinggemma-300m')
                # calibrated_logits[CalibrationMethods.TF] = get_tc_logits(params, combined_sentences, train_labels, train_logits, calibrated_logits[CalibrationMethods.TF], 'google/embeddinggemma-300m')
                metrics[CalibrationMethods.TF] = ClassificationMetrics(logits, calibrated_logits[CalibrationMethods.TF], test_labels)
            
            if CalibrationMethods.TF_HF in calibration_methods:
                train_hidden = [llm_result.in_context_logprobs[test_idx].hidden_features for test_idx in range(len(test_sentences))]
                test_hidden = [llm_result.logprobs[test_idx].hidden_features for test_idx in range(len(test_sentences))]
                combined_hidden = [train_hidden[test_idx] + test_hidden[test_idx] for test_idx in range(len(test_sentences))]
                
                calibrated_logits[CalibrationMethods.TF_HF] = get_tc_logits(params, combined_hidden, train_labels, train_logits, logits, 'hidden_features')
                # calibrated_logits[CalibrationMethods.TF_HF] = get_tc_logits(params, combined_hidden, train_labels, train_logits, calibrated_logits[CalibrationMethods.TF_HF], 'hidden_features')
                metrics[CalibrationMethods.TF_HF] = ClassificationMetrics(logits, calibrated_logits[CalibrationMethods.TF_HF], test_labels)
                
            if CalibrationMethods.TF_QE in calibration_methods:
                combined_sentences = [train_sentences[test_idx] + [test_sentences[test_idx]] for test_idx in range(len(test_sentences))]
                calibrated_logits[CalibrationMethods.TF_QE] = get_tc_logits(params, combined_sentences, train_labels, train_logits, logits, 'Qwen/Qwen3-Embedding-4B')
                metrics[CalibrationMethods.TF_QE] = ClassificationMetrics(logits, calibrated_logits[CalibrationMethods.TF_QE], test_labels)
            
            if CalibrationMethods.ICT in calibration_methods:
                assert np.array_equal(combined_logits[:, -1, :].argmax(axis=-1), logits.argmax(axis=-1)), "Combined logits issue"
                calibrated_logits[CalibrationMethods.ICT] = get_ict_logits(params, combined_logits, combined_labels)
                metrics[CalibrationMethods.ICT] = ClassificationMetrics(logits, calibrated_logits[CalibrationMethods.ICT], test_labels)
                
            if CalibrationMethods.FS_ICT in calibration_methods:
                calibrated_logits[CalibrationMethods.FS_ICT] = get_fs_ict_logits(params, train_sentences, train_labels, test_sentences, logits)
                metrics[CalibrationMethods.FS_ICT] = ClassificationMetrics(logits, calibrated_logits[CalibrationMethods.FS_ICT], test_labels)
            
            if CalibrationMethods.ICC in calibration_methods:
                logger.info("Getting ICC calibrated logits")
                calibrated_logits[CalibrationMethods.ICC] = get_icc_logits(params, train_sentences, train_labels, probs)
                metrics[CalibrationMethods.ICC] = ClassificationMetrics(logits, calibrated_logits[CalibrationMethods.ICC], test_labels)
            
            if CalibrationMethods.PERMUT_AVG in calibration_methods:
                calibrated_logits[CalibrationMethods.PERMUT_AVG] = get_permutation_averaged_logits(params, train_sentences, train_labels, test_sentences, probs, num_permutations=5)
                metrics[CalibrationMethods.PERMUT_AVG] = ClassificationMetrics(logits, calibrated_logits[CalibrationMethods.PERMUT_AVG], test_labels)
            
            if has_calibration_set:
                calib_set_probs, calib_set_logits = get_results(params, train_sentences, train_labels, calib_set_sentences)

                if CalibrationMethods.GC in calibration_methods:
                    num_classes = probs.shape[-1]
                    labels_sample_set = random.sample(all_train_labels, min(20*num_classes, len(all_train_labels)))
                    
                    calibrated_logits[CalibrationMethods.GC] = get_gc_logits(probs, calib_set_probs, labels_sample_set)
                    metrics[CalibrationMethods.GC] = ClassificationMetrics(logits, calibrated_logits[CalibrationMethods.GC], test_labels)
                
                if CalibrationMethods.BC in calibration_methods:
                    calibrated_logits[CalibrationMethods.BC] = get_bc_logits(logits, calib_set_logits)
                    metrics[CalibrationMethods.BC] = ClassificationMetrics(logits, calibrated_logits[CalibrationMethods.BC], test_labels)
                
                if CalibrationMethods.BTF in calibration_methods:
                    btf_seq_len = 20
                    btf_calib_sentences, btf_calib_labels, btf_calib_idxs = random_sampling(calib_set_sentences, calib_set_labels, btf_seq_len, EntropyLevels.RANDOM) 
                    btf_calib_embeddings = [calib_set_embeddings[idx] for idx in btf_calib_idxs]
                    btf_calib_logits = [calib_set_logits[idx] for idx in btf_calib_idxs]
                    
                    btf_calib_sentences = [btf_calib_sentences for i in range(len(test_sentences))]
                    btf_calib_embeddings = np.vstack([btf_calib_embeddings for i in range(len(test_sentences))])
                    btf_calib_labels = [btf_calib_labels for i in range(len(test_sentences))]
                    btf_calib_logits = [btf_calib_logits.copy() for i in range(len(test_sentences))]
                    
                    # Apply transformer calibration using calibration set examples as context
                    calibrated_logits[CalibrationMethods.BTF] = get_tc_logits(params, btf_calib_sentences, btf_calib_embeddings, btf_calib_labels, btf_calib_logits, logits)
                    metrics[CalibrationMethods.BTF] = ClassificationMetrics(logits, calibrated_logits[CalibrationMethods.BTF], test_labels)
                
                if CalibrationMethods.PTF in calibration_methods:
                    ptf_allowed_methods = [CalibrationMethods.GC, CalibrationMethods.BC]
                    for method in ptf_allowed_methods:
                        if method in calibration_methods:
                            ptf_calibration_method = method
                            break

                    calibrated_logits[CalibrationMethods.PTF] = get_tc_logits(params, train_sentences, train_embeddings, train_labels, train_logits, calibrated_logits[ptf_calibration_method])
                    metrics[CalibrationMethods.PTF] = ClassificationMetrics(logits, calibrated_logits[CalibrationMethods.PTF], test_labels)
                    
            accuracies.extend([metrics[method].calibrated_accuracy for method in calibration_methods])
            eces.extend([metrics[method].calibration_metrics.calibrated_ece for method in calibration_methods])
            mces.extend([metrics[method].calibration_metrics.calibrated_mce for method in calibration_methods])
            confs.extend([metrics[method].mean_calibrated_conf for method in calibration_methods])

        print(f"Accuracies: {accuracies}")
        print(f"Ece      : {eces}")
        print(f"confidence      : {confs}")
        
        # add to result_tree
        match params['sampling_strategy']:
            case SamplingStrategy.ENTROPY:
                exp_setting = params['entropy_level']
            case SamplingStrategy.SIMILARITY:
                exp_setting = 'similarity_sampling'

        keys = [params['dataset'], params['model'], exp_setting, num_shots]

        node = result_tree # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        seed = params['seed']
        node[seed] = accuracies
        entropy_node = result_tree[keys[0]][keys[1]][keys[2]]
       
        if not f"{keys[3]}_ece" in entropy_node.keys():
            entropy_node[f"{keys[3]}_ece"] = dict()
        if not f"{keys[3]}_mce" in entropy_node.keys():
            entropy_node[f"{keys[3]}_mce"] = dict()        
        if not f"{keys[3]}_conf" in entropy_node.keys():
            entropy_node[f"{keys[3]}_conf"] = dict()

        entropy_node[f"{keys[3]}_ece"][seed] = eces
        entropy_node[f"{keys[3]}_mce"][seed] = mces
        entropy_node[f"{keys[3]}_conf"][seed] = confs

        # save to file
        result_to_save = dict()
        params_to_save = deepcopy(params)
        
        result_to_save['params'] = params_to_save
        result_to_save['probs'] = probs
        result_to_save['logits'] = logits
        result_to_save['labels'] = test_labels
        
        result_to_save['eces'] = eces
        result_to_save['mces'] = mces
        result_to_save['confs'] = confs
        result_to_save['accuracies'] = accuracies
        result_to_save['metrics'] = metrics
            
        print_results(result_tree, calibration_methods)
        save_pickle_tmp(params, result_to_save)

    print_results(result_tree, calibration_methods, log=logger.info)
    
def get_autoregregressive_train_results(params, train_sentences, train_labels):
    if params['prompt_shared']:
        probs, logits = get_autoregressive_results(params, train_sentences[0], train_labels[0])    
        train_probs = [probs.copy() for _ in range(len(train_sentences))]
        train_logits = [logits.copy() for _ in range(len(train_sentences))] 
    else:
        train_probs, train_logits = [], []
        
        for i in range(len(train_sentences)):
            probs, logits = get_autoregressive_results(params, train_sentences[i], train_labels[i])
            train_probs.append(probs)
            train_logits.append(logits)
            
    return train_probs, train_logits 
        
def get_ict_logits(params, combined_logits: np.ndarray, combined_labels: np.ndarray):
    combined_logits = torch.from_numpy(combined_logits)
    combined_labels = torch.from_numpy(combined_labels)
        
    data = {
        "logits": combined_logits,
        "labels": combined_labels
    }
    
    num_shots = params['num_shots']
    B, T, num_classes = combined_logits.shape
    
    temperatures = get_shotwise_dynamic_temperatures(data, shots_start=num_shots)
    calibrated_logits = combined_logits[:, -1, :] * temperatures[num_shots].view(B, 1)
    
    return calibrated_logits.cpu().numpy()

def get_fs_ict_logits(params, train_sentences, train_labels, test_sentences, logits):
    num_shots = params['num_shots']
    num_shots_fs = round((num_shots-1)/2.0)
    
    logger.info(f"Learning temperature from (k-1)/2 = {num_shots_fs} shot predictions on the k = {num_shots} in-context examples ")
    
    all_chosen_idxs = []
    for fs_test_sentence_idx in range(num_shots):
        available_idxs = [idx for idx in list(range(num_shots)) if idx!=fs_test_sentence_idx]
        chosen_idxs = np.random.permutation(available_idxs)[:num_shots_fs]
        all_chosen_idxs.append(chosen_idxs)
        
    combined_logits = []
    combined_labels = []
    
    for test_sentence_idx in range(len(test_sentences)):
        # for fs_test_sentence_idx in range(num_shots):
        fs_train_sentences = [[train_sentences[test_sentence_idx][chosen_idx] for chosen_idx in chosen_idxs] for chosen_idxs in all_chosen_idxs]
        fs_train_labels = [[train_labels[test_sentence_idx][chosen_idx] for chosen_idx in chosen_idxs] for chosen_idxs in all_chosen_idxs]
        fs_test_sentences = [train_sentences[test_sentence_idx][fs_test_sentence_idx] for fs_test_sentence_idx in range(num_shots)]
        fs_test_labels = [train_labels[test_sentence_idx][fs_test_sentence_idx] for fs_test_sentence_idx in range(num_shots)]
            
        llm_result = get_results(params, fs_train_sentences, fs_train_labels, fs_test_sentences)
        # probs = np.array([llm_result.logprobs[i].probs[0] for i in range(len(test_sentences))])
        fs_logits = np.array([llm_result.logprobs[i].logits[0] for i in range(len(fs_test_sentences))])
        cur_combined_logits = np.vstack([fs_logits, logits[test_sentence_idx]])
        cur_combined_labels = fs_test_labels + [-1] # dummy label since it isn't needed for temperature scaling calculation
        
        combined_logits.append(cur_combined_logits)
        combined_labels.append(cur_combined_labels)
        # print(train_sentences[0])
        # print(permuted_train_sentences[0])
        # print(train_labels[0])
        # print(permuted_train_labels[0])
        # exit()
            
    combined_logits = torch.tensor(np.array(combined_logits))
    combined_labels = torch.tensor(combined_labels)
        
    data = {
        "logits": combined_logits,
        "labels": combined_labels
    }
    
    B, T, num_classes = combined_logits.shape
    
    temperatures = get_shotwise_dynamic_temperatures(data, shots_start=num_shots)
    calibrated_logits = combined_logits[:, -1, :] * temperatures[num_shots].view(B, 1)
    
    return calibrated_logits.cpu().numpy()
    
def get_tc_logits(
    params: Dict, 
    combined_sentences: List[str] | List[np.ndarray],
    train_labels: List[int],
    train_logits: np.ndarray,
    logits: np.ndarray):

    inputs_batch = []
    for i in range(len(logits)):
        inputs = generate_calibrator_inputs(train_labels[i], train_logits[i].copy(), logits[i].copy())
        inputs_batch.append(inputs)
    
    inputs_batch = torch.stack(inputs_batch)
    logits = torch.tensor(logits)
    
    B, T, C = inputs_batch.shape
    assert (B, T) == (len(combined_sentences), len(combined_sentences[-1])), f"{(B, T)} vs {(len(combined_sentences), len(combined_sentences[-1]))} mismatch"
    
    params['tc_input_dim'] = inputs_batch[0].shape[-1]
    calibrated_logits = transformer_calibrate(params, inputs_batch, logits)
    
    return calibrated_logits.cpu().numpy()

def generate_calibrator_inputs(
    train_labels: List[int],
    train_logits: np.ndarray,
    test_logits: np.ndarray):
        
    logits = np.vstack([train_logits, test_logits])
    
    probs = torch.from_numpy(logits).softmax(dim=-1)
    sorted_probs = torch.sort(probs, dim=-1, descending=True).values
    second_highest_probs = sorted_probs[:, [1]].detach().clone().to(torch.float32) 

    probs = probs.numpy()
    preds = np.argmax(probs, axis=-1)
    pred_probs = probs[np.arange(len(preds)), preds]
    
    # repeat last train label to get a dummy test label since the func expects same length list as the other probs and preds
    shifted_features = get_shifted_features(probs, preds, train_labels+train_labels[-1:]) 
    
    # Input to the calibration transformer is a concatenation of t<k shot probs and other features
    inputs = [
        np.concatenate((
            [pred_probs[sent_idx]], 
            [shifted_features.correctness[sent_idx]], 
            [shifted_features.gt_probs[sent_idx]], 
        )).tolist()
        for sent_idx in range(len(pred_probs))
    ]

    inputs = torch.tensor(inputs, dtype=torch.float32)

    gt_prob_mses = torch.cat(
        (
            torch.tensor([[0.25]]), 
            (1 - inputs[1:,[2]])**2
        ), 
        dim=0)
    
    inputs = torch.cat((gt_prob_mses, inputs[:,:3], second_highest_probs), dim=-1)
    
    return inputs

def get_permutation_averaged_logits(params, train_sentences, train_labels, test_sentences, probs, num_permutations=5):
    logger.info(f"Averaging over {num_permutations} predictions with permutations of in-context examples")

    pred_probs = [probs]
    
    for permut in range(num_permutations-1):
        shuffled_idxs = np.random.permutation(params['num_shots'])
        permuted_train_sentences = [[train_sentences[test_idx][train_idx] for train_idx in shuffled_idxs] for test_idx in range(len(test_sentences))]
        permuted_train_labels = [[train_labels[test_idx][train_idx] for train_idx in shuffled_idxs] for test_idx in range(len(test_sentences))]
        
        llm_result = get_results(params, permuted_train_sentences, permuted_train_labels, test_sentences)
        permut_probs = np.array([llm_result.logprobs[i].probs[0] for i in range(len(test_sentences))])
        
        pred_probs.append(permut_probs)
        # print(train_sentences[0])
        # print(permuted_train_sentences[0])
        # print(train_labels[0])
        # print(permuted_train_labels[0])
        # exit()
    permutation_averaged_probs = np.mean(pred_probs, axis=0)
    
    return np.log(permutation_averaged_probs)
    
def get_gc_logits(probs, calib_set_probs, label_sample_set):
    B, num_classes = probs.shape

    label_marginal = np.mean(calib_set_probs, axis=0) # marginalized probs 
    
    test_label_counter = Counter(label_sample_set)
    if len(test_label_counter)==num_classes:
        label_prior = np.array(
            [freq  for label, freq in sorted(list(test_label_counter.items()))]
            )/len(label_sample_set)
    else:
        logger.error(f"Found only {len(test_label_counter)} classes for Generative Calibration in the labels sample set instead of {num_classes}. Defaulting to uniform prior.")
        label_prior = np.full(num_classes, 1/num_classes)

    logprobs = np.log(probs)
    label_prior_logprob = np.log(label_prior)
    label_marginal_logprob = np.log(label_marginal)

    calibrated_logprob = logprobs + label_prior_logprob - label_marginal_logprob

    return calibrated_logprob

def get_bc_logits(logits, calib_set_logits):
    label_marginal = np.mean(calib_set_logits, axis=0) # marginalized logits
    calibrated_logits = logits-label_marginal

    return calibrated_logits

def get_icc_logits(params, train_sentences, train_labels, probs):
    semantic_prior = None
    if params['prompt_shared']:
        semantic_prior = get_prompt_semantic_prior(params, train_sentences[0], train_labels[0])

    calibrated_probs = np.copy(probs)
    for i in range(len(train_sentences)):
        cur_semantic_prior = semantic_prior if semantic_prior is not None else get_prompt_semantic_prior(params, train_sentences[i], train_labels[i])        
        calibrated_probs[i] = probs[i]/cur_semantic_prior
        
    return np.log(calibrated_probs)    # logprobs for logits
    
def get_prompt_semantic_prior(params, sentences, labels):
    train_sentences = [sentences[:sent_idx] + sentences[sent_idx+1:] for sent_idx in range(len(sentences))]
    train_labels = [labels[:sent_idx] + labels[sent_idx+1:] for sent_idx in range(len(sentences))]
    test_sentences = [sentences[sent_idx] for sent_idx in range(len(sentences))]
    
    llm_result = get_results(params, train_sentences, train_labels, test_sentences)
    probs = np.array([llm_result.logprobs[i].probs[0] for i in range(len(test_sentences))])
    
    semantic_prior = np.mean(probs, axis=0)
    
    return semantic_prior 

def eval_accuracy(all_label_probs, test_labels):
    correctness_list, prob_list = [], []
    
    low_prob_count = 0
    mean_min_conf = np.min(all_label_probs, axis=-1).mean()
    
    if mean_min_conf==0:
        raise ValueError(f"Mean min prediction confidence: {mean_min_conf}, check if your label tokens are appropriate. Probs: {all_label_probs[-2]}")
    assert len(all_label_probs) == len(test_labels)
    
    for i, (label_probs, true_label) in enumerate(zip(all_label_probs, test_labels)):
        # print("Label probs:", label_probs)
        if np.max(label_probs)<0.1: 
            low_prob_count += 1
            # logger.warning(f"Your unnormalised probs are sketchy: {label_probs}, check for logical errors")
            # logger.info(raw_resp[i]['logprobs']['top_logprobs'][0])
            # exit()
        # print(label_probs, type(label_probs), label_probs.shape); exit()
        label_probs = label_probs / np.sum(label_probs) # normalize to 1
        ans_conf = np.max(label_probs)
        ans_label = np.argmax(label_probs)
        
        prob_list.append(ans_conf)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    
    logger.info("%d/%d test inputs with low unnormalized label probs", low_prob_count, len(test_labels))

    return np.mean(correctness_list), np.mean(prob_list)

def args_check(args: Dict):
    if args['subsample_test_set'] is None:
        logger.warning("No test set size provided, will use the ENTIRE dataset for ")
        
    calibration_methods = args.get('calibration')
        
    if set(calibration_methods).intersection({CalibrationMethods.GC, CalibrationMethods.BC}):
        assert args.get('entropy_levels')==[EntropyLevels.RANDOM_SHARED] and args['prompt_shared'], "Only shared random ICL prompt supported for now during calibration"
        assert args.get('calibration_set_size') is not None, "Please provide a \"test\" set size to get label marginal from, for calibration"
        assert args.get('calibration_set_size')>0, "Need non empty test set for calibration"
    if CalibrationMethods.TF in calibration_methods:
        assert args.get("calibrator_name"), "Please provide the calibrator model name"
    if args['sampling_strategy']==SamplingStrategy.SIMILARITY:
        assert args.get('prompt_shared')!=True, "Can not share the in-context examples/prompt when similarity sampling"
        
def process_args(args):
    args['models'] = convert_to_list(args['models'])
    args['datasets'] = convert_to_list(args['datasets'])
    args['all_shots'] = convert_to_list(args['all_shots'], int)
    if args.get('calibration'):
        args['calibration'] = convert_to_list(args['calibration'], lambda method : CalibrationMethods[method.upper()])
    else:
        args['calibration'] = []
            
    if args['sampling_strategy']=='similarity':
        args['sampling_strategy'] = SamplingStrategy.SIMILARITY
        logger.warning("Entropy levels need will be ignored for similarity sampling. Setting it to None")
        args['entropy_levels'] = None
        if 0 in args['all_shots']:
            logger.warning("Removing 0 shot from similarity sampling.")
            args['all_shots'].remove(0)
        
        disallowed = {CalibrationMethods.BC, CalibrationMethods.GC}
        args['calibration'] = [method for method in args['calibration'] if method not in disallowed]
                        
    if args.get('entropy_levels'):
        args['sampling_strategy'] = SamplingStrategy.ENTROPY
        args['entropy_levels'] = convert_to_list(args['entropy_levels'], lambda x : EntropyLevels[x.upper()] if x[0].isalpha() else float(x))
        if len(args['entropy_levels'])==0:
            raise ValueError("No entropy levels provided")
        
        if set(args['calibration']).intersection({CalibrationMethods.GC, CalibrationMethods.BC}):
            args['prompt_shared'] = True
    
    args['gpu_ids'] = convert_to_list(args['gpu_ids'], int)
    args['overwrite_type'] = OverWriteType[args['overwrite_type'].upper()]
    
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--model', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--seed_start', dest='seed_start', action='store', required=False, default=0, help='Seed idx to start from', type=int)
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, help='num training examples to use')

    # sampling args
    parser.add_argument('--sampling_strategy', dest='sampling_strategy', action='store', required=True, default="entropy", 
                            choices=["entropy", "similarity"],
                            help='how to sample the ICL examples')
    parser.add_argument('--entropy_levels', dest='entropy_levels', action='store', required=False,
                            help='the levels of entropy for sampling ICL examples')
    parser.add_argument('--prompt_shared', dest='prompt_shared',  action='store_true', required=False, default=False,
                            help='Whether or not to use the same in context examples for the batch of test inputs. \
                                  Not to be used for similarity sampled in context examples.')
    # calibration args
    parser.add_argument('--calibration', dest='calibration', action='store', required=False, 
                            help='calibration strategies for the ICL prompt')
    parser.add_argument('--calibrator_name', dest='calibrator_name', action='store', required=False, help='calibrator model name')
    parser.add_argument('--calibrator_model_path', dest='calibrator_model_path', action='store', required=False, help='calibrator model path')
    parser.add_argument('--calibration_set_size', dest='calibration_set_size', action='store', required=False, type=int,
                            default=100, help='calibration strategy for the ICL prompt')
    
    # inference args
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=True, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--api_num_logprob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=200, help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    
    # flags
    parser.add_argument('--use_saved_results', action='store_true', default=False,
                        help='whether to load the results from pickle files and not run the model')
    parser.add_argument('--overwrite_type', action='store', default="SKIP",
                        help='whether to load the results from pickle files and not run the model')
    
    parser.add_argument('--gpu_ids', dest='gpu_ids', action='store', default="0", required=False, help='Which CUDA gpu to run model on')
    
    parser.add_argument('--start_delay', dest='start_delay', action='store', default=0, required=False, 
                        help='How long to wait in mins before starting the script', type=float)
    
    args = parser.parse_args()
    args = vars(args)

    process_args(args)
    args_check(args)
    
    sleep_time = args['start_delay']
    if sleep_time!=0:
        print(f"Sleeping for {sleep_time} mins before we start...")
    sleep(sleep_time*60)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(gpu_id) for gpu_id in args['gpu_ids']])
    setup_vllm_env_settings()
    from utils.run_utils import *
    
    logger = setup_logger(__name__)
    logger.propagate=False 
    
    main(**args)