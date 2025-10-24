import argparse
import os
import shutil
from data_utils import load_dataset_with_embeddings, set_prompt_params, IclDataset, IclDatasetSplit
from utils import *
from losses import ECELoss
from calibrate.tempture import *
from tqdm import tqdm 
import numpy as np
from time import time
from sampling_strategies import SamplingStrategy
import logging
import multiprocessing as mp

logFormatter = logging.Formatter(
    "{asctime} - {levelname} - {message}", 
    style="{",
    datefmt="%Y-%m-%d %H:%M"
)
logger = logging.getLogger()

fileHandler = logging.FileHandler("./calibration.log")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

logger.setLevel(logging.INFO)

SAVE_DIR_TMP = './raw_logits_high_bs'
os.makedirs(SAVE_DIR_TMP, exist_ok=True)

def save_pickle_tmp(params, data):
    # save results from model
    sampling = params['entropy_level'] if params['sampling_strategy']==SamplingStrategy.ENTROPY else 'similarity'
    file_name = (f"{SAVE_DIR_TMP}/{params['model'].replace('/','_')}/{params['dataset']}/" # In case it's an HF model
                 f"{sampling}/{params['num_shots']}_shot/{params['seed']}_seed.pkl") 
    
    if os.path.isfile(file_name):
        logger.warning("WARNING! overwriting existing saved files")
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    
    logger.info(f"Saved to {file_name}")

def main(models, datasets, all_shots, num_seeds, subsample_test_set, api_num_log_prob, approx, use_saved_results, bs, entropy_levels, half=False, **kwargs):
    """
    Run experiment or load past results, print accuracy
    """
    default_params = {
        'subsample_test_set': subsample_test_set,
        'api_num_log_prob': api_num_log_prob,
        'approx': approx,
        'bs': bs
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
                        if entropy_level not in ("rand", "randshared") and num_shots==0:
                            continue
                        for seed in range(num_seeds):
                            p = deepcopy(default_params)
                            p['model'] = model
                            p['dataset'] = dataset
                            p['num_shots'] = num_shots
                            p['entropy_level'] = entropy_level
                            p['seed'] = seed
                            p['half'] = half
                            p.update(kwargs)
                            p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{p['entropy_level']}_entropy_level_seed{p['seed']}"
                            all_params.append(p)
            elif sampling_strategy==SamplingStrategy.SIMILARITY:
                for num_shots in all_shots:
                    if num_shots==0:
                        continue
                    for seed in range(num_seeds):
                        p = deepcopy(default_params)
                        p['model'] = model
                        p['dataset'] = dataset
                        p['num_shots'] = num_shots
                        p['seed'] = seed
                        p['half'] = half
                        p.update(kwargs)
                        p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_similarity_sampling_seed{p['seed']}"
                        all_params.append(p)

    calibration_method = all_params[0].get('calibration')
    if calibration_method is not None:
        for params in all_params:
            params['expr_name'] = params['expr_name'].replace('_seed', f'_{calibration_method}_seed')

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

        run_and_save_results(all_params, datasets_dict)
        time_taken = time()-start
        logger.info("Time taken to finish: %d hours, %d mins", time_taken//3600, (time_taken%3600)//60)

def run_and_save_results(params_list: List[Dict], datasets: Dict):#, min_confidence_limits: np.ndarray):
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """
    result_tree = dict()
    ece_loss = ECELoss(n_bins=30)

    for param_index, params in tqdm(enumerate(params_list), total=len(params_list), desc="Processing experiments"):
        file_name = os.path.join(SAVE_DIR_TMP, f"{params['expr_name'].replace('/','_')}.pkl")
        # if os.path.isfile(file_name):
        #     logging.info("Skipping experiment, already done before.")
        #     continue
        logger.info(params)
        logger.info("\nExperiment name: %s", params['expr_name'])
        
        ### load data
        start = time()
        dataset: IclDataset = datasets[params['dataset']]
        all_train_sentences, all_train_labels, all_train_embeddings = dataset.train.sentences, dataset.train.labels, dataset.train.embeddings 
        all_test_sentences, all_test_labels, all_test_embeddings = dataset.test.sentences, dataset.test.labels, dataset.test.embeddings
        
        logger.info("Train sizes: %d sentences, %d labels, %d embeddings | Test sizes: %d sentences, %d labels, %d embeddings",
                len(all_train_sentences), len(all_train_labels), len(all_train_embeddings),
                len(all_test_sentences), len(all_test_labels), len(all_test_embeddings)
        )
        # params_check(params)

        ### sample few-shot training examples
        # np.random.seed(params['seed'])
        num_shots = params['num_shots'] 
        calibration_method = params.get('calibration')

        if params['sampling_strategy']==SamplingStrategy.ENTROPY:
            # np.random.seed(0)
            test_sentences, test_labels, test_embeddings = get_test_data(all_test_sentences, all_test_labels, all_test_embeddings, params['subsample_test_set'])
            # np.random.seed(params['seed'])
            train_sentences, train_labels = [], []
            train_embeddings = [] # used in transformer calibration 
            
            for i in range(len(test_sentences)):
                test_label = test_labels[i] if params['entropy_level'] in ('labelspike', 'labelsuppress') else None
                selected_sentences, selected_labels, selected_idxs = random_sampling(all_train_sentences, all_train_labels,
                                                                      num_shots, params['entropy_level'], test_label) 
                if calibration_method in ('GC', 'GC_unif', 'BC'):
                    calibration_test_sentences, calibration_test_labels = [], [] 

                    remaining_sentences = [sentence for i, sentence in enumerate(all_train_sentences) if i not in set(selected_idxs)]
                    remaining_labels = [label for i, label in enumerate(all_train_labels) if i not in set(selected_idxs)]
                   
                    if calibration_method in ('GC', 'BC'):
                        calib_set_entropy = 'rand'
                    elif calibration_method == 'GC_unif':
                        calib_set_entropy = 'max' # balanced/uniform distribution for labels to not get class imbalance bias 
                    
                    calibration_test_sentences, calibration_test_labels, _ = random_sampling(remaining_sentences, remaining_labels,
                                                                                              params['calibration_set_size'], calib_set_entropy)
                    # sharing the k-shot examples is expected for GC, BC
                    train_sentences = [selected_sentences]*len(test_sentences)
                    train_labels = [selected_labels]*len(test_sentences)
                    break
                elif calibration_method=='TC':
                    train_embeddings.append([all_train_embeddings[idx] for idx in selected_idxs])
                    # print(train_embeddings, selected_idxs); exit()
                elif calibration_method is None:
                    pass
                else:
                    raise NotImplementedError(f"{calibration_method} calibration method not defined")
                                        
                if params['entropy_level']=='randshared':
                    train_sentences = [selected_sentences]*len(test_sentences)
                    train_labels = [selected_labels]*len(test_sentences)
                    break
                
                train_sentences.append(selected_sentences)
                train_labels.append(selected_labels)

        elif params['sampling_strategy']==SamplingStrategy.SIMILARITY:
            shuffle_examples = True
            if params['subsample_test_set']>=len(all_test_sentences) and not shuffle_examples and params['seed']>0:
                logger.warning("Found less test inputs than requested, saving seed 0 results to avoid repeated experiments")
                model_name = params['expr_name'].replace('/','_') # In case it's an HF model
                file_name = os.path.join(SAVE_DIR_TMP, f"{model_name}.pkl")
                shutil.copy(file_name.replace(f"seed{params['seed']}", 'seed0'), file_name)
                continue
            test_sentences, test_labels, test_embeddings = get_test_data(all_test_sentences, all_test_labels, all_test_embeddings, params['subsample_test_set'])

            sampled_data = similarity_sampling(all_train_sentences, all_train_embeddings, all_train_labels, 
                                               test_embeddings, num_shots, test_sentences, 
                                               shuffle=shuffle_examples, return_embeddings=True)
            train_sentences, train_embeddings, train_labels = sampled_data.sentences, sampled_data.embeddings, sampled_data.labels
            
        logger.info("Time taken to load %s dataset: %d sec", params['dataset'], round(time()-start))
        # print(np.array(train_embeddings).shape, np.array(test_embeddings).shape) ; exit()
        # for prompt construction
        set_prompt_params(params)   

        ### Evaluate the performance and save all results
        logger.info(f"getting raw resp for {len(test_sentences)} test sentences with {num_shots} ICL examples.")
        
        all_label_probs, all_label_raw_logits = get_results(params, train_sentences, train_labels, test_sentences)
        # print(all_label_probs[-1]); exit()
        # raw_resp_test2, all_label_probs2, all_label_raw_logits2 = get_results(params, train_sentences, train_labels, test_sentences)

        # print(all_label_probs[-1][0])
        # # assert raw_resp_test==raw_resp_test2
        # assert np.array_equal(all_label_probs, all_label_probs2), f"{all_label_probs[-10:], all_label_probs2[-10:]}"
        # assert np.array_equal(all_label_raw_logits,all_label_raw_logits2) 
        # assert np.array_equal(np.argmax(all_label_probs, axis=-1), np.argmax(all_label_raw_logits, axis=-1)), f"Probs: {all_label_probs[-5:]}, Logits: {all_label_raw_logits[-5:]}"
        # cur_rep = np.stack([resp['logprobs']['hidden_states'][-1].numpy() for resp in raw_resp_test])
        # norm = np.linalg.norm(cur_rep, axis=1)  
        # norm = [norm.mean()]

        acc_original, conf_ori = eval_accuracy(all_label_probs, test_labels)
        ece_original = ece_loss(all_label_raw_logits, test_labels)
        accuracies = [acc_original]
        eces = [ece_original.item()]
        confs = [conf_ori]

        if calibration_method is not None:
            if 'TC' in calibration_method:
                combined_sentences = [train_sentences[i] + [test_sentences[i]] for i in range(len(test_sentences))]
                combined_embeddings = np.array([np.vstack((train_embeddings[i], [test_embeddings[i]])) for i in range(len(test_sentences))])
                combined_labels = [train_labels[i] + [test_labels[i]] for i in range(len(test_sentences))]
                
                # print(all_train_embeddings.shape, train_embeddings.shape, combined_embeddings[0].shape); exit()
                calibrated_label_probs = get_tc_probs(params, combined_sentences, combined_embeddings, combined_labels, all_label_raw_logits)
                # assert np.array_equal(
                #     np.argmax(all_label_raw_logits, axis=-1),
                #     np.argmax(all_label_probs, axis=-1)
                # )
                # assert np.array_equal(
                #     np.argmax(calibrated_label_probs, axis=-1),
                #     np.argmax(all_label_probs, axis=-1)
                # )
            else:
                label_probs, label_raw_logits = get_results(params, train_sentences, train_labels, calibration_test_sentences)

                if 'GC' in calibration_method:
                    calibrated_label_probs = get_gc_probs(params, all_label_probs, label_probs, calibration_test_labels)
                elif 'BC' in calibration_method:
                    calibrated_label_probs = get_bc_probs(all_label_raw_logits, label_raw_logits)
            
            acc_calibrated, conf_calibrated = eval_accuracy(calibrated_label_probs, test_labels)
            ece_calibrated = ece_loss(calibrated_label_probs, test_labels, input_type='probs')
            
            # cur_rep = np.stack([resp['logprobs']['hidden_states'][-1].numpy() for resp in raw_resp_test])
            # norm_calibrated = np.linalg.norm(cur_rep, axis=1)  
            # norm_calibrated = norm_calibrated.mean()     
            accuracies.append(acc_calibrated)
            eces.append(ece_calibrated.item())
            # norm.append(norm_calibrated)
            confs.append(conf_calibrated)

        print(f"Accuracies: {accuracies}")
        print(f"Ece      : {eces}")
        # print(f"Norm      : {norm}")
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
        if not f"{keys[3]}_diff" in entropy_node.keys():
            entropy_node[f"{keys[3]}_diff"] = dict()
        # if not f"{keys[3]}_norm" in entropy_node.keys():
        #     entropy_node[f"{keys[3]}_norm"] = dict()
        if not f"{keys[3]}_conf" in entropy_node.keys():
            entropy_node[f"{keys[3]}_conf"] = dict()
        entropy_node[f"{keys[3]}_ece"][seed] = eces
        entropy_node[f"{keys[3]}_conf"][seed] = confs
        # entropy_node[f"{keys[3]}_norm"][seed] = norm

        # save to file
        result_to_save = dict()
        params_to_save = deepcopy(params)
        result_to_save['params'] = params_to_save
        result_to_save['all_label_probs'] = all_label_probs
        result_to_save['raw_logits'] = all_label_raw_logits
        result_to_save['all_labels_prob_mass'] = np.sum(all_label_probs, axis=-1)
        result_to_save['eces'] = eces
        # result_to_save['norm'] = norm
        result_to_save['confs'] = confs
        result_to_save['accuracies'] = accuracies
        if params.get('calibration'):
            result_to_save['calibrated_label_probs'] = calibrated_label_probs
            result_to_save['calibrated_eces'] = [ece_calibrated.item()]
            result_to_save['calibrated_confs'] = [conf_calibrated]
            result_to_save['calibrated_accuracies'] = [acc_calibrated]
        if 'prompt_func' in result_to_save['params'].keys():
            params_to_save['prompt_func'] = None
        print_results(result_tree)
        save_pickle_tmp(params, result_to_save)
        # exit()
    print_results(result_tree, log=logger.info)

# outdated
def calibrate_probs(params, all_label_probs, all_label_raw_logits, train_sentences, train_labels, test_sentences, test_labels):
    label_probs, label_raw_logits = get_results(params, train_sentences, train_labels, test_sentences)

    method = params['calibration']
    if 'GC' in method:
        return get_gc_probs(params, all_label_probs, label_probs, test_labels)
    elif 'BC' in method:
        return get_bc_probs(all_label_raw_logits, label_raw_logits)

    
def get_gc_probs(params, all_label_probs, label_probs, test_labels):
    all_label_probs = all_label_probs/np.sum(all_label_probs, axis=-1, keepdims=True) # normalize
    label_probs = label_probs/np.sum(label_probs, axis=-1, keepdims=True) # normalize
    label_marginal = np.mean(label_probs, axis=0) # marginalized probs 
    if params['calibration'] == 'GC':
        test_label_counter = Counter(test_labels)
        label_prior = np.array(
            [freq  for label, freq in sorted(list(test_label_counter.items()))]
            )/len(test_labels)
    elif params['calibration'] == 'GC_unif':
        label_prior = np.full(len(params['label_dict']), 1/len(params['label_dict']))

    all_label_logprobs = np.log(all_label_probs)
    label_prior_logprob = np.log(label_prior)
    label_marginal_logprob = np.log(label_marginal)

    calibrated_label_logprob = torch.from_numpy(all_label_logprobs + label_prior_logprob - label_marginal_logprob)
    calibrated_label_probs = torch.softmax(calibrated_label_logprob, dim=-1)

    return calibrated_label_probs.numpy()

def get_bc_probs(all_label_raw_logits, label_raw_logits):
    label_marginal = np.mean(label_raw_logits, axis=0) # marginalized logits

    calibrated_label_logits = torch.from_numpy(all_label_raw_logits-label_marginal)
    calibrated_label_probs = torch.softmax(calibrated_label_logits, dim=-1)

    return calibrated_label_probs.numpy()

def get_tc_probs(params: Dict, sentences: List[str], embeddings: np.ndarray, labels: List[int], original_logits: np.ndarray):
    if isinstance(original_logits, np.ndarray):
        original_logits = torch.tensor(original_logits, dtype=torch.float32)
        
    batch = {"inputs":[], "logits":[]}
    
    for i in range(len(sentences)):
        data = generate_data_class_agnostic(params, sentences[i], embeddings[i], labels[i])
        batch['inputs'].append(data['inputs'])
        batch['logits'].append(data['logits'])
    
    batch = {
        "inputs": torch.stack(batch['inputs']),
        "logits": torch.stack(batch['logits'])
    }
    # batch['logits'][:,-1,:] = original_logits
      
    params['input_dim'] = batch['inputs'][0].shape[-1]
    calibrated_logits = transformer_calibrate(params, batch)
    calibrated_probs = torch.softmax(calibrated_logits, dim=-1)
    
    if not torch.equal(torch.argmax(batch['logits'][:,-1,:], dim=-1), torch.argmax(original_logits, dim=-1)):
        print(f"Shape: {batch['logits'][:,-1,:].shape}, {original_logits.shape}. Logits: {batch['logits'][:,-1,:][-5:, : ]}, {original_logits[-5:, : ]}")
    # assert torch.equal(torch.argmax(calibrated_logits, dim=-1), torch.argmax(original_logits, dim=-1)), f"Shape: {calibrated_logits.shape}, {original_logits.shape}. Logits: {calibrated_logits[-1:, : ]}, {original_logits[-1:, : ]}"
    # print(original_logits[-2:,:], calibrated_logits[-2:,:]) ;exit()
    return calibrated_probs.numpy()
    
# Different feature set, made for causal temperature regression
def generate_data2(params: Dict, sentences: List[str], embeddings: np.ndarray, labels: List[int]):
    data = {
        "inputs" : [],
        "logits": [],
    }
    
    num_classes = len(params['label_dict'])
    shifted_onehot_labels = np.zeros((len(labels), num_classes)) # T,num_classes
    shifted_onehot_labels[0] = np.array([1/num_classes]*num_classes)

    for i in range(len(labels)-1):
        shifted_onehot_labels[i+1][labels[i]] = 1
    
    similarity_vectors = np.zeros((len(embeddings), params['tc_input_dim']-2*num_classes))
    # print(similarity_vectors.shape)
    similarity_vectors[:, :len(embeddings)] += np.tril(get_similarities(embeddings, embeddings)) # only upper triangular to make it causal
    # print(similarity_vectors.shape, embeddings.shape, params['tc_input_dim'])
    
    # LLM's logits(y|x, C) to be calibrated by transformer's output T
    train_sentences = [sentences[:sent_idx] for sent_idx in range(len(sentences))]
    train_labels = [labels[:sent_idx] for sent_idx in range(len(sentences))]
    test_sentences = [sentences[sent_idx] for sent_idx in range(len(sentences))]
    
    all_label_probs, all_label_raw_logits = get_results(params, train_sentences, train_labels, test_sentences)
    # Input to the calibration transformer is a concatenation of t<k shot probs and the similarities
    # print(all_label_probs.shape, shifted_onehot_labels.shape, similarity_vectors.shape); exit()
    data['inputs'] = np.array(
        [
            np.concat(
                (all_label_probs[sent_idx], shifted_onehot_labels[sent_idx], similarity_vectors[sent_idx])
            ) 
            for sent_idx in range(len(sentences))
        ]
    )
    data['inputs'] = torch.tensor(data['inputs'], dtype=torch.float32)
    data['logits'] = torch.tensor(all_label_raw_logits, dtype=torch.float32)

    # print(data['inputs'].shape)
    # exit()
    return data

# Different feature set, made for causal temperature regression | class agnostic
def generate_data_class_agnostic(params: Dict, sentences: List[str], embeddings: np.ndarray, labels: List[int]):
    data = {
        "inputs" : [],
        "logits": [],
    }
    
    num_classes = len(params['label_dict'])

    # LLM's logits(y|x, C) to be calibrated by transformer's output T
    train_sentences = [sentences[:sent_idx] for sent_idx in range(len(sentences))]
    train_labels = [labels[:sent_idx] for sent_idx in range(len(sentences))]
    test_sentences = [sentences[sent_idx] for sent_idx in range(len(sentences))]
    
    all_label_probs, all_label_raw_logits = get_results(params, train_sentences, train_labels, test_sentences)

    shifted_correctness = np.zeros((len(labels),)) # num_classes, 
    shifted_correctness[0] = 0.5
        
    row_sums = np.sum(all_label_probs, axis=-1, keepdims=True)

    if np.any(row_sums == 0):
        print("Zero probs in here! :", all_label_probs)
        print(sentences)
        print(labels)    
        
    # if np.any(row_sums < 0.1):
    #     print("Near zero probs in here! :", all_label_probs)
    #     print(sentences)
    #     print(labels)    
        
    probs = all_label_probs/row_sums
    preds = np.argmax(probs, axis=-1)
    pred_probs = probs[np.arange(len(preds)), preds]
    shifted_gt_probs = np.concatenate((
        np.array([-1]), 
        probs[np.arange(len(preds)-1), labels[:-1]]
    )) # exclude last position and shift

    for i in range(len(labels)-1):
        shifted_correctness[i+1] = int(preds[i]==labels[i])

    # normalized_entropies = np.array([
    #     probs[sent_idx]@np.log(probs[sent_idx].T) 
    #     for sent_idx in range(len(sentences))
    # ]) / np.log(num_classes)
    normalized_entropies = -(probs@np.log(probs.T + 1e-9)).diagonal() / np.log(num_classes) # (B, num_classes) × (num_classes, B) --> (B, B) --> diagonal elements

    similarity_vectors = np.zeros((len(embeddings), params['tc_input_dim']-4))
    # print(similarity_vectors.shape)
    similarity_vectors[:, :len(embeddings)] += np.tril(get_similarities(embeddings, embeddings)) # only upper triangular to make it causal
    
    # Input to the calibration transformer is a concatenation of t<k shot probs and the similarities
    # print(pred_probs.shape, shifted_correctness.shape, similarity_vectors.shape, embeddings.shape)

    data['inputs'] = np.array([
        np.concatenate((
            [pred_probs[sent_idx]], 
            [shifted_correctness[sent_idx]], 
            [shifted_gt_probs[sent_idx]], 
            [normalized_entropies[sent_idx]], 
            similarity_vectors[sent_idx]
        ))
        for sent_idx in range(len(sentences))
    ])

    data['inputs'] = torch.tensor(data['inputs'], dtype=torch.float32)
    data['logits'] = torch.tensor(all_label_raw_logits, dtype=torch.float32)
    
    # print(data['inputs'])
    # exit()
    return data

def eval_accuracy(all_label_probs, test_labels):
    correctness_list, prob_list = [], []
    low_prob_count = 0
    mean_min_conf = np.min(all_label_probs, axis=-1).mean()
    if mean_min_conf==0:
        logger.warning(f"Mean min prediction confidence: {mean_min_conf}, check if your label tokens are appropriate. Probs: {all_label_probs}")
    assert len(all_label_probs) == len(test_labels)
    for i, (label_probs, true_label) in enumerate(zip(all_label_probs, test_labels)):
        # print("Label probs:", label_probs)
        if np.max(label_probs)<0.1: 
            low_prob_count += 1
            # logger.warning(f"Your unnormalised probs are sketchy: {label_probs}, check for logical errors")
            # logger.info(raw_resp[i]['logprobs']['top_logprobs'][0])
            # exit()
        # print(label_probs, type(label_probs), label_probs.shape)
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
    calibration_method = args.get('calibration')
    if calibration_method in ('GC', 'BC', 'GC_unif'):
        assert args.get('entropy_levels')==['randshared'], "Only shared random ICL prompt supported for now during calibration"
        assert args.get('calibration_set_size') is not None, "Please provide a \"test\" set size to get label marginal from, for calibration"
        assert args.get('calibration_set_size')>0, "Need non empty test set for calibration"

if __name__ == '__main__':
    # vllm stuff
    # mp.set_start_method('spawn', force=True)
    # setup_single_threading()
    setup_vllm_env_settings()
    
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--model', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, help='num training examples to use')

    # sampling args
    parser.add_argument('--sampling_strategy', dest='sampling_strategy', action='store', required=True, default="entropy", 
                            choices=["entropy", "similarity"],
                            help='how to sample the ICL examples')
    parser.add_argument('--entropy_levels', dest='entropy_levels', action='store', required=False,
                            # default="rand", 
                            # choices=["rand", "rand_shared", "max", "labelspike", "labelsuppress"],
                            help='the levels of entropy for sampling ICL examples')
    
    # calibration args
    parser.add_argument('--calibration', dest='calibration', action='store', required=False, 
                            choices=["GC", "GC_unif", "BC", "TC"],
                            help='calibration strategy for the ICL prompt')
    parser.add_argument('--calibrator_model_path', dest='calibrator_model_path', action='store', required=False, help='calibrator model path')
    parser.add_argument('--calibration_set_size', dest='calibration_set_size', action='store', required=False, type=int,
                            default=100, help='calibration strategy for the ICL prompt')
    parser.add_argument('--tc_input_dim', dest='tc_input_dim', action='store', required=False, type=int,
                        default=None, help='calibrator input dims')
    
    # inference args
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=True, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--api_num_log_prob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=200, help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    # flags
    parser.add_argument('--use_saved_results', dest='use_saved_results', action='store_const', const=True, default=False,
                        help='whether to load the results from pickle files and not run the model')
    parser.add_argument('--approx', dest='approx', action='store_const', const=True, default=False,
                        help='whether to set token prob to zero if not in top 100')
    
    parser.add_argument('--half', action='store_true', default=True,
                        help='Use float 16')
    parser.add_argument('--gpu_id', dest='gpu_id', action='store', default=0, required=False, help='Which CUDA gpu to run model on', type=int)
    
    args = parser.parse_args()
    args = vars(args)
    # print(args)

    args['models'] = convert_to_list(args['models'])
    args['datasets'] = convert_to_list(args['datasets'])
    args['all_shots'] = convert_to_list(args['all_shots'], int)
    if args.get('entropy_levels'):
        args['sampling_strategy'] = SamplingStrategy.ENTROPY
        args['entropy_levels'] = convert_to_list(args['entropy_levels'], lambda x : x if x.isalpha() else float(x))
        if len(args['entropy_levels'])==0:
            raise ValueError("No entropy levels provided")
    if args['sampling_strategy']=='similarity':
        args['sampling_strategy'] = SamplingStrategy.SIMILARITY
        logger.warning("Entropy levels need will be ignored for similarity sampling. Setting it to rand")
        args['entropy_levels'] = ["rand"]
        if 0 in args['all_shots']:
            logger.warning("Removing 0 shot from similarity sampling.")
            args['all_shots'].remove(0)

    args_check(args)
    main(**args)