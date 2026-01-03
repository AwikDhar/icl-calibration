import argparse
import os
import shutil
from tqdm import tqdm 
import numpy as np
from time import time
import multiprocessing as mp

from data_utils import load_dataset_with_embeddings, set_prompt_params, IclDataset, IclDatasetSplit
from utils import *
from metrics import ClassificationMetrics
from losses import smooth_ece

from calibration_methods import CalibrationMethods
from sampling_strategies import SamplingStrategy
from calibration.generate_calibration_dataset import get_autoregressive_results, get_shifted_features, generate_data_class_agnostic_with_embeddings

import logging

logFormatter = logging.Formatter(
    "{asctime} - {levelname} - {message}", 
    style="{",
    datefmt="%Y-%m-%d %H:%M"
)
logger = logging.getLogger(__name__)

fileHandler = logging.FileHandler(ROOT_DIR/"calibration.log")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

logger.setLevel(logging.INFO)

SAVE_DIR_TMP = ROOT_DIR/"saved_results"
os.makedirs(SAVE_DIR_TMP, exist_ok=True)

def save_pickle_tmp(params, data):
    # save results from model
    sampling = params['entropy_level'].name if params['sampling_strategy']==SamplingStrategy.ENTROPY else 'similarity'
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
                        if entropy_level!=EntropyLevels.RANDOM and num_shots==0:
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

    for _, params in tqdm(enumerate(params_list), total=len(params_list), desc="Processing experiments"):
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
        # all_sentences, all_labels, all_embeddings = dataset.train.sentences, dataset.train.labels, dataset.train.embeddings 
        # split_size = int(0.8*len(all_sentences))
        # all_train_sentences, all_train_labels, all_train_embeddings = all_sentences[:split_size], all_labels[:split_size], all_embeddings[:split_size]
        # all_test_sentences, all_test_labels, all_test_embeddings = all_sentences[split_size:], all_labels[split_size:], all_embeddings[split_size:]
        
        logger.info("Train sizes: %d sentences, %d labels, %d embeddings | Test sizes: %d sentences, %d labels, %d embeddings",
                len(all_train_sentences), len(all_train_labels), len(all_train_embeddings),
                len(all_test_sentences), len(all_test_labels), len(all_test_embeddings)
        )

        ### sample few-shot training examples
        num_shots = params['num_shots'] 
        calibration_methods = params.get('calibration')
        has_calibration_set = set(calibration_methods).intersection({CalibrationMethods.GC, CalibrationMethods.BC})             

        train_sentences, train_labels, train_embeddings = [], [], [] 
            
        if params['sampling_strategy']==SamplingStrategy.ENTROPY:
            test_sentences, test_labels, test_embeddings = get_test_data(all_test_sentences, all_test_labels, all_test_embeddings, params['subsample_test_set'])

            
            for i in range(len(test_sentences)):
                selected_sentences, selected_labels, selected_idxs = random_sampling(all_train_sentences, all_train_labels,
                                                                      num_shots, params['entropy_level'], test_labels[i]) 
                                   
                if has_calibration_set:
                    calib_set_sentences, calib_set_labels = [], [] 

                    remaining_sentences = [sentence for i, sentence in enumerate(all_train_sentences) if i not in set(selected_idxs)]
                    remaining_labels    = [label    for i, label    in enumerate(all_train_labels)    if i not in set(selected_idxs)]
                   
                    calib_set_sentences, calib_set_labels, _ = random_sampling(remaining_sentences, remaining_labels,
                                                                                              params['calibration_set_size'], EntropyLevels.RANDOM)
                
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
            shuffle_examples = True
            if params['subsample_test_set']>=len(all_test_sentences) and not shuffle_examples and params['seed']>0:
                logger.warning("Found less test inputs in dataset than requested, saving seed 0 results to avoid repeated experiments")
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
        
        probs, logits = get_results(params, train_sentences, train_labels, test_sentences)
        # print(all_label_probs[-1]); exit()
        # raw_resp_test2, all_label_probs2, all_label_raw_logits2 = get_results(params, train_sentences, train_labels, test_sentences)

        # print(all_label_probs[-1][0])
        # assert np.array_equal(all_label_probs, all_label_probs2), f"{all_label_probs[-10:], all_label_probs2[-10:]}"
        # assert np.array_equal(all_label_raw_logits,all_label_raw_logits2) 
        # assert np.array_equal(np.argmax(all_label_probs, axis=-1), np.argmax(all_label_raw_logits, axis=-1)), f"Probs: {all_label_probs[-5:]}, Logits: {all_label_raw_logits[-5:]}"

        acc_original, conf_ori = eval_accuracy(probs, test_labels)
        ece_original = smooth_ece(logits, test_labels)
        accuracies = [acc_original]
        eces = [ece_original.item()]
        confs = [conf_ori]

        metrics = {}
        if calibration_methods is not None:
            if set(calibration_methods).intersection({CalibrationMethods.TF, CalibrationMethods.ICT}):
                train_probs, train_logits = get_autoregregressive_train_results(params, train_sentences, train_labels)
                combined_logits = np.concatenate([train_logits, np.expand_dims(logits, axis=1)], axis=1)
                combined_labels = np.array([train_labels[i] + [test_labels[i]] for i in range(len(test_sentences))])
            
            if CalibrationMethods.TF in calibration_methods:
                combined_sentences = [train_sentences[i] + [test_sentences[i]] for i in range(len(test_sentences))]
                combined_embeddings = np.array([np.vstack((train_embeddings[i], [test_embeddings[i]])) for i in range(len(test_sentences))])
                
                # print(all_train_embeddings.shape, train_embeddings.shape, combined_embeddings[0].shape); exit()
                # transformer_calibrated_logits = get_tc_logits_tmp(params, combined_sentences, combined_embeddings, combined_labels)
                transformer_calibrated_logits = get_tc_logits(params, train_sentences, train_embeddings, train_labels, train_logits, logits)
                metrics[CalibrationMethods.TF] = ClassificationMetrics(logits, transformer_calibrated_logits, test_labels)
                # assert np.array_equal(
                #     np.argmax(logits, axis=-1),
                #     np.argmax(probs, axis=-1)
                # )
                # assert np.array_equal(
                #     np.argmax(calibrated_probs, axis=-1),
                #     np.argmax(probs, axis=-1)
                # )
            if CalibrationMethods.ICT in calibration_methods:
                ict_calibrated_logits = get_ict_logits(params, combined_logits, combined_labels)
                metrics[CalibrationMethods.ICT] = ClassificationMetrics(logits, ict_calibrated_logits, test_labels)
                
            if CalibrationMethods.ICC in calibration_methods:
                incontext_calibrated_logits = get_icc_logits(params, train_sentences, train_labels, probs)
                metrics[CalibrationMethods.ICC] = ClassificationMetrics(logits, incontext_calibrated_logits, test_labels)
            
            if has_calibration_set:
                calib_set_probs, calib_set_logits = get_results(params, train_sentences, train_labels, calib_set_sentences)

                if CalibrationMethods.GC in calibration_methods:
                    generative_calibrated_logits = get_gc_logits(probs, calib_set_probs, calib_set_labels)
                    metrics[CalibrationMethods.GC] = ClassificationMetrics(logits, generative_calibrated_logits, test_labels)
                if CalibrationMethods.BC in calibration_methods:
                    batch_calibrated_logits = get_bc_logits(logits, calib_set_logits)
                    metrics[CalibrationMethods.BC] = ClassificationMetrics(logits, batch_calibrated_logits, test_labels)

            accuracies.extend([metrics[method].calibrated_accuracy for method in calibration_methods])
            eces.extend([metrics[method].calibration_metrics.calibrated_ece for method in calibration_methods])
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
        if not f"{keys[3]}_conf" in entropy_node.keys():
            entropy_node[f"{keys[3]}_conf"] = dict()

        entropy_node[f"{keys[3]}_ece"][seed] = eces
        entropy_node[f"{keys[3]}_conf"][seed] = confs

        # save to file
        result_to_save = dict()
        params_to_save = deepcopy(params)
        
        result_to_save['params'] = params_to_save
        result_to_save['probs'] = probs
        result_to_save['logits'] = logits
        
        result_to_save['eces'] = eces
        result_to_save['confs'] = confs
        result_to_save['accuracies'] = accuracies
        result_to_save['metrics'] = metrics
            
        print_results(result_tree, calibration_methods)
        save_pickle_tmp(params, result_to_save)
        # exit()
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

def get_tc_logits_tmp(params: Dict, sentences: List[str], embeddings: np.ndarray, labels: List[int]):
    from calibration.data_utils import recalculate_features
    
    inputs_batch = []
    logits_batch = []
    
    for i in range(len(sentences)):
        data = generate_data_class_agnostic_with_embeddings(params, sentences[i], embeddings[i], labels[i])
        data = {
            'inputs':torch.tensor(data['inputs']),
            'logits':torch.tensor(data['logits'])
        }
        recalculate_features(data)
        inputs_batch.append(data['inputs'])
        logits_batch.append(data['logits'])
        
    inputs_batch = torch.stack(inputs_batch)
    logits_batch = torch.stack(logits_batch)
    
    B, T, C = inputs_batch.shape
    assert (B, T) == (len(sentences), len(sentences[-1])), f"{(B, T)} vs {(len(sentences), len(sentences[-1]))} mismatch"
    
    params['tc_input_dim'] = inputs_batch[0].shape[-1]
    calibrated_logits = transformer_calibrate_tmp(params, inputs_batch, logits_batch)
    
    return calibrated_logits.cpu().numpy()

def get_tc_logits(params: Dict, train_sentences: List[str], train_embeddings: np.ndarray, train_labels: List[int], train_logits: np.ndarray, logits: np.ndarray):
    inputs_batch = []
    for i in range(len(logits)):
        inputs = generate_calibrator_inputs(params, train_sentences[i], train_embeddings[i], train_labels[i], train_logits[i].copy(), logits[i].copy())
        inputs_batch.append(inputs)
    
    inputs_batch = torch.stack(inputs_batch)
    logits = torch.tensor(logits)
    
    B, T, C = inputs_batch.shape
    assert (B, T) == (len(train_sentences), len(train_sentences[-1])+1), f"{(B, T)} vs {(len(train_sentences), len(train_sentences[-1])+1)} mismatch"
    
    params['tc_input_dim'] = inputs_batch[0].shape[-1]
    calibrated_logits = transformer_calibrate(params, inputs_batch, logits)
    
    return calibrated_logits.cpu().numpy()

# Different feature set, made for causal temperature regression | class agnostic
def generate_calibrator_inputs(params: Dict, train_sentences: List[str], train_embeddings: np.ndarray, train_labels: List[int], train_logits: np.ndarray, test_logits: np.ndarray):    
    if train_logits is None:
        train_probs, train_logits = get_autoregressive_results(params, train_sentences, train_labels)   
    
    logits = np.vstack([train_logits, test_logits])
    
    probs = torch.from_numpy(logits).softmax(dim=-1).numpy()
    # print(probs.shape)
        
    preds = np.argmax(probs, axis=-1)
    pred_probs = probs[np.arange(len(preds)), preds]
    
    # repeat last train label to get a dummy test label since the func expects same length list as the other probs and preds
    shifted_features = get_shifted_features(probs, preds, train_labels+train_labels[-1:]) 

    # input_similarity_vectors = np.tril(get_similarities(train_embeddings, train_embeddings)) # only lower triangular to make it causal
    
    # lower_dim_embeddings = get_embeddings(params, train_sentences)
    
    # T = len(train_sentences) # number of timesteps
    # pred_similarity_vectors = np.zeros((T, T))
    # for timestep in range(T):
    #     for prev_timestep in range(timestep + 1):  # only compute for j <= i (causal)
    #         js_div = js_divergence(probs[timestep], probs[prev_timestep])
    #         pred_similarity_vectors[timestep, prev_timestep] = 1 - (js_div / np.log(2))  # Maps [0, log(2)] → [1, 0]

    # Input to the calibration transformer is a concatenation of t<k shot probs, similarities and other features
    inputs = [
        np.concatenate((
            [pred_probs[sent_idx]], 
            [shifted_features.correctness[sent_idx]], 
            [shifted_features.gt_probs[sent_idx]], 
            # pred_similarity_vectors[sent_idx], 
            # input_similarity_vectors[sent_idx],
            # lower_dim_embeddings[sent_idx]
        )).tolist()
        for sent_idx in range(len(pred_probs))
    ]

    inputs = torch.tensor(inputs, dtype=torch.float32)

    gt_prob_mses = torch.cat(
        (
            torch.tensor([[0.5]]), 
            (1 - inputs[1:,[2]])**2
        ), 
        dim=0)
    inputs = torch.cat((gt_prob_mses, inputs[:,:3]), dim=-1)
    
    # print(data['inputs']); exit()
    return inputs

def get_gc_logits(probs, calib_set_probs, calib_set_labels):
    label_marginal = np.mean(calib_set_probs, axis=0) # marginalized probs 

    test_label_counter = Counter(calib_set_labels)
    label_prior = np.array(
        [freq  for label, freq in sorted(list(test_label_counter.items()))]
        )/len(calib_set_labels)
    # label_prior = np.full(len(params['label_dict']), 1/len(params['label_dict']))

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
    
    probs, logits = get_results(params, train_sentences, train_labels, test_sentences)
    semantic_prior = np.mean(probs, axis=0)
    
    return semantic_prior 

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
        
    calibration_methods = args.get('calibration')
        
    if set(calibration_methods).intersection({CalibrationMethods.GC, CalibrationMethods.BC}):
        assert args.get('entropy_levels')==[EntropyLevels.RANDOM] and args['prompt_shared'], "Only shared random ICL prompt supported for now during calibration"
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
    args['calibration'] = convert_to_list(args['calibration'], lambda method : CalibrationMethods[method.upper()])
    
    if args['sampling_strategy']=='similarity':
        args['sampling_strategy'] = SamplingStrategy.SIMILARITY
        logger.warning("Entropy levels need will be ignored for similarity sampling. Setting it to None")
        args['entropy_levels'] = None
        if 0 in args['all_shots']:
            logger.warning("Removing 0 shot from similarity sampling.")
            args['all_shots'].remove(0)
        
        disallowed = {CalibrationMethods.BC, CalibrationMethods.GC}
        args['calibration'] = list(set(args['calibration']) - disallowed)
                        
    if args.get('entropy_levels'):
        args['sampling_strategy'] = SamplingStrategy.ENTROPY
        args['entropy_levels'] = convert_to_list(args['entropy_levels'], lambda x : EntropyLevels[x.upper()] if x.isalpha() else float(x))
        if len(args['entropy_levels'])==0:
            raise ValueError("No entropy levels provided")
        
        if set(args['calibration']).intersection({CalibrationMethods.GC, CalibrationMethods.BC}):
            args['prompt_shared'] = True
        
if __name__ == '__main__':
    # vllm stuff
    # mp.set_start_method('fork', force=True)
    setup_single_threading()
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
    parser.add_argument('--prompt_shared', dest='prompt_shared',  action='store_true', required=False, default=False,
                            help='Whether or not to use the same in context examples for the batch of test inputs. \
                                  Not to be used for similarity sampled in context examples.')
    # calibration args
    parser.add_argument('--calibration', dest='calibration', action='store', required=False, 
                            # choices=["GC", "GC_unif", "BC", "TC"],
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
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None, help='batch size for model queries.')
    
    # flags
    parser.add_argument('--use_saved_results', dest='use_saved_results', action='store_true', default=False,
                        help='whether to load the results from pickle files and not run the model')
    parser.add_argument('--approx', dest='approx', action='store_const', const=True, default=False,
                        help='whether to set token prob to zero if not in top 100')
    
    parser.add_argument('--gpu_id', dest='gpu_id', action='store', default=0, required=False, help='Which CUDA gpu to run model on', type=int)
    
    args = parser.parse_args()
    args = vars(args)

    process_args(args)
    args_check(args)
    
    main(**args)