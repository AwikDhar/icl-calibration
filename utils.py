import inspect
import json
import numpy as np
from copy import deepcopy
import os
import torch
import pickle
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, EetqConfig
from typing import Callable, List, Dict
from collections import Counter
import torch
from vllm import LLM, SamplingParams
from calibration.model import CalibrationTransformer
from labels_trie import LabelsTrie

# import transformers
# transformers.logging.set_verbosity_error()

import logging
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(ROOT_DIR, 'saved_results')
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print(f"mkdir at {SAVE_DIR} for saving results")
from data_utils import SampledData 

use_vllm = True
infer_model = None
compile = False
infer_tokenizer = None
calibrator = None
gpu_id = None

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunk_size_helper(params: Dict):
    # Set the batch size (the size of the chunks determines the batch size). Default to 4 for GPT-2 and 20 for OpenAI if
    # no batch size is specified.
    bs = params['bs']
    if bs is None:
        if '/' in params['model']:  # hf model
            return 16 if (params['dataset'] in ('rte', 'cb')) and params['num_shots']>8 else 32
        elif params['model'] in ['ada', 'babbage', 'curie', 'davinci', 'ada-beta', 'babbage-beta', 'curie-beta', 'davinci-beta']:
            return 20
        else:
            return 8
    else:
        return bs
    
def get_test_data(sentences, labels, embeddings, count=None):
        ### sample test set
    if count is None:
        test_sentences, test_labels, test_embeddings = sentences, labels, embeddings
        logger.info(f"selecting full test set ({len(labels)} examples)")
    else:
        if len(sentences)<count:
            logger.error(f"Found only {len(sentences)} available test inputs against a request of {count} "
                            "test inputs, will only use the available ones")
            count = len(sentences)
        # np.random.seed(0)
        test_sentences, test_labels, test_embeddings = random_test_sampling(sentences, labels, embeddings, count)
        logger.info(f"selecting {len(test_labels)} subsample of test set")

    return test_sentences, test_labels, test_embeddings

def random_test_sampling(sentences, labels, embeddings, num):
    """randomly sample subset of the test data"""
    assert len(sentences) == len(labels)
    if num > len(labels):
        assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"

    idxs = np.random.choice(len(labels), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    selected_embeddings = None  # Not none only for similarity sampling 
    if embeddings is not None:
        selected_embeddings = [embeddings[i] for i in idxs]

    return selected_sentences, selected_labels, selected_embeddings

def random_sampling(sentences, labels, num, entropy_level, test_label=None):
    """
    randomly sample subset of the training/test pairs
    num: num_shots/k in k-shot... or the number of test input pairs
    entropy_level: max for balanced label distribution, rand for random sampling, 
                   label spike to have atleast 50% of ICL examples from test label,
                   label suppress to have atleast 50% of ICL examples NOT from the test label
    """
    assert len(sentences) == len(labels)
    assert num <= len(labels), f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"
    
    # 0 shot case
    if num==0:  
        return [], [], []

    label_set = set(labels)
    num_classes = len(label_set)

    if isinstance(entropy_level, float) and num>=2:
        assert num_classes==2, "Entropy specification only supported for binary classification currently!"

        def binary_search(entropy_level):
            entropy_fn = lambda p: -p*np.log2(p) - (1-p)*np.log2(1-p)
            l,r = 0.0, 0.5
            for i in range(100):
                mid = (l+r)/2
                gap = entropy_fn(mid)-entropy_level
                if abs(gap)<0.01:
                    return mid
                if gap>0:
                    r = mid
                else:
                    l = mid
            print("Not enough iters for entropy probab binary search")
            return mid

        p = binary_search(entropy_level)
        neg_label_count = random.sample([round(p*num), round((1-p)*num)],1)[0] 

        idx0 = [i for i,label in enumerate(labels) if label==0]        
        idx1 = [i for i,label in enumerate(labels) if label==1]  
        if neg_label_count > len(idx0) or (num - neg_label_count) > len(idx1):
            raise ValueError(
                f"Not enough samples in pool to satisfy requested entropy "
                f"(need {neg_label_count} zeros and {num-neg_label_count} ones)"
            )
        idxs = random.sample(idx0, neg_label_count) + random.sample(idx1, num-neg_label_count)   
        random.shuffle(idxs)  
    elif isinstance(entropy_level, str) and num>=1:
        match entropy_level:
            case "max":
                per_label_count = num//num_classes
                remainder = num % num_classes

                idx_label_map = {label : [i for i,l in enumerate(labels) if l==label] for label in label_set}
                idxs = []
                for label, indices in idx_label_map.items():
                    if len(indices) < per_label_count:
                        raise ValueError(f"Not enough samples for label {label}")
                    idxs.extend(random.sample(indices, per_label_count))
                
                if remainder > 0:
                    extra_labels = random.sample(list(label_set), remainder)
                    for label in extra_labels:
                        idxs.extend(random.sample(idx_label_map[label], 1))
                random.shuffle(idxs)  
            case "rand" | "randshared":
                idxs = np.random.choice(len(labels), size=num, replace=False)
            case "labelspike":
                test_label_idxs = [i for i, lb in enumerate(labels) if lb==test_label]
                other_label_idxs = [i for i, lb in enumerate(labels) if lb!=test_label]

                # Half of the training examples should have the same label as test input
                idxs = random.sample(test_label_idxs, round(num/2))
                rem_idxs = list(set(test_label_idxs)-set(idxs)) + other_label_idxs
                idxs.extend(random.sample(rem_idxs, num-len(idxs)))
                random.shuffle(idxs)  
            case "labelsuppress":
                test_label_idxs = [i for i, lb in enumerate(labels) if lb==test_label]
                other_label_idxs = [i for i, lb in enumerate(labels) if lb!=test_label]
                
                # Half of the training examples should label DIFFERENT from test input
                idxs = random.sample(other_label_idxs, round(num/2))
                rem_idxs = list(set(other_label_idxs)-set(idxs)) + test_label_idxs
                idxs.extend(random.sample(rem_idxs, num-len(idxs)))
                random.shuffle(idxs)  

    
    if entropy_level is None or num<2:
        idxs = np.random.choice(len(labels), size=num, replace=False)

    assert num==len(idxs)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    assert min(selected_labels)>=0, f"{selected_labels}, {selected_sentences}"
    random_sampling_check(selected_labels, entropy_level, num_classes, test_label)
    # print(f"{len(selected_labels)} labels, {sum(selected_labels)} ones")
    
    return selected_sentences, selected_labels, idxs

def random_sampling_check(selected_labels: List[int], entropy_level, num_classes, test_label:int=None):
    label_counts_map = Counter(selected_labels)

    match entropy_level:
        case "max":
            if len(selected_labels)==1: # no "balanced distribution" for 1 shot, it'll be balanced on avg
                return
            label_counts = label_counts_map.values()
            target = int(len(selected_labels) > num_classes) # only when k-shot is more than num_classes there can be 1 extra ex in some class
            assert max(label_counts) - min(label_counts) <= target, f"Max entropy sampling logic seems to be wrong. label counts: {label_counts}, target:{target}"
        case "labelspike":
            assert label_counts_map[test_label]>=round(len(selected_labels)/2), "Label ain't spiking"
        case "labelsuppress":
            assert label_counts_map[test_label]<=len(selected_labels)-round(len(selected_labels)/2), "Label ain't supppressing"
            
def similarity_sampling(
    sentences:List[str], 
    sentence_embeddings:np.array, 
    labels:List[int], 
    test_embeddings:np.array, 
    num_shots:int,
    test_sentences:List[str]=None,
    shuffle=True,
    return_embeddings=False
    ):
    
    assert len(sentences)==len(sentence_embeddings)==len(labels), "FATAL: Mismatch between train sentences, embeddings and labels lengths"

    similarities = get_similarities(test_embeddings, sentence_embeddings)
    top_idxs = torch.topk(torch.tensor(similarities), k=num_shots, dim=1).indices.numpy() 

    if shuffle:
        for idx_row in top_idxs:
            np.random.shuffle(idx_row)
            
    sampled_data = SampledData()
    sampled_data.sentences = [[sentences[j] for j in idx_row] for idx_row in top_idxs]
    sampled_data.labels = [[labels[j] for j in idx_row] for idx_row in top_idxs]

    # if num_shots==3:
    #     print(train_sentences[1], test_sentences[1])
    #     exit()
    if return_embeddings:
        sampled_data.embeddings = np.array([[sentence_embeddings[j] for j in idx_row] for idx_row in top_idxs])
    return sampled_data

def get_similarities(test_embeddings: np.ndarray, sentence_embeddings: np.ndarray):
    test_embeddings = test_embeddings/np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    sentence_embeddings = sentence_embeddings/np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)

    similarities = np.dot(test_embeddings, sentence_embeddings.T)
    return similarities

def setup_model(model_name, gpu_id_=0):
    global infer_model
    global infer_tokenizer
    global gpu_id
    global use_vllm
    
    if infer_model is None:
        gpu_id = gpu_id_
        device = f'cuda:{gpu_id}'

        cache_dir = os.environ.get('HF_HOME')
        model_type = 'vllm' if use_vllm else 'HF' 
        
        logger.info(f"Setting up {model_type} model: {model_name} with cache dir: {cache_dir}")

        if use_vllm:
            # Configure vLLM model
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            infer_model = LLM(
                model=model_name,
                # tensor_parallel_size=2,
                max_model_len=20000,
                quantization='fp8',
                seed=42,
                # enable_chunked_prefill=False,  # Disable chunked prefill
                max_num_seqs=1,  # Force sequential processing
                # enable_prefix_caching=False,
                # enforce_eager=True,
                limit_mm_per_prompt={"image": 0}, # to skip initialization of vision tower of multimodel models
                max_logprobs=1000,
                download_dir=cache_dir,
                gpu_memory_utilization=0.95
            )
            del os.environ['CUDA_VISIBLE_DEVICES']
            logger.info(f"Loaded {model_name} via vllm")
        else:
            
            attn_implementation="kernels-community/vllm-flash-attn3"
            # attn_implementation="flash_attention_2"
            quantization_config = EetqConfig("int8")

            infer_model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False, trust_remote_code=True, #torch_dtype=torch.bfloat16,
                                                            quantization_config=quantization_config, device_map=device, #tp_plan="auto",
                                                            attn_implementation=attn_implementation, cache_dir=cache_dir)
            if compile:
                torch._dynamo.config.automatic_dynamic_shapes = False
                torch._dynamo.config.assume_static_by_default = True
                torch._dynamo.config.cache_size_limit = 64  # Increase cache
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch._dynamo.config.capture_scalar_outputs = True 
                torch._inductor.config.coordinate_descent_tuning = True
                torch._inductor.config.triton.unique_kernel_names = True
                torch._inductor.config.fx_graph_cache = True  # Enable graph caching
                torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True

                # Compile with max-autotune for best kernel selection
                infer_model = torch.compile(
                    infer_model,
                    mode="max-autotune",  # Searches for best CUDA kernels
                    fullgraph=False,  # True if model supports it
                    dynamic=True
                )
                # infer_model.config.max_position_embeddings = 3048
                # infer_model.config.output_hidden_states = True
                # infer_model = torch.compile(infer_model, mode="max-autotune", fullgraph=False, dynamic=False)
            infer_model.eval()
        
        infer_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device_map=device, cache_dir=cache_dir)
 
        # to batch generation, we pad on the left and mask those positions out.
        infer_tokenizer.padding_side = "left"
        infer_tokenizer.pad_token = infer_tokenizer.eos_token
        if isinstance(infer_tokenizer.eos_token_id, list):
            # Use the first EOS token ID
            logger.info("EOS token ids: " + str(infer_tokenizer.eos_token_id))
            infer_tokenizer.pad_token_id = infer_tokenizer.eos_token_id[-1]
            # infer_model.config.pad_token_id = infer_model.config.eos_token_id[-1]
        else:
            infer_tokenizer.pad_token_id = infer_tokenizer.eos_token_id
            # infer_model.config.pad_token_id = infer_model.config.eos_token_id

        logger.info("Finished loading model")
        
    return infer_model, infer_tokenizer

def setup_calibrator(params):
    global calibrator
    global gpu_id
    device = f'cuda:{gpu_id}'

    with open(f"calibration/models/transformer_config.json", 'r') as file:
        config = json.load(file)

    calibrator = CalibrationTransformer(
        in_features=params['tc_input_dim'], 
        context_length=config['context_length'], 
        embedding_dim=config['embedding_dim'], 
        num_heads=config['num_heads'], 
        num_layers=config['num_layers']
    ).to(device)
    
    model_path = params.get('calibrator_model_path')
    if model_path is None:
        model_path = f"./calibration/models/{params['model'].replace('/','_')}/{params['dataset']}/calibrator"

    state_dict = torch.load(model_path, weights_only=True)
    calibrator.load_state_dict(state_dict)
    
    return calibrator.to(device)

def transformer_calibrate(params:Dict, data: Dict):
    # global gpu_id
    global calibrator
    if calibrator is None:
        setup_calibrator(params)
    device=f'cuda:{gpu_id}'
    
    calibrator.eval()
    with torch.no_grad():
        inputs, logits = data['inputs'].to(device), data['logits'].to(device) # len(eval),T,C | len(eval),T,num_classes | len(eval),T
        
        temperatures = calibrator(inputs) # B,T,1
        calibrated_logits = logits*temperatures # B,T,num_classes
        
        original_argmax = torch.argmax(logits, dim=-1)
        calibrated_argmax = torch.argmax(calibrated_logits, dim=-1)
        if not torch.all(original_argmax == calibrated_argmax):
            mismatch_mask = original_argmax != calibrated_argmax
            print(f"Argmax mismatches at {mismatch_mask.sum()} positions")
            print(f"Original logits at mismatches: {logits[mismatch_mask]}")
            print(f"Calibrated logits at mismatches: {calibrated_logits[mismatch_mask]}")
            print(f"Temperatures at mismatches: {temperatures[mismatch_mask]}")
            raise AssertionError("Argmax changed after calibration")        
        
        return calibrated_logits[:,-1,:].cpu()        

def complete_generation_opt(prompts, num_log_probs=None):
    ''' This function runs GPT-2 locally but places the outputs into an json that looks just like the one
     provided by the OpenAI API. '''
    global gpu_id
    
    torch.manual_seed(42)
    with torch.no_grad():
        if isinstance(prompts, str):
            prompts = [prompts] # the code below assumes a list
        # print(prompts[-1], prompts[-1][-1])
        # exit()
                # Store original order and sort by length
        original_indices = list(range(len(prompts)))
        indexed_prompts = list(zip(original_indices, prompts))
        indexed_prompts.sort(key=lambda x: len(x[1]))  # Sort by prompt length
        sorted_indices, sorted_prompts = zip(*indexed_prompts) if indexed_prompts else ([], [])
        sorted_prompts = list(sorted_prompts)
        
        # print(prompts[0])
        # exit()
        if infer_tokenizer.chat_template is not None:
            messages = [[{"role": "user", "content":prompt}] for prompt in sorted_prompts]
            kwargs = {}
            if 'enable_thinking' in inspect.signature(infer_tokenizer.apply_chat_template).parameters:
                kwargs = {'enable_thinking':False}
            sorted_prompts = infer_tokenizer.apply_chat_template(messages, continue_final_message=True, tokenize=False, **kwargs)
        
        input_ids = infer_tokenizer(sorted_prompts, padding=True, return_tensors='pt').to(next(infer_model.parameters()).device)
        
        # we are left padding, so we need to adjust the position IDs
        attention_mask = input_ids['attention_mask']
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        # get the logits for the input context
        with torch.inference_mode():
            outputs = infer_model.forward(
                input_ids=input_ids['input_ids'], 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                output_hidden_states=False, 
                return_dict=True
            )
        
        # Reorder outputs back to original order
        reorder_indices = [0] * len(sorted_indices)
        for new_idx, orig_idx in enumerate(sorted_indices):
            reorder_indices[orig_idx] = new_idx
        reorder_tensor = torch.tensor(reorder_indices, device=outputs.logits.device)
        
        # Generic reordering of all tensor values in outputs
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                outputs[key] = value[reorder_tensor]
            elif isinstance(value, tuple) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                outputs[key] = tuple(v[reorder_tensor] for v in value)
                
        # Also reorder input_ids dict
        input_ids['input_ids'] = input_ids['input_ids'][reorder_tensor]
        input_ids['attention_mask'] = input_ids['attention_mask'][reorder_tensor]
        
        logits = outputs.logits.detach().cpu().float()
        # hidden_states = outputs.hidden_states[-1].detach().cpu().float()
        # get the logits for the last position (where next token will be predicted)
        next_token_logits = logits[:,-1:] # last position logits
        # next_token_hidden = hidden_states[:,-1:]
        
        # get next token greedily (argmax since do_sample=False)
        next_tokens = torch.argmax(next_token_logits, dim=-1)  # [batch_size, 1]
        
        # create full sequences with the new token
        total_sequences = torch.cat([input_ids['input_ids'], next_tokens.to(input_ids['input_ids'].device)], dim=1)
        
        # compute probabilities if needed
        if num_log_probs is not None:
            probs = torch.softmax(next_token_logits, dim=2).cpu()
            top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
            top_logits, top_logit_tokens = torch.topk(next_token_logits, k=num_log_probs)
            logprobs = torch.log(probs)
            top_log_probs = torch.log(top_probs)
        
        torch.cuda.empty_cache()
        
        # create the return value to resemble OpenAI
        return_json = {}
        choices = []
        for batch_id in range(len(prompts)):
            curr_json = {}
            # text is just the generated 1 token
            curr_json['text'] = infer_tokenizer.decode(total_sequences[batch_id][-1:], skip_special_tokens=True)

            # fill the return json with the top tokens and probs to match the OpenAI return value.
            if num_log_probs is not None:
                curr_json['logprobs'] = {}
                curr_json['logprobs']['top_logprobs'] = []
                curr_json['logprobs']['token_logprobs'] = []
                curr_json['logprobs']['tokens'] = []
                # curr_json['logprobs']['hidden_states'] = []
                curr_json['logprobs']['token_logits'] = []
                
                # process the single generated token's logprobs
                current_element_top_log_probs = top_log_probs[batch_id][0]  # single position
                current_element_top_tokens = top_tokens[batch_id][0]
                token_logits = top_logits[batch_id][0]
                # hidden = next_token_hidden[batch_id][0]
                
                # tokens is a list of the top token at each position
                curr_json['logprobs']['tokens'].append(infer_tokenizer.decode([current_element_top_tokens[0]]))
                # token_logprobs is a list of the logprob of the top token at each position  
                curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                temp, temp_logits = {}, {}
                for log_prob, token, logit in zip(current_element_top_log_probs, current_element_top_tokens, token_logits):
                    string_ = infer_tokenizer.decode(token.item())
                    if string_ not in temp.keys():
                        temp[string_] = log_prob.item()
                        temp_logits[string_] = logit.item()
                curr_json['logprobs']['top_logprobs'].append(temp)
                curr_json['logprobs']['token_logits'].append(temp_logits)
                # curr_json['logprobs']['hidden_states'].append(hidden)

            choices.append(curr_json)
        return_json['choices'] = choices
        torch.cuda.empty_cache()
        # print(prompts[-1], curr_json['logprobs']['token_logits'])
        # exit()
        return return_json
    
def complete_generation_vllm(prompts, num_log_probs=None):
    ''' This function runs inference using vLLM but places the outputs into a json that looks just like the one
     provided by the OpenAI API. '''
    
    # for deterministic outputs
    torch.manual_seed(42)
    
    if isinstance(prompts, str):
        prompts = [prompts]  # the code below assumes a list
    # print(prompts[0]), exit()
    # if infer_tokenizer.chat_template is not None:
    #     messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    #     kwargs = {}
    #     if 'enable_thinking' in inspect.signature(infer_tokenizer.apply_chat_template).parameters:
    #         kwargs = {'enable_thinking': False}
    #     # prompts = infer_tokenizer.apply_chat_template(messages, continue_final_message=True, tokenize=False, **kwargs)
    #     prompts = infer_tokenizer.apply_chat_template(messages, continue_final_message=False, add_generation_prompt=True, tokenize=False, **kwargs)
    
    # Configure sampling parameters for single token generation
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy sampling (equivalent to argmax)
        max_tokens=1,     # Generate exactly 1 token
        logprobs=num_log_probs if num_log_probs is not None else None,
        prompt_logprobs=None,  # We don't need prompt logprobs
        skip_special_tokens=True,
        seed=42,
        
    )
    
    # Generate using vLLM
    outputs = infer_model.generate(prompts, sampling_params, use_tqdm=False)
    
    # Process outputs to match OpenAI format
    return_json = {}
    choices = []
    
    for output in outputs:
        curr_json = {}
        
        # Get the generated token
        generated_token = output.outputs[0]
        curr_json['text'] = generated_token.text
        
        # Handle logprobs if requested
        if num_log_probs is not None:
            curr_json['logprobs'] = {}
            curr_json['logprobs']['top_logprobs'] = []
            curr_json['logprobs']['token_logprobs'] = []
            curr_json['logprobs']['tokens'] = []
            curr_json['logprobs']['hidden_states'] = []  # Note: vLLM doesn't expose hidden states by default
            curr_json['logprobs']['token_logits'] = []
            
            if generated_token.logprobs:
                # Get logprobs for the generated token
                token_logprobs = generated_token.logprobs[0]  # First (and only) generated token
                # print(token_logprobs); exit()
                # Extract top token and its logprob
                top_tokens = list(token_logprobs.keys())
                if top_tokens:
                    top_token = max(token_logprobs.keys(), key=lambda x: token_logprobs[x].logprob)
                    curr_json['logprobs']['tokens'].append(infer_tokenizer.decode([top_token]))
                    curr_json['logprobs']['token_logprobs'].append(token_logprobs[top_token].logprob)
                
                # Create top_logprobs dict
                temp = {}
                temp_logits = {}  # vLLM doesn't directly provide logits, so we'll compute from logprobs
                
                # collect logits into a single tensor
                logits = torch.tensor([logprob_data.logprob for logprob_data in token_logprobs.values()])

                # stable softmax denominator
                log_sum_exp = torch.logsumexp(logits, dim=0)
                for token_id, logprob_data in token_logprobs.items():
                    token_str = infer_tokenizer.decode([token_id])
                    if token_str not in temp:
                        temp[token_str] = logprob_data.logprob - log_sum_exp
                        # vLLM doesn't expose raw logits but we modified their sampler.py to return logits in the logprob property
                        temp_logits[token_str] = logprob_data.logprob  
                
                curr_json['logprobs']['top_logprobs'].append(temp)
                curr_json['logprobs']['token_logits'].append(temp_logits)
                
                # Hidden states: vLLM doesn't expose these by default
                # You would need to modify vLLM or use a custom worker to get hidden states
                curr_json['logprobs']['hidden_states'].append(None)  # Placeholder
        
        choices.append(curr_json)
    
    # print(curr_json['logprobs']['top_logprobs'], prompts[-1]); exit()
    return_json['choices'] = choices
    return return_json

def complete(prompt, temp=0, num_log_probs=None):
    """complete the prompt using a language model"""
    global use_vllm

    assert temp >= 0
  
    if use_vllm:
        return complete_generation_vllm(prompt, num_log_probs=num_log_probs)
    else:
        return complete_generation_opt(prompt, num_log_probs=num_log_probs)

def construct_prompt(params, train_sentences, train_labels, test_sentence):
    global infer_tokenizer
    
    if infer_tokenizer.chat_template is not None:
        return construct_chat_prompt(params, train_sentences, train_labels, test_sentence)    
    else:
        return construct_base_prompt(params, train_sentences, train_labels, test_sentence)
    
def construct_base_prompt(params, train_sentences, train_labels, test_sentence):
    """construct a single prompt to be fed into the model"""
    # take the prompt template and fill in the training and test example
        
    prompt = params["prompt_prefix"]+'\n\n'
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]

    for sentence, label_idx in zip(train_sentences, train_labels):
        prompt += q_prefix + sentence + "\n"
        prompt += a_prefix + params['label_dict'][label_idx]['label'] + "\n\n"
    
    prompt += q_prefix + test_sentence + "\n"
    
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    # if(len(train_sentences))==0:
    # print(prompt)
    # exit()
    return prompt

def construct_chat_prompt(params, train_sentences, train_labels, test_sentence):    
    # messages = [{"role": "user", "content": params["prompt_prefix"]+'\n\n'}]
    
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]

    messages = []
    for i, (sentence, label_idx) in enumerate(zip(train_sentences, train_labels)):    
        messages.append({"role": "user", "content":q_prefix + sentence + "\n"})
        messages.append({"role": "assistant", "content":a_prefix + params['label_dict'][label_idx]['label'] + "\n\n"})
    
    messages.append({"role": "user", "content":q_prefix + test_sentence + "\n"})
    assert a_prefix[-1] == ' '
    messages.append({"role": "assistant", "content":a_prefix[:-1]})

    messages[0]['content'] = params["prompt_prefix"]+'\n\n' + messages[0]['content'] # First user message has the base prompt

    kwargs = {}
    if 'enable_thinking' in inspect.signature(infer_tokenizer.apply_chat_template).parameters:
        kwargs = {'enable_thinking': False}
    prompt = infer_tokenizer.apply_chat_template(messages, continue_final_message=True, tokenize=False, **kwargs)
    # prompt = infer_tokenizer.apply_chat_template(messages, continue_final_message=False, add_generation_prompt=True, tokenize=False, **kwargs)
    
    return prompt
    

def populate_trie_recursive(prompt_prefix, path, label_trie, params):
    """Recursively generate and populate trie with logits for all branches"""
    node = label_trie.root
    for token in path:
        node = node.children[token]
    
    # If no children, we've reached a leaf
    if not node.children:
        return
    
    if len(node.children)==1:
        # 10 as a dummy logit, value doesn't matter since prob will be 1 after softmax regardless
        child_token = list(node.children.keys())[0]
        token_logits = {child_token : 10} 
    else:
        # Generate next token to get logits
        resp = complete([prompt_prefix], num_log_probs=params['api_num_log_prob'])
        token_logits = resp['choices'][0]['logprobs']['token_logits'][0]
        # if len(path)==0: 
        #     print(token_logits)
    
    # Update trie with logits at current path
    label_trie.update_logits(path, token_logits)
    
    # Recursively explore each child branch
    for token in node.children:
        new_prompt = prompt_prefix + token
        new_path = path + [token]
        populate_trie_recursive(new_prompt, new_path, label_trie, params)
        
def get_results(params, train_sentences, train_labels, test_sentences):
    """Get results for multi-token labels"""
    setup_model(params['model'], gpu_id_=params['gpu_id'])

    all_label_probs = []
    all_label_raw_logits = []    # will be filled with logprobs 
    
    for i, test_sentence in enumerate(test_sentences):
        prompt = construct_prompt(params, train_sentences[i], train_labels[i], test_sentence)
        # if len(train_labels[i])==8:
        #     print(prompt); exit()
        # print(prompt)
        # Create fresh trie for this test instance
        label_trie = LabelsTrie(params['label_dict'])

        # Recursively populate entire trie
        populate_trie_recursive(prompt, [], label_trie, params)
        
        # Get all label probabilities
        label_probs = label_trie.get_all_label_probs()

        # with np.printoptions(precision=3, suppress=True):
        # first_logits = [token.logit for token in list(label_trie.root.children.values())]
        # label_trie.print_trie(); print(first_logits, np.ma.masked_invalid(first_logits).mean()); print(label_probs); exit()
        # print([label_trie.root.children[child].__dict__ for child in label_trie.root.children])
        # assert len(zero_toks) == sum(label_probs==0), f"{len(zero_toks) , sum(label_probs==0)} {zero_toks}, {label_probs==0}"; exit()
        all_label_probs.append(label_probs)
    
    eps = 1e-7 # for avoiding log(0)
    for label_probs in all_label_probs:
        if 0 in label_probs:
            label_probs += eps
            
        all_label_raw_logits.append(np.log(label_probs)) # logprobs
        
    # print(np.array(all_label_probs)); exit()
    return np.array(all_label_probs), np.array(all_label_raw_logits)

def load_pickle(params):
    # load saved results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")
    return data

def print_results(tree, names=('Original Accuracy  ','Calibrated Accuracy'), log:Callable = print):
    # print out all results
    root = deepcopy(tree)

    for dataset in root.keys():
        log(f"\n\nDataset: {dataset}")
        models_node = root[dataset]
        for model in models_node.keys():
            log(f"\nModel: {model}")
            entropy_node = models_node[model]
            for entropy_level in entropy_node.keys():
                log(f"\nEntropy level: {entropy_level}")
                num_shots_node = entropy_node[entropy_level]
                for num_shots in num_shots_node.keys():
                    accuracies = np.array(list(num_shots_node[num_shots].values()))
                    if len(accuracies) == 0:
                        continue
                    accuracies_mean = np.mean(accuracies, axis=0)
                    accuracies_low = np.min(accuracies, axis=0)
                    accuracies_high = np.max(accuracies, axis=0)
                    accuracies_std = np.std(accuracies, axis=0)

                    if isinstance(num_shots, str)  and 'ece' in num_shots:
                        names = ('Original ECE', 'Calibrated ECE')
                    elif isinstance(num_shots, str)  and 'diff' in num_shots:
                        names = ('Conf diff', )
                    elif isinstance(num_shots, str)  and 'norm' in num_shots:
                        names = ('Feature norm', 'Calibrated norm')
                    elif isinstance(num_shots, str)  and 'temp' in num_shots:
                        names = ('Best tempture', 'Specific')
                    elif isinstance(num_shots, str)  and 'entropy' in num_shots:
                        names = ('Entropy', 'Calibrated entropy')
                    elif isinstance(num_shots, str)  and 'conf' in num_shots:
                        names = ('Confidence', 'Calibrated confidence')
                    else:
                        names = ('Original Accuracy','Calibrated Accuracy')
                        log(f"\n{num_shots}-shot, {entropy_level} ICL entropy, {len(accuracies)} seeds")
                    
                    # for aligned | char
                    max_len = max(len(name) for name in names)
                    names = [name + ' '*(max_len-len(name)) for name in names]

                    for i, (m, l, h, s) in enumerate(zip(accuracies_mean, accuracies_low, accuracies_high, accuracies_std)):
                        log(f"{names[i]} | Mean: {m:.4f}, Low: {l:.4f}, High: {h:.4f}, Std: {s:.4f}")
                    print()

def load_results(params_list):
    # load saved results from model
    result_tree = dict()
    for params in params_list:
        saved_result = load_pickle(params)
        keys = [params['dataset'], params['model'], params['entropy_levels'], params['num_shots']]
        node = result_tree # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        node[params['seed']] = saved_result['accuracies']
    print_results(result_tree)
    

def convert_to_list(items, cvt_func=None):
    if cvt_func:
        return [cvt_func(s.strip()) for s in items.split(",")]
    else:
        return [s.strip() for s in items.split(",")]
    
# Force single-threaded execution to avoid conflicts
def setup_single_threading():
    """Setup single-threading to avoid vLLM conflicts"""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    # Set torch to use single thread
    # torch.set_num_threads(1)
    
def setup_vllm_env_settings():    
    # os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
    
    # for deterministic behaviour
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.set_num_threads(1)
    # torch.use_deterministic_algorithms(True)  
    # torch.backends.cudnn.deterministic = True  
    # torch.backends.cudnn.benchmark = False     
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)