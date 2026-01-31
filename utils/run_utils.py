from vllm import LLM, SamplingParams, config, PoolingParams
from vllm.config.compilation import CompilationConfig, PassConfig, CUDAGraphMode

import inspect
import json
from pathlib import Path
import numpy as np
from copy import deepcopy
import os
import torch
import pickle
import random
from typing import Callable, List, Dict
from collections import Counter

from transformers import AutoTokenizer, AutoModelForCausalLM, EetqConfig
import torch
from sentence_transformers import SentenceTransformer
from gemini.gemini_model import GeminiModel

from calibration.model import CalibrationTransformer, PositionEmbeddingType
from calibration.temperature import get_equivalent_temp, get_equivalent_temp_gd, get_shotwise_dynamic_temperatures

from labels_trie import LabelsTrie
from llm_framework import LlmFramework
from utils.gen_utils import get_llm_framework

import logging
logger = logging.getLogger(__name__)

logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)

llm_framework = LlmFramework.VLLM
infer_model = None
compile = False
infer_tokenizer = None
calibrator = None
gpu_ids = None
embedding_model = None

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunk_size_helper(params: Dict):
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

def get_model_snapshot_path(model_name, cache_dir=None):
    global llm_framework
    
    if llm_framework is LlmFramework.GOOGLE:
        return model_name
    
    if cache_dir is None:
        cache_dir = os.environ.get('HF_HOME') or os.environ.get('HF_HUB_CACHE')
    
    cache_name = "models--" + model_name.replace("/", "--")
    snapshots_dir = Path(cache_dir) / cache_name / "snapshots"
    
    if not snapshots_dir.exists():
        logger.error(f"No snapshots found: {snapshots_dir}")
        download_model = input("Download unavailable model from HF? (y/n): ")
        if download_model.lower().strip()=='y':
            return model_name
        else:
            exit()
            
    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot directories found in {snapshots_dir}")
    
    latest_snapshot = max(snapshot_dirs, key=lambda p: p.stat().st_mtime)
    
    return str(latest_snapshot)
          
def setup_embedding_model(params: Dict):
    global embedding_model
    global gpu_ids
    
    # if embedding_model is None:
    #     model_name = params.get('embedding_model', 'google/embeddinggemma-300m')
    #     truncate_dim = params.get('embedding_dim', 128)
    #     embedding_model = SentenceTransformer(f"{model_name}", device=f'cuda:{gpu_ids[0]}', truncate_dim=truncate_dim, cache_folder=os.environ['HF_HOME'], model_kwargs={'dtype':torch.float16})
    if embedding_model is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(gpu_id) for gpu_id in gpu_ids])
        model_name = params.get('embedding_model', 'google/embeddinggemma-300m')
        # model_name = params.get('embedding_model', 'Qwen/Qwen3-Embedding-4B')
        HF_HOME = os.environ.get('HF_HOME')
        cache_dir = os.environ.get('HF_HUB_CACHE', HF_HOME)

        model_snapshot_or_card = get_model_snapshot_path(model_name, cache_dir)
        embedding_model = LLM(
            model=model_snapshot_or_card,
            runner="pooling",
            # attention_config=config.AttentionConfig(backend="TRITON_ATTN"),
            # dtype='float16',
            tensor_parallel_size=len(gpu_ids),
            download_dir=cache_dir,
            seed=42,
            gpu_memory_utilization=0.2,
            hf_overrides={"is_matryoshka": True},
            trust_remote_code=True,
            # enforce_eager=True
        )
        
        del os.environ['CUDA_VISIBLE_DEVICES']
        
# Get embeddings of sentences at a specified truncated dim. 
# Full embeddings for similarity sampling, truncated embeddings for calibrator features for easier learning 
def get_embeddings(params: Dict, sentences: List[str]):
    global embedding_model
    
    setup_embedding_model(params)
    # with torch.inference_mode():
    #     sentences_embeddings = embedding_model.encode(sentences, prompt_name="Classification", show_progress_bar=False, convert_to_numpy=True) 
    
    truncate_dim = params.get('embedding_dim', 128)
    pooling_params = PoolingParams(dimensions=truncate_dim)
    
    outputs = embedding_model.embed(sentences, pooling_params=pooling_params, use_tqdm=False)
    sentences_embeddings = np.array([output.outputs.embedding for output in outputs])
    
    return sentences_embeddings

def setup_llm(model_name, num_log_probs = 1000, gpu_ids_=[0]):
    global infer_model
    global gpu_ids
    global llm_framework
    
    if infer_model is None:

        llm_framework = get_llm_framework(model_name)

        gpu_ids = gpu_ids_
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(gpu_id) for gpu_id in gpu_ids])

        HF_HOME = os.environ.get('HF_HOME')
        cache_dir = os.environ.get('HF_HUB_CACHE', HF_HOME)

        model_snapshot_or_card = get_model_snapshot_path(model_name, cache_dir)
        
        logger.info(f"Setting up {llm_framework.name} model: {model_name} with cache dir: {cache_dir}")

        match llm_framework:
            case LlmFramework.GOOGLE:
                infer_model = GeminiModel(model_name=model_name)
            
            case LlmFramework.VLLM:
                infer_model = LLM(
                    model=model_snapshot_or_card,
                    tensor_parallel_size=len(gpu_ids),
                    max_model_len=40000,
                    # quantization='fp8',
                    # attention_config=config.AttentionConfig(backend="TRITON_ATTN"),
                    # attention_config=config.AttentionConfig(backend="FLASH_ATTN"),
                    attention_config=config.AttentionConfig(backend="FLASHINFER"),
                    kv_cache_dtype="auto",
                    seed=42,
                    # dtype='float16', # for deterministic results. bf16 has slight precision issues, which make matrix operations non-deterministic
                    # enable_chunked_prefill=False, 
                    max_num_batched_tokens=40000, 
                    max_num_seqs=8,  # Force sequential processing
                    enable_prefix_caching=True,
                    # enforce_eager=True,
                    limit_mm_per_prompt={"image": 0}, # to skip initialization of vision tower of multimodel models
                    max_logprobs=num_log_probs,
                    download_dir=cache_dir,
                    gpu_memory_utilization=0.75,
                    # compilation_config={"compile_sizes": [1]},
                    # kv_transfer_config={"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"},
                    compilation_config=CompilationConfig(
                        # pass_config=PassConfig(
                        #     fuse_allreduce_rms=True,
                        #     eliminate_noops=False
                        # ),
                        compile_sizes=[1],
                        cudagraph_mode="FULL",  # or "PIECEWISE" if you hit IMA errors
                        cudagraph_capture_sizes=[1],
                    ),
                    trust_remote_code=True
                )
                
                logger.info(f"Loaded {model_name} via vllm")
            
            case LlmFramework.HF:
                
                # attn_implementation="flash_attention_2"
                attn_implementation="kernels-community/vllm-flash-attn3"
                quantization_config = None#EetqConfig("int8")

                infer_model = AutoModelForCausalLM.from_pretrained(model_snapshot_or_card, use_cache=False, trust_remote_code=True, dtype=torch.bfloat16,
                                                                quantization_config=quantization_config, device_map=f"cuda:0",#tp_plan="auto",
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
                        mode="max-autotune", 
                        fullgraph=False,  
                        dynamic=True
                    )
                    
                infer_model.eval()
                logger.info(f"Loaded {model_name} via HF")
                
        del os.environ['CUDA_VISIBLE_DEVICES']
        
        if llm_framework in (LlmFramework.VLLM, LlmFramework.HF):
            setup_tokenizer(model_snapshot_or_card, cache_dir)

def setup_tokenizer(model_name, cache_dir):
    global infer_tokenizer
    
    infer_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
 
    # to batch generation, we pad on the left and mask those positions out.
    infer_tokenizer.padding_side = "left"
    infer_tokenizer.pad_token = infer_tokenizer.eos_token
    if isinstance(infer_tokenizer.eos_token_id, list):
        # Use the first EOS token ID
        logger.info("EOS token ids: " + str(infer_tokenizer.eos_token_id))
        infer_tokenizer.pad_token_id = infer_tokenizer.eos_token_id[-1]
    else:
        infer_tokenizer.pad_token_id = infer_tokenizer.eos_token_id
        
def setup_calibrator(params):
    global calibrator
    device = f'cuda:{gpu_ids[0]}'

    with open(f"calibration/models/transformer_config.json", 'r') as file:
        config = json.load(file)

    calibrator = CalibrationTransformer(
        in_features=params['tc_input_dim'], 
        context_length=config['context_length'], 
        embedding_dim=config['embedding_dim'], 
        num_heads=config['num_heads'], 
        num_layers=config['num_layers'],
        pos_embedding_type=PositionEmbeddingType.SINUSOIDAL
    ).to(device)
    
    model_path = params.get('calibrator_model_path')
    # if model_path is None:
    #     if params['llm_agnostic']:
    #         model_dir = f"./calibration/models/llm_agnostic"
    #     else:
    #         model_dir = f"./calibration/models/{params['model'].replace('/','_')}"
            
    #     # if sampling_strategy:
    #     #     model_dir += f"/{sampling_strategy}"
            
    #     model_path = f'{model_dir}/{params['calibrator_name']}'
        
    if model_path is None:
        model_path = f"./calibration/models/llm_agnostic/{params['calibrator_name']}"

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    calibrator.load_state_dict(state_dict)
    
    return calibrator.to(device)

def transformer_calibrate(params:Dict, inputs: torch.Tensor, logits: torch.Tensor):
    global calibrator
    if calibrator is None:
        setup_calibrator(params)
    device=f'cuda:{gpu_ids[0]}'
    
    calibrator.eval()
    with torch.no_grad():
        inputs = inputs.to(device) # len(eval),T,C | len(eval),T,num_classes | len(eval),T
        logits = logits.to(device)
        
        B, num_classes = logits.shape
        
        calibrated_pred_probs = calibrator(inputs) # B,T,1
        calibrated_pred_probs = torch.clamp(calibrated_pred_probs, min=1/num_classes + 2e-2, max = 1.0 - 2e-2)
    
    logit_std = logits.std(dim=-1, keepdim=True)  # B, 1
    is_uniform = logit_std < 1e-3
    perfectly_uniform = logit_std == 0
    
    if is_uniform.any():
        logger.warning(f"Found {is_uniform.sum().item()}/{B} instances with near uniform probabilities, {perfectly_uniform.sum()} uniform. Consider increasing num_logprobs/better prompting")
        argmax_class = logits.argmax(dim=-1)  # B
        boost = torch.zeros_like(logits)
        boost.scatter_(1, argmax_class.unsqueeze(1), 0.01)  # Add an eps to argmax to be abel to scale
        
        # Only boost uniform samples
        logits = torch.where(is_uniform, logits + boost, logits)
    
    temperatures = get_equivalent_temp(logits.view(B, 1, num_classes), calibrated_pred_probs[:, [-1], :])  # (B, 1, num_classes) (B, 1, 1)
    temperatures = torch.nan_to_num(temperatures, nan=1.0)
    calibrated_logits = logits*temperatures[:, -1, :] # (B, num_classes) * (B,1)
    
    # Deviation analysis
    # target_probs = calibrated_pred_probs[:, -1, :].squeeze()
    # calibrated_probs = calibrated_logits.softmax(dim=-1).max(dim=-1).values
    # original_probs = logits.softmax(dim=-1).max(dim=-1).values
    
    # abs_err = (calibrated_probs - target_probs).abs()
    # rel_err = (abs_err / (target_probs + 1e-8)) * 100

    # print(f"Mean absolute deviation: {abs_err.mean():.4f} | Mean relative deviation: {rel_err.mean():.2f}%")
    # print(f"% beyond 1% abs: {(abs_err > 0.01).float().mean()*100:.2f}% | beyond 5% rel: {(rel_err > 5).float().mean()*100:.2f}%")
    # print(f"% beyond 2% abs: {(abs_err > 0.02).float().mean()*100:.2f}% | beyond 10% rel: {(rel_err > 10).float().mean()*100:.2f}%\n")
    # print("Temperatures: ", temperatures.mean().item(), temperatures.min().item(), temperatures.max().item())
    # print("Target probs: ", target_probs.mean().item(), target_probs.min().item(), target_probs.max().item(), target_probs.std().item())
    # print("Calibrated probs: ", calibrated_probs.mean().item(), calibrated_probs.min().item(), calibrated_probs.max().item(), calibrated_probs.std().item())

    # worst_idx = (calibrated_probs - target_probs).abs().argmax()
    # worst_temp = temperatures[worst_idx, -1, :].item()
    # worst_original_prob = original_probs[worst_idx].item()
    # print(f"Worst sample: original={worst_original_prob:.4f}, target={target_probs[worst_idx]:.4f}, achieved={calibrated_probs[worst_idx]:.4f}, T={worst_temp:.2f}\n")
    # print(temperatures[:, -1, :].shape, logits.shape)
    
    logger.info("Transformer calibrated logits")  
    return calibrated_logits

def transformer_calibrate_tmp(params:Dict, inputs: torch.Tensor, logits: torch.Tensor):
    # global gpu_id
    global calibrator
    if calibrator is None:
        setup_calibrator(params)
    device=f'cuda:{gpu_id}'
    
    calibrator.eval()
    with torch.no_grad():
        inputs = inputs.to(device) # len(eval),T,C | len(eval),T,num_classes | len(eval),T
        logits = logits.to(device)
        
        B, T, num_classes = logits.shape
        
        calibrated_pred_probs = calibrator(inputs) # B,T,1
        calibrated_pred_probs = torch.clamp(calibrated_pred_probs, min=1/num_classes + 2e-2, max = 1.0 - 2e-2)
    
        temperatures = get_equivalent_temp(logits, calibrated_pred_probs)  # (B, 1, num_classes) (B, 1, 1)
        temperatures = torch.nan_to_num(temperatures, nan=1.0)
        calibrated_logits = logits*temperatures # (B, num_classes) * (B,1)
        
        print(temperatures[:, -1, :].mean().item(), temperatures[:, -1, :].min().item(), temperatures[:, -1, :].max().item())
        print(calibrated_pred_probs[:, -1, :].mean().item(), calibrated_pred_probs[:, -1, :].min().item(), calibrated_pred_probs[:, -1, :].max().item(), calibrated_pred_probs[:, -1, :].std().item())
        
        calibrated_probs, calibrated_preds = torch.nn.functional.softmax(calibrated_logits, dim=-1).max(dim=-1)
        print(calibrated_probs[:, -1].mean().item(), calibrated_probs[:, -1].min().item(), calibrated_probs[:, -1].max().item(), calibrated_probs[:, -1].std().item())
        
        # print(temperatures.mean().item(), temperatures.min().item(), temperatures.max().item())
        # print(calibrated_pred_probs.mean().item(), calibrated_pred_probs.min().item(), calibrated_pred_probs.max().item(), calibrated_pred_probs.std().item())
        # calibrated_probs = calibrated_logits.softmax(dim=-1).max(dim=-1).values
        # print(calibrated_probs.mean().item(), calibrated_probs.min().item(), calibrated_probs.max().item(), calibrated_probs.std().item())
        # print(temperatures[:, -1, :].shape, logits.shape)
        
    return calibrated_logits[:, -1, :]

def complete_generation_hf(prompts, num_log_probs=None):
    ''' This function runs GPT-2 locally but places the outputs into an json that looks just like the one
     provided by the OpenAI API. '''
    global gpu_id
    
    # torch.manual_seed(42)
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
    
def complete_generation_vllm(prompts, num_log_probs=None, seed=None):
    ''' This function runs inference using vLLM but places the outputs into a json that looks just like the one
     provided by the OpenAI API. '''

    assert seed is not None, "Please set a seed for determinism"
    
    if isinstance(prompts, str):
        prompts = [prompts]  # the code below assumes a list
    # print(prompts[0]), exit()

    # Configure sampling parameters for single token generation
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy sampling (equivalent to argmax)
        max_tokens=1,     # Generate exactly 1 token
        logprobs=num_log_probs if num_log_probs is not None else None,
        prompt_logprobs=None,  # We don't need prompt logprobs
        skip_special_tokens=True,
        seed=seed,
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
                token_logprobs = generated_token.logprobs[0]  # Logprobs of first (and only) generated token
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

def complete_generation_google(prompts, num_log_probs=None):    
    if isinstance(prompts, str):
        prompts = [prompts]
    
    return_json = {}
    choices = []
    
    for prompt in prompts:
        curr_json = {}
        
        response = infer_model.generate(prompt=prompt, num_log_probs=20)
        
        curr_json['text'] = response.text
        
        # Handle logprobs if requested
        if num_log_probs is not None:
            curr_json['logprobs'] = {}
            curr_json['logprobs']['top_logprobs'] = []
            curr_json['logprobs']['token_logprobs'] = []
            curr_json['logprobs']['tokens'] = []
            curr_json['logprobs']['hidden_states'] = []
            curr_json['logprobs']['token_logits'] = []
            
            # Access logprobs from response
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Check for logprobs_result
                if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
                    logprobs_result = candidate.logprobs_result
                    
                    # Process top candidates for the generated token
                    if hasattr(logprobs_result, 'top_candidates') and logprobs_result.top_candidates:
                        # Get the first (and only) token's logprobs
                        token_candidates = logprobs_result.top_candidates[0].candidates
                        
                        # Find the top token (highest logprob)
                        if token_candidates:
                            top_candidate = max(token_candidates, key=lambda x: x.log_probability)
                            
                            # Store top token
                            curr_json['logprobs']['tokens'].append(top_candidate.token)
                            curr_json['logprobs']['token_logprobs'].append(top_candidate.log_probability)
                            
                            # Create dictionaries for all top tokens
                            temp = {}
                            temp_logits = {}
                            
                            # Collect all logprobs for normalization
                            all_logprobs = [token_candidate.log_probability for token_candidate in token_candidates]
                            
                            # Calculate log_sum_exp for stable softmax
                            max_logprob = max(all_logprobs)
                            log_sum_exp = max_logprob + np.log(
                                sum(np.exp(lp - max_logprob) for lp in all_logprobs)
                            )
                            
                            # Populate dictionaries
                            for candidate in token_candidates:
                                token_str = candidate.token
                                if token_str not in temp:
                                    # Normalized log probability
                                    temp[token_str] = candidate.log_probability - log_sum_exp
                                    # Raw log probability as "logit"
                                    temp_logits[token_str] = candidate.log_probability
                            
                            curr_json['logprobs']['top_logprobs'].append(temp)
                            curr_json['logprobs']['token_logits'].append(temp_logits)
                            curr_json['logprobs']['hidden_states'].append(None)    
                        else:
                            # No candidates, add empty entries
                            print("No cand")
                            print(logprobs_result.top_candidates)
                    else:
                        # No top_candidates, add empty entries
                        print("No top cand")
                        print(response)
                else:
                    # No logprobs_result, add empty entries
                    print("No logprob")
                    print(response, response.candidates[0].content)
            else:
                # No candidates, add empty entries
                print("No candd")
                print(response)

        choices.append(curr_json)
    return_json['choices'] = choices
    # print(return_json); exit()
    return return_json

def complete_generation(prompt, num_log_probs=None, seed=None):
    """complete the prompt using a language model"""
    global llm_framework
     
    match llm_framework:
        case LlmFramework.GOOGLE:
            return complete_generation_google(prompt, num_log_probs=num_log_probs)
        case LlmFramework.VLLM:
            return complete_generation_vllm(prompt, num_log_probs=num_log_probs, seed=seed)
        case LlmFramework.HF:
            return complete_generation_hf(prompt, num_log_probs=num_log_probs)

def construct_prompt(params, train_sentences, train_labels, test_sentence):
    global infer_tokenizer
    
    if infer_tokenizer and infer_tokenizer.chat_template is not None:
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
    messages = [{"role":"system", "content":"You are a helpful assistant."}] + messages
    
    kwargs = {}
    if 'enable_thinking' in inspect.signature(infer_tokenizer.apply_chat_template).parameters:
        kwargs = {'enable_thinking': False}
    prompt = infer_tokenizer.apply_chat_template(messages, continue_final_message=True, tokenize=False, **kwargs)
    
    # print(prompt); exit()
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
        resp = complete_generation([prompt_prefix], num_log_probs=params['api_num_log_prob'], seed=params['seed'])
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
        
def get_results_dfs(params, train_sentences, train_labels, test_sentences):
    """Get results for multi-token labels"""
    setup_llm(params['model'], params['api_num_log_prob'], gpu_ids_=params['gpu_ids'])
    global llm_framework
    params['llm_framework'] = llm_framework
    
    all_label_probs = []
    all_label_raw_logits = []    # will be filled with logprobs 
    
    for i, test_sentence in enumerate(test_sentences):
        prompt = construct_prompt(params, train_sentences[i], train_labels[i], test_sentence)
        # if len(train_labels[i])==2:
        #     print([prompt]); exit()
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

def get_results_bfs(params, train_sentences, train_labels, test_sentences):
    """Batched results for multi-token labels using breadth-first trie population."""
    setup_llm(params['model'], params['api_num_log_prob'], gpu_ids_=params['gpu_ids'])
    global llm_framework
    params['llm_framework'] = llm_framework
    
    # 1. Initialize Tries and Prompts for all test instances
    tries = []
    active_paths = [] # List of tuples: (instance_idx, current_prompt, path_list)

    for i, test_sentence in enumerate(test_sentences):
        prompt = construct_prompt(params, train_sentences[i], train_labels[i], test_sentence)
        tries.append(LabelsTrie(params['label_dict']))
        active_paths.append((i, prompt, []))

    # 2. Breadth-First Level Population
    while active_paths:
        prompts_to_fire = []
        meta_info = []

        for idx, prompt, path in active_paths:
            node = tries[idx].root
            for token in path:
                node = node.children[token]
            
            # If node has no children, it's a leaf; if it has 1, logit is deterministic
            if not node.children:
                continue
            if len(node.children) == 1:
                child_token = list(node.children.keys())[0]
                tries[idx].update_logits(path, {child_token: 10.0})
                continue

            # Need LLM for this node
            prompts_to_fire.append(prompt)
            meta_info.append((idx, path))

        # Fire batch to LLM
        if prompts_to_fire:
            # Complete generation handles list of prompts automatically
            batch_resp = complete_generation(prompts_to_fire, num_log_probs=params['api_num_log_prob'], seed=params['seed'])
            
            for i, resp in enumerate(batch_resp['choices']):
                idx, path = meta_info[i]
                token_logits = resp['logprobs']['token_logits'][0]
                tries[idx].update_logits(path, token_logits)

        # 3. Generate the next "Frontier" (Level+1)
        next_active_paths = []
        for idx, prompt, path in active_paths:
            node = tries[idx].root
            for token in path:
                node = node.children[token]
            
            # Add all children of the current node to the next level's queue
            for child_token in node.children:
                next_active_paths.append((idx, prompt + child_token, path + [child_token]))
        
        active_paths = next_active_paths

    # 4. Final probability extraction
    all_label_probs = []
    all_label_raw_logits = []
    
    for trie, label_dict_ref in zip(tries, [params['label_dict']]*len(tries)):
        label_probs = trie.get_all_label_probs()
        all_label_probs.append(label_probs)
    
    eps = 1e-7 # for avoiding log(0)
    for label_probs in all_label_probs:
        if 0 in label_probs:
            label_probs += eps
            
        all_label_raw_logits.append(np.log(label_probs)) # logprobs
        
    # print(np.array(all_label_probs)); exit()
    return np.array(all_label_probs), np.array(all_label_raw_logits)
    
def get_results(params, train_sentences, train_labels, test_sentences):
    return get_results_dfs(params, train_sentences, train_labels, test_sentences)
    # return get_results_bfs(params, train_sentences, train_labels, test_sentences)

# Force single-threaded execution to avoid conflicts
def setup_single_threading():
    """Setup single-threading to avoid vLLM conflicts"""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    # Set torch to use single thread
    torch.set_num_threads(1)
    
    # os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    # os.environ['VLLM_USE_PRECOMPILED'] = '1'
    # os.environ['FLASHINFER_USE_PRECOMPILED'] = '1'
    # os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
    
    # for deterministic behaviour
    # torch.set_num_threads(1)
    # port = 8100
    # os.environ["PYTHONHASHSEED"] = "0"
    # # Use experimental features in LMCache
    # os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    # # LMCache is set to use 256 tokens per chunk
    # os.environ["LMCACHE_CHUNK_SIZE"] = "64"
    # # Disable local CPU backend in LMCache
    # os.environ["LMCACHE_LOCAL_CPU"] = "False"
    # # Set local CPU memory buffer limit to 5.0 GB
    # os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "20.0"
    # # Set the remote URL for LMCache server
    # os.environ["LMCACHE_REMOTE_URL"] = f"lm://localhost:{port}"
    # # Set the serializer/deserializer between vllm and LMCache server
    # # `naive` indicates using raw bytes of the tensor without any compression
    # os.environ["LMCACHE_REMOTE_SERDE"] = "naive"