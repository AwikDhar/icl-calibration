import gc
from vllm import LLM, SamplingParams, config, PoolingParams
# from vllm.config.compilation import CompilationConfig, PassConfig, CUDAGraphMode
# from vllm.sampling_params import StructuredOutputsParams

import inspect
import json
from pathlib import Path
import numpy as np
import os
import torch
from tqdm import tqdm
from typing import Callable, List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, EetqConfig
import torch
from sentence_transformers import SentenceTransformer
from gemini.gemini_model import GeminiModel

from calibration.model import CalibrationTransformer, PositionEmbeddingType
from calibration.temperature import get_equivalent_temp, get_shotwise_dynamic_temperatures

from labels_trie import LabelsTrie
from llm_framework import LlmFramework
from llm_result import LlmResult, Logprobs
from utils.gen_utils import ROOT_DIR, get_llm_framework

import logging

def setup_logger(name: str = None):
    logFormatter = logging.Formatter(
        "{asctime} - {levelname} - {message}", 
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    name = name or __name__
    logger = logging.getLogger(name)

    fileHandler = logging.FileHandler(ROOT_DIR/"calibration.log")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)

    return logger

logger = setup_logger(__name__)
logger.propagate = False

logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)

llm_framework = LlmFramework.VLLM
embedding_framework = LlmFramework.VLLM
infer_model = None
compile = False
infer_tokenizer = None
calibrator = None
gpu_ids = None
embedding_models = {}

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_model_snapshot_path(model_name, cache_dir=None):
    global llm_framework
    
    if 'embedding' not in model_name.lower() and llm_framework is LlmFramework.GOOGLE:
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
          
def calculate_utilization(model_name: str, num_gpus: int, buffer_factor: float = 2) -> float:
    total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    total_vram_gb = total_vram_bytes / (1024**3)
    
    if "4B" in model_name:
        est_model_size_gb = 8
    elif "300m" in model_name.lower():
        est_model_size_gb = 0.6
    else:
        est_model_size_gb = 4 

    needed_per_gpu = (est_model_size_gb * buffer_factor) / num_gpus
    
    utilization = needed_per_gpu / total_vram_gb
    return min(utilization, 0.9)

# Get embeddings of sentences at a specified truncated dim. 
# Full embeddings for similarity sampling, truncated embeddings for calibrator features for easier learning 
def get_embeddings(params: Dict, sentences: List[str], embedding_model_name: str):
    global embedding_models
    global embedding_framework
    
    if embedding_model_name=='hidden_features':
        return get_hidden_feature_embeddings(hidden_features=sentences)
    
    setup_embedding_model(embedding_model_name)

    if embedding_framework is LlmFramework.HF:
        with torch.inference_mode():
            sentences_embeddings = embedding_models[embedding_model_name].encode(sentences, show_progress_bar=False, prompt_name="Classification", convert_to_numpy=True) 

    elif embedding_framework is LlmFramework.VLLM:
        # sentences = [get_embedding_prompt(embedding_model_name, sentence) for sentence in sentences]
        embedding_model = embedding_models[embedding_model_name]
        truncate_dim = params.get('embedding_dim', 128)
        pooling_params = PoolingParams(dimensions=truncate_dim)
            
        outputs = embedding_model.embed(sentences, pooling_params=pooling_params, truncate_prompt_tokens=-1, use_tqdm=False)
        sentences_embeddings = np.array([output.outputs.embedding for output in outputs])

    return sentences_embeddings

def get_hidden_feature_embeddings(hidden_features: List[np.ndarray]):
    if not isinstance(hidden_features, np.ndarray):
        hidden_features = np.array(hidden_features)
    
    embedding_dim = 128
    hidden_dim = hidden_features.shape[1]

    # Zero-mean Gaussian scaled by 1/sqrt(hidden_dim) satisfies JL property, preserving cosine similarity
    down_proj = np.random.randn(hidden_dim, embedding_dim) / np.sqrt(hidden_dim)

    hidden_features = hidden_features @ down_proj
    hidden_features = hidden_features / np.linalg.norm(hidden_features, axis=-1, keepdims=True)

    return hidden_features

def setup_embedding_model(embedding_model_name: str):
    global embedding_models
    global gpu_ids
    global embedding_framework
    
    if embedding_model_name not in embedding_models:
        HF_HOME = os.environ.get('HF_HOME')     
        cache_dir = os.environ.get('HF_HUB_CACHE', HF_HOME)

        model_snapshot_or_card = get_model_snapshot_path(embedding_model_name, cache_dir)
        
        if embedding_framework is LlmFramework.HF:
            attn_implementation = "kernels-community/flash-attn2" 
            embedding_models[embedding_model_name] = SentenceTransformer(model_snapshot_or_card, device=f'cuda:0', truncate_dim=128, cache_folder=cache_dir, 
                                                                         model_kwargs={'dtype':torch.bfloat16, 'attn_implementation':attn_implementation})
        elif embedding_framework is LlmFramework.VLLM:
            
            # max_model_len = 20000 if 'qwen' in embedding_model_name.lower() else None
            
            embedding_models[embedding_model_name] = LLM(
                model=model_snapshot_or_card,
                runner="pooling",
                max_model_len=None,
                # attention_config=config.AttentionConfig(backend=backend),
                tensor_parallel_size=1,
                download_dir=cache_dir,
                seed=42,
                gpu_memory_utilization=calculate_utilization(embedding_model_name, 1),
                hf_overrides={"is_matryoshka": True},
                trust_remote_code=True,
            )
                         
def get_embedding_prompt(model_name: str, input_text: str, task_desc: str = "classification: "):
    if "qwen" in model_name.lower():
        # Qwen3 format: Instruct: {task}\nQuery:{query}
        full_prompt = f"Instruct: {task_desc}\nQuery: {input_text}"
    else:
        # Gemma format: task: {task} | query: {query}
        full_prompt = f"task: {task_desc} | query: {input_text}"
        
    return full_prompt
    
def setup_llm(model_name, gpu_ids_=[0]):
    global infer_model
    global gpu_ids
    global llm_framework
    
    if infer_model is None:

        llm_framework = get_llm_framework(model_name)

        gpu_ids = gpu_ids_

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
                    max_num_batched_tokens=200_000, 
                    max_num_seqs=8,  
                    enable_prefix_caching=True,
                    enforce_eager=True,
                    limit_mm_per_prompt={"image": 0, "video": 0}, # to skip any initialization/tuning for multimodal inputs
                    max_logprobs=-1,
                    download_dir=cache_dir,
                    gpu_memory_utilization=0.85,
                    trust_remote_code=True
                )
                
                # logger = logging.getLogger(__name__)
                # logger.setLevel(logging.INFO)
                logger.info(f"Loaded {model_name} via vllm")
            
            case LlmFramework.HF:
                # attn_implementation="flash_attention_2"
                attn_implementation="kernels-community/flash-attn2"
                infer_model = AutoModelForCausalLM.from_pretrained(model_snapshot_or_card, use_cache=False, trust_remote_code=False, dtype=torch.bfloat16,
                                                                # quantization_config=quantization_config, 
                                                                device_map=f"cuda:0",#tp_plan="auto",
                                                                attn_implementation=attn_implementation,
                                                                cache_dir=cache_dir)
                
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
    device = f'cuda:0'

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
        
    if model_path is None:
        model_path = f"./calibration/models/llm_agnostic/{params['calibrator_name']}"

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    calibrator.load_state_dict(state_dict)
    
    return calibrator.to(device)

def transformer_calibrate(params:Dict, inputs: torch.Tensor, logits: torch.Tensor):
    global calibrator
    if calibrator is None:
        setup_calibrator(params)
    device=f'cuda:0'
    
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

def complete_generation_hf(prompts, label_token_ids, batch_size=8, num_log_probs=None, in_context_logprobs=False):
    # 1. Store original order and sort everything by length
    # This ensures that prompts within each chunk of 8 have similar lengths
    original_indices = list(range(len(prompts)))
    indexed_prompts = sorted(zip(original_indices, prompts), key=lambda x: len(x[1]))
    # indexed_prompts = zip(original_indices, prompts)
    
    sorted_indices, sorted_prompts = zip(*indexed_prompts)
    
    results_dict = {} 
    chunked_prompts = list(chunks(list(sorted_prompts), batch_size))
    chunked_indices = list(chunks(list(sorted_indices), batch_size))
    
    prompt_iterator = tqdm(enumerate(chunked_prompts), total=len(chunked_prompts), desc="processing prompts") if len(prompts)>20 else enumerate(chunked_prompts)

    for i, test_chunk_prompts in prompt_iterator:
        resp = complete_batch_generation_hf(test_chunk_prompts, label_token_ids, num_log_probs=num_log_probs, in_context_logprobs=in_context_logprobs)
        
        current_chunk_indices = chunked_indices[i]
        for local_idx, answer in enumerate(resp['choices']):
            orig_idx = current_chunk_indices[local_idx]
            results_dict[orig_idx] = answer
            
    # 4. Shuffle back to original order
    results = {
        'choices':[results_dict[i] for i in range(len(prompts))]
        }
    
    return results
            
def complete_batch_generation_hf(prompts, label_token_ids, num_log_probs=None, in_context_logprobs=False):    
    with torch.inference_mode():
        if isinstance(prompts, str):
            prompts = [prompts] # the code below assumes a list

        input_ids = infer_tokenizer(prompts, padding=True, return_tensors='pt').to(next(infer_model.parameters()).device)
        
        # we are left padding, so we need to adjust the position IDs
        attention_mask = input_ids['attention_mask']
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
       # get the logits for the input context
        outputs = infer_model.forward(
            input_ids=input_ids['input_ids'], 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            output_hidden_states=True, 
            return_dict=True
        )        
        
        logits = outputs.logits.detach()
        hidden_states = outputs.hidden_states
     
        label_token_ids = torch.tensor(label_token_ids)
        label_token_strings = [infer_tokenizer.decode(token_id.item()) for token_id in label_token_ids]
         
        # create the return value to resemble OpenAI
        return_json = {}
        choices = []
        
        if in_context_logprobs:
            in_context_token_positions = get_in_context_token_positions(input_ids['input_ids'])

        for batch_idx in range(len(prompts)):
            curr_json = {}

            # fill the return json with the top tokens and probs to match the OpenAI return value.
            curr_json['logprobs'] = get_logprobs_for_position(logits, hidden_states, label_token_ids, label_token_strings, batch_idx, position=-1)
            
            if in_context_logprobs:
                for in_context_token_position in in_context_token_positions[batch_idx]: 
                    curr_json['in_context_logprobs'] = get_logprobs_for_position(logits, hidden_states, label_token_ids, label_token_strings,
                                                                                batch_idx, position=in_context_token_position,
                                                                                logprobs=curr_json.get('in_context_logprobs'))
                
            choices.append(curr_json)
            
        return_json['choices'] = choices
        # print(prompts[-1], curr_json['logprobs']['token_logits'])
        # exit()
        return return_json

def get_logprobs_for_position(logits, hidden_states, label_token_ids, label_token_strings, batch_idx, position, logprobs=None):
    if not logprobs:
        logprobs = {
            'top_logprobs':[],
            'token_logits':[],
            'hidden_states':[]
        }
    
    top_logits = logits[batch_idx][position][label_token_ids].to(torch.float32)
    top_logprobs = torch.log_softmax(top_logits, dim=-1)
    
    num_layers = len(hidden_states)
    percentile_layers = [int(p * num_layers) for p in (0.6, 0.7, 0.8)]
    cur_position_hidden_states = np.concatenate(
        [hidden_states[l][batch_idx][position].cpu().float().numpy() for l in percentile_layers]
    )
    
    temp, temp_logits = {}, {}
    
    for log_prob, token_string, logit in zip(top_logprobs, label_token_strings, top_logits):
        if token_string not in temp.keys():
            temp[token_string] = log_prob.item()
            temp_logits[token_string] = logit.item()
    
    logprobs['top_logprobs'].append(temp)
    logprobs['token_logits'].append(temp_logits)
    logprobs['hidden_states'].append(cur_position_hidden_states.cpu().float().numpy())
    
    return logprobs

def get_in_context_token_positions(input_ids: torch.Tensor):    
    input_ids = input_ids.to('cuda:0')
    tok0, tok1 = input_ids[0, -2], input_ids[0, -1]
    
    # tokdec = infer_tokenizer.decode(input_ids[0, -3])
    # tok0dec = infer_tokenizer.decode(tok0)
    # tok1dec = infer_tokenizer.decode(tok1)
    # print('\n',tokdec, tok0dec, '|', tok1dec,'\n')
    # print(tokdec[-1]=='\n', tokdec[-2], len(tokdec), tok0dec=='Category',  tok1dec==':')
    
    matches_tok0 = (input_ids[:, :-1] == tok0)
    matches_tok1 = (input_ids[:, 1:]  == tok1)
    pair_matches = matches_tok0 & matches_tok1  # [batch, seq_len-1]
    
    # nonzero returns sorted, so shape is [batch * num_shots, 2]
    all_positions = pair_matches.nonzero()
    all_positions[:, 1] += 1  # shift to the ":" token
    
    # reshape to [batch, num_shots] and drop last column (test query)
    num_shots = (pair_matches[0].sum().item())  # includes test query
    shots_per_item = pair_matches.sum(dim=1)  # [batch]
    assert (shots_per_item == num_shots).all(), \
        f"Inconsistent number of a_prefix matches across batch items: {shots_per_item.tolist()}. " \
        f"Expected {num_shots} everywhere. Possible padding token collision or mismatched prompts."
        
    positions = all_positions[:, 1].reshape(input_ids.shape[0], num_shots)
    
    return positions[:, :-1]  # [batch, num_shots-1], drop test query

def complete_generation_vllm(prompts, label_token_ids=None, num_log_probs=None, in_context_logprobs=False, sample_logprobs=False, sample_n=20):
    ''' This function runs inference using vLLM but places the outputs into a json that looks just like the one
     provided by the OpenAI API. '''
    global infer_tokenizer
    
    if isinstance(prompts, str):
        prompts = [prompts]  # the code below assumes a list
    
    sampling_params = SamplingParams(
        temperature=1,  
        max_tokens=1,    
        logprobs=len(label_token_ids),
        # logprobs=1000,
        # flat_logprobs=True,
        # prompt_logprobs=len(label_token_ids),  
        skip_special_tokens=True,
        allowed_token_ids=label_token_ids,
        seed=42,
    )
    
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
            curr_json['logprobs']['token_logits'] = []
            
            if generated_token.logprobs:
                token_logprobs = generated_token.logprobs[0]
                # print(token_logprobs); exit()

                top_tokens = list(token_logprobs.keys())
                if top_tokens:
                    top_token = max(token_logprobs.keys(), key=lambda x: token_logprobs[x].logprob)
                    curr_json['logprobs']['tokens'].append(infer_tokenizer.decode([top_token]))
                    curr_json['logprobs']['token_logprobs'].append(token_logprobs[top_token].logprob)
                
                temp = {}
                temp_logits = {}
                
                logits = torch.tensor([logprob_data.logprob for logprob_data in token_logprobs.values()])
                log_sum_exp = torch.logsumexp(logits, dim=0)

                if sample_logprobs:
                    # Build probability distribution over label tokens from the vLLM logprobs,
                    # then simulate N generations by sampling from it, and estimate probabilities
                    # from the resulting frequencies — as if we had no access to logprobs.
                    token_ids = list(token_logprobs.keys())
                    token_log_probs = torch.tensor([token_logprobs[tid].logprob for tid in token_ids])
                    token_probs = token_log_probs.softmax(dim=0)  # true distribution we pretend not to know

                    eps = 1e-7
                    sampled_indices = torch.multinomial(token_probs, num_samples=sample_n, replacement=True)
                    counts = torch.bincount(sampled_indices, minlength=len(token_ids)).float()
                    counts = counts + eps                          # Laplace smoothing: Dirichlet prior with alpha=eps
                    freq_probs = counts / counts.sum()             # re-normalize after smoothing
                    freq_log_probs = torch.log(freq_probs)         # clean, no further eps needed
                    freq_log_sum_exp = torch.logsumexp(freq_log_probs, dim=0)
                    # print(token_probs, freq_probs, sample_n); exit()
                    
                    for i, token_id in enumerate(token_ids):
                        token_str = infer_tokenizer.decode([token_id])
                        if token_str not in temp:
                            temp[token_str] = (freq_log_probs[i] - freq_log_sum_exp).item()
                            temp_logits[token_str] = freq_log_probs[i].item()
                else:
                    for token_id, logprob_data in token_logprobs.items():
                        token_str = infer_tokenizer.decode([token_id])
                        if token_str not in temp:
                            temp[token_str] = logprob_data.logprob - log_sum_exp
                            temp_logits[token_str] = logprob_data.logprob  
                
                curr_json['logprobs']['top_logprobs'].append(temp)
                curr_json['logprobs']['token_logits'].append(temp_logits)
        
        choices.append(curr_json)
    
    # token_strs = [infer_tokenizer.decode([token_id]) for token_id in label_token_ids]
    # print(curr_json['logprobs']['top_logprobs'], token_strs)
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

def complete_generation(prompt, label_token_ids=None, num_log_probs=None, in_context_logprobs=False):
    """complete the prompt using a language model"""
    global llm_framework
     
    match llm_framework:
        case LlmFramework.GOOGLE:
            return complete_generation_google(prompt, num_log_probs=num_log_probs)
        case LlmFramework.VLLM:
            return complete_generation_vllm(prompt, label_token_ids, num_log_probs=num_log_probs, in_context_logprobs=in_context_logprobs)
        case LlmFramework.HF:
            return complete_generation_hf(prompt, label_token_ids, num_log_probs=num_log_probs, in_context_logprobs=in_context_logprobs)

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
    prompt += a_prefix[:-1] # no trailing space, so that modle outputs ' labeltoken'
    # if(len(train_sentences))==0:
    # print(prompt)
    # exit()
    return prompt

def construct_chat_prompt(params, train_sentences, train_labels, test_sentence):        
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]

    messages = []
    for i, (sentence, label_idx) in enumerate(zip(train_sentences, train_labels)):    
        if a_prefix[:-1] in sentence:
            logger.warning("Found the answer prefix in the input sentence. Replacing with lowercase string")
            sentence = sentence.replace(a_prefix[:-1], a_prefix[:-1].lower()) # to prevent the answer prefix being picked from input
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
    # prompt = infer_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kwargs)
    prompt = infer_tokenizer.apply_chat_template(messages, continue_final_message=True, tokenize=False, **kwargs)
    # prompt += a_prefix[:-1]
    
    assert prompt[-1]!=' '
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
        resp = complete_generation([prompt_prefix], num_log_probs=params['api_num_log_prob'])
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
        
def get_results_dfs(params, train_sentences, train_labels, test_sentences, in_context_logprobs=False):
    """Get results for multi-token labels"""
    setup_llm(params['model'], gpu_ids_=params['gpu_ids'])
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
        first_logits = [token.logit for token in list(label_trie.root.children.values())]
        label_trie.print_trie(); print(first_logits, np.ma.masked_invalid(first_logits).mean()); print(label_probs); exit()
        # print([label_trie.root.children[child].__dict__ for child in label_trie.root.children])
        # assert len(zero_toks) == sum(label_probs==0), f"{len(zero_toks) , sum(label_probs==0)} {zero_toks}, {label_probs==0}"; exit()
        all_label_probs.append(label_probs)
    
    # eps = 1e-7 # for avoiding log(0)
    for label_probs in all_label_probs:
        # if 0 in label_probs:
        #     label_probs += eps
            
        all_label_raw_logits.append(np.log(label_probs)) # logprobs
        
    # print(np.array(all_label_probs)); exit()
    return np.array(all_label_probs), np.array(all_label_raw_logits)

def get_results_bfs(params, train_sentences, train_labels, test_sentences, in_context_logprobs=False):
    """Batched results for multi-token labels using breadth-first trie population."""
    setup_llm(params['model'], gpu_ids_=params['gpu_ids'])
    global infer_tokenizer
    global llm_framework
    params['llm_framework'] = llm_framework

    num_shots = params['num_shots']    
    
    # 1. Initialize Tries and Prompts for all test instances
    tries = []
    in_context_tries = []
    active_paths = [] # List of tuples: (instance_idx, current_prompt, path_list)

    for i, test_sentence in enumerate(test_sentences):
        prompt = construct_prompt(params, train_sentences[i], train_labels[i], test_sentence)
        tries.append(LabelsTrie(params['label_dict']))
        in_context_tries.append([LabelsTrie(params['label_dict']) for shot in range(num_shots)])
        active_paths.append((i, prompt, []))

    # 2. Breadth-First Level Population
    while active_paths:
        prompts_to_fire = []
        meta_info = []
        
        # label tokens/token ids whose logprobs we want at this level of label token trie. Only these logprobs will be returned by vllm for efficiency
        added_label_tokens = set()
        label_token_ids = []
        
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
            
            cur_label_tokens = set(node.children.keys())
            added_label_tokens = added_label_tokens.union(cur_label_tokens)
         
        label_token_ids = None
        if llm_framework!=LlmFramework.GOOGLE:       
            label_token_ids = [infer_tokenizer.encode(label_token, add_special_tokens=False)[0] for label_token in added_label_tokens]
        
        # Fire batch to LLM
        if prompts_to_fire:
            # Complete generation handles list of prompts automatically
            batch_resp = complete_generation(prompts_to_fire, label_token_ids, num_log_probs=params['api_num_log_prob'], in_context_logprobs=in_context_logprobs)
            
            for i, resp in enumerate(batch_resp['choices']):
                test_idx, path = meta_info[i]
                token_logits = resp['logprobs']['token_logits'][0]
                tries[test_idx].update_logits(path, token_logits)
                # tries[test_idx].print_trie(); exit()
                hidden_features = resp['logprobs'].get('hidden_states')
                if hidden_features:
                    tries[test_idx].root.hidden_features = hidden_features[0]
                     
                if in_context_logprobs:
                    assert path == [], f"Populating logits for in context examples not supported for label tries of depth > 1. Current depth {len(path)+1}"
                    
                    for shot in range(num_shots):
                        token_logits = resp['in_context_logprobs']['token_logits'][shot]
                        in_context_tries[test_idx][shot].update_logits(path, token_logits)
                        
                        hidden_features = resp['in_context_logprobs'].get('hidden_states')
                        if hidden_features:
                            in_context_tries[test_idx][shot].root.hidden_features = hidden_features[shot]
                            
                    
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
    llm_result = LlmResult(logprobs=[], in_context_logprobs=[])
    
    for test_idx, trie in enumerate(tries):
        label_probs = trie.get_all_label_probs()
        
        eps = 1e-7 if llm_framework==LlmFramework.GOOGLE else 0
        llm_result.logprobs.append(
            Logprobs(probs=[label_probs], logits=[np.log(label_probs + eps)])
        )
        
        if trie.root.hidden_features is not None:
            llm_result.logprobs[test_idx].hidden_features = [trie.root.hidden_features]
            
        if in_context_logprobs:
            llm_result.in_context_logprobs.append(Logprobs(probs=[], logits = []))
            ic_logprobs = llm_result.in_context_logprobs[test_idx]
            
            for shot in range(num_shots):
                shot_label_probs = in_context_tries[test_idx][shot].get_all_label_probs()
                shot_hidden_features = in_context_tries[test_idx][shot].root.hidden_features

                ic_logprobs.probs.append(shot_label_probs)
                ic_logprobs.logits.append(np.log(shot_label_probs))
                if shot_hidden_features is not None:
                    if ic_logprobs.hidden_features is None:
                        ic_logprobs.hidden_features = []
                    ic_logprobs.hidden_features.append(shot_hidden_features)                        

        llm_result.to_numpy()
        
    return llm_result
    
def get_results(params, train_sentences, train_labels, test_sentences, in_context_logprobs=False):
    # return get_results_dfs(params, train_sentences, train_labels, test_sentences, prompt_logprobs=False)
    return get_results_bfs(params, train_sentences, train_labels, test_sentences, in_context_logprobs=in_context_logprobs)

# Force single-threaded execution to avoid conflicts
def setup_single_threading():
    """Setup single-threading to avoid vLLM conflicts"""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    # Set torch to use single thread
    torch.set_num_threads(1)
