import random
from typing import Dict, List, Optional
import msgspec
from tqdm import tqdm

import torch
import torch.nn.functional as F

def get_batch(data: Dict, batch_size: Optional[int] = None):
    N = data["inputs"].shape[0] # dataset size
    if batch_size is None:
        return (
        data["inputs"],
        data["logits"],
        data["labels"],
    )
        
    # idxs = torch.randint(N, (batch_size,), device=data["inputs"].device)
    idxs = torch.randperm(N, device=data["inputs"].device)[:batch_size]
    return (
        data["inputs"][idxs],
        data["logits"][idxs],
        data["labels"][idxs],
    )

def load_datasets(
        llms: List[str], 
        datasets: list, 
        device: str, 
        feature_type: str, 
        shots_end: int = None,
        splits = ('train', 'test'),
        sampling_strategies = None,
        task='correctness_pred',
        temp_augment: bool = False,
        label_augment: bool = False,
        purpose: str = "calibrator training/eval",
    ):
    """Preload all datasets into GPU memory for fast training (with optional feature recalculation)."""
    data = {}
    tasks = []
    
    for llm in llms:
        data[llm] = {}
        for dataset in datasets:
            tasks.append((llm, dataset))
            
    
    for (llm, dataset) in tqdm(tasks, desc=f"Loading datasets for {purpose}"):
        data[llm][dataset] = {}
        
        for split in splits:
            has_nan_count = 0
            
            path = f"calibration/datasets/{llm.replace('/','_')}/{dataset}/{feature_type}/{split}.json"
            with open(path) as file:
                split_data = msgspec.json.decode(file.read())
                for sampling_strategy in sampling_strategies:
                    data[llm][dataset][sampling_strategy] = {}
                    
                    # print(len(split_data),set([item['sampling_strategy'] for item in split_data]) )
                    sampling_split_data = [item for item in split_data if item['sampling_strategy']==sampling_strategy.upper()] 
                    total_items = len(sampling_split_data)
                    
                    filtered_split_data = []
                    
                    # convert each sample into tensors and recalc features
                    for idx in range(len(sampling_split_data)):
                        shots_end = shots_end or len(sampling_split_data[idx]["inputs"])
                        sampling_split_data[idx]["inputs"] = torch.tensor(sampling_split_data[idx]["inputs"], dtype=torch.float32, device=device)[:shots_end+1, :]
                        sampling_split_data[idx]["logits"] = torch.tensor(sampling_split_data[idx]["logits"], dtype=torch.float32, device=device)[:shots_end+1, :]
                        sampling_split_data[idx]["labels"] = torch.tensor(sampling_split_data[idx]["labels"], dtype=torch.long, device=device)[:shots_end+1]

                        
                        # Check for NaN in inputs
                        if torch.isnan(sampling_split_data[idx]["logits"]).any():
                            has_nan_count += 1
                        else:
                            
                            if task=='correctness_pred':
                                recalculate_features(sampling_split_data[idx], 
                                                    temp_augment=temp_augment, 
                                                    label_augment=label_augment)
                            elif task=='bias_pred':
                                recalculate_surprise_features(sampling_split_data[idx], 
                                                            temp_augment=temp_augment, 
                                                            label_augment=label_augment)
                            else:
                                raise NotImplementedError(f"Task specification for recalculating features not reccognised: {task}")
                        
                        filtered_split_data.append(sampling_split_data[idx])

                    nan_percentage = (has_nan_count / total_items * 100) if total_items > 0 else 0
                    if nan_percentage>0:
                        print(f"{llm} | {dataset} | {split}: {nan_percentage:.2f}% items with NaN ({has_nan_count}/{total_items})")

                    # breakpoint()
                    # batchify entire dataset (only non-NaN items)
                    inputs_all = torch.stack([item["inputs"] for item in filtered_split_data])
                    logits_all = torch.stack([item["logits"] for item in filtered_split_data])
                    labels_all = torch.stack([item["labels"] for item in filtered_split_data])

                    data[llm][dataset][sampling_strategy][split] = {
                        "inputs": inputs_all,
                        "logits": logits_all,
                        "labels": labels_all
                    }

    return data

def recalculate_features(item: Dict, temp_augment=False, label_augment=False):
    device = item['inputs'].device
    # print(item['inputs'])

    if temp_augment:
        apply_temp_augmentation(item)    
        
    T, num_classes = item['logits'].shape
    probs = F.softmax(item['logits'], dim=-1)
        
    pred_probs = probs.max(dim=-1).values

    item['inputs'][:,0] = pred_probs    
    sorted_probs = torch.sort(probs, dim=-1, descending=True).values
    second_highest_probs = sorted_probs[:, [1]]
    
    if label_augment:
        apply_label_augmentation(item)  
        item['inputs'][1:, 1] = (item['logits'][:-1].argmax(dim = -1) == item['labels'][:-1]).float()

    if temp_augment or label_augment:  
        # shifted_gt_probs = torch.concatenate(
        #     (
        #         torch.tensor([0.5], device=device), 
        #         probs[torch.arange(T-1), item['labels'][:-1]]
        #     )
        # ) # exclude last position and shift
        
        item['inputs'][1:,2] = probs[torch.arange(T-1), item['labels'][:-1]]
    
    gt_prob_mses = torch.cat(
        (
            torch.tensor([[0.25]], device=device), 
            (1 - item['inputs'][1:,[2]])**2
        ), 
        dim=0)
    # gt_prob_mses = torch.zeros((T,1), device=device)
    # gt_prob_mses[0,0] = 0.5
    # gt_prob_mses[1:,[0]] = (1 - item['inputs'][1:,[2]])**2

    permute_embeddings = True
    if permute_embeddings:
        embedding = item['inputs'][:, 46:]
        # idxs = torch.randperm(128, device=item["inputs"].device)
        permuted_embeddings = embedding#[:, idxs]

        item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], second_highest_probs, permuted_embeddings), dim=-1)
        # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], permuted_embeddings), dim=-1)
    else:
        # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], second_highest_probs), dim=-1)
        # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3]), dim=-1)
        item['inputs'] = item['inputs'][:,:2]
        
    # print(item['inputs']); exit()
    # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], item['inputs'][:, 25:46]), dim=-1)
    # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3]), dim=-1)
    # correctness = (item['labels']==item['logits'].argmax(dim=-1)).float().view(T, 1).to(device) # leakage test
    # item['inputs'] = torch.cat((gt_prob_mses, correctness, item['inputs'][:,:3]), dim=-1) # leakage test
    # print(item['inputs']), item['inputs'][:,25:46]
    # item['inputs'][:, 0] = torch.linspace(0.5, 1, len(item['inputs']))
    # item['inputs'][:, 2] = 0
    # print(item['inputs'][:,1])
    # print(item['logits'].argmax(dim=-1).float())
    # print(item['labels'].float()); exit()
    # print(item['inputs'][-10:,:25])
    # print(item['inputs'][-5:,:50])
    # exit()
    # add_noise_to_features(item)
 
def recalculate_surprise_features(item: Dict, temp_augment=False, label_augment=False, max_classes=100):
    device = item['inputs'].device
    T, num_classes = item['logits'].shape

    assert num_classes <= max_classes, f"num_classes ({num_classes}) exceeds max_classes ({max_classes})"

    if temp_augment:
        apply_temp_augmentation(item)    
    
    if label_augment:
        apply_label_augmentation(item)  

    probs = F.softmax(item['logits'], dim=-1)   
    logprobs = torch.log(probs)
    surprise_vectors = logprobs
    timestep_idxs = torch.arange(T, device=device)
    surprise_vectors[timestep_idxs, item['labels']] *= -1
    
    padded_suprise_vectors = torch.zeros(T, max_classes, device=device)
    padded_suprise_vectors[:, :num_classes] = surprise_vectors

    item['inputs'] = padded_suprise_vectors
    # print(item['inputs'], item['labels']); exit()
    
def apply_temp_augmentation(item: Dict):
    item['inputs'][0][2] = 0.5
    mean_conf = item['inputs'][:, 0].mean()
    # mean_correctness = (item['logits'][:-1].argmax(dim = -1) == item['labels'][:-1]).mean()
    beta = 4 if mean_conf < 0.95 else 5
    max_temp = 1 + torch.clamp(torch.tanh(beta * (mean_conf - 0.5)), min=0) 
    # else:
    #     max_temp = 4
    # max_temp = get_correctness_matching_temp(item)
    temp = random.uniform(1,max_temp)
    item['logits'] /= temp
    # print(mean_conf.item(), temp.item()); exit()
    
def get_correctness_matching_temp(item):
    mean_correctness = (item['logits'][:-1].argmax(dim = -1) == item['labels'][:-1]).mean()
    mean_conf = item['logits'].softmax(dim=-1).max(dim=-1).values().mean()
    
    if mean_conf < mean_correctness:
        return 1.0
    
    T_low, T_high = 1, 3
    
    for i in range(10):
        temperature = (T_low + T_high)/2
        scaled_logits = item['logits']/temperature
        mean_conf = scaled_logits.softmax(dim=-1).max(dim=-1).values().mean()
        
        gap = mean_conf - mean_correctness
        if gap>0:
            if gap<0.02:
                return temperature
            T_low = temperature
        else:
            T_high = temperature
            
    return temperature
    
def apply_label_augmentation(item: Dict):
    mean_conf = item['inputs'][:, 0].mean()
    max_temp = 1 + torch.clamp(torch.tanh(4 * (mean_conf - 0.5)), min=0)
    temp = random.uniform(1,max_temp)
    logits = item['logits'] / temp
    
    probs = F.softmax(logits, dim=-1)
    preds = torch.multinomial(probs, 1).squeeze(-1)
    
    item['labels'] = preds
        
def add_noise_to_features(item: Dict):
    device = item['inputs'].device

    T, C = item['inputs'].shape
    perturbed_shots = 7
    noise = torch.normal(mean=1, std=5, size=(perturbed_shots, C), device=device)
    
    item['inputs'][:perturbed_shots,:] += noise