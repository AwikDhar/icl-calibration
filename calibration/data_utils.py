import random
from typing import Dict, List, Optional
import msgspec

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
        sampling_strategy = None,
        temp_augment: bool = False,
        label_augment: bool = False
    ):
    """Preload all datasets into GPU memory for fast training (with optional feature recalculation)."""
    data = {}

    for llm in llms:
        data[llm] = {}
        for dataset in datasets:
            data[llm][dataset] = {}
            for split in splits:
                has_nan_count = 0
                
                path = f"calibration/datasets/{llm.replace('/','_')}/{dataset}/{feature_type}/{split}.json"
                with open(path) as file:
                    split_data = msgspec.json.decode(file.read())
                    if sampling_strategy:
                        split_data = [item for item in split_data if item['sampling_strategy']==sampling_strategy.upper()] 

                total_items = len(split_data)
                filtered_split_data = []
                
                # convert each sample into tensors and recalc features
                for idx in range(len(split_data)):
                    shots_end = shots_end or len(split_data[idx]["inputs"])
                    split_data[idx]["inputs"] = torch.tensor(split_data[idx]["inputs"], dtype=torch.float32, device=device)[:shots_end+1, :]
                    split_data[idx]["logits"] = torch.tensor(split_data[idx]["logits"], dtype=torch.float32, device=device)[:shots_end+1, :]
                    split_data[idx]["labels"] = torch.tensor(split_data[idx]["labels"], dtype=torch.long, device=device)[:shots_end+1]

                    
                    # Check for NaN in inputs
                    if torch.isnan(split_data[idx]["logits"]).any():
                        has_nan_count += 1
                    else:
                        recalculate_features(split_data[idx], temp_augment=temp_augment, label_augment=label_augment)
                        filtered_split_data.append(split_data[idx])

                nan_percentage = (has_nan_count / total_items * 100) if total_items > 0 else 0
                if nan_percentage>0:
                    print(f"{llm} | {dataset} | {split}: {nan_percentage:.2f}% items with NaN ({has_nan_count}/{total_items})")

                # batchify entire dataset (only non-NaN items)
                inputs_all = torch.stack([item["inputs"] for item in filtered_split_data])
                logits_all = torch.stack([item["logits"] for item in filtered_split_data])
                labels_all = torch.stack([item["labels"] for item in filtered_split_data])

                data[llm][dataset][split] = {
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
        # normalized_entropies = -(probs * torch.log(probs + 1e-9)).sum(dim=-1) / torch.log(torch.tensor(num_classes)) # (B, num_classes) × (num_classes, B) --> (B, B) --> diagonal elements

        item['inputs'][:,0] = pred_probs    
        # item['inputs'][:,3] = normalized_entropies
    
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
            torch.tensor([[0.5]], device=device), 
            (1 - item['inputs'][1:,[2]])**2
        ), 
        dim=0)
    # gt_prob_mses = torch.zeros((T,1), device=device)
    # gt_prob_mses[0,0] = 0.5
    # gt_prob_mses[1:,[0]] = (1 - item['inputs'][1:,[2]])**2

    # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], item['inputs'][:, 46:]), dim=-1)
    # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], item['inputs'][:, 46:]), dim=-1)
    item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3]), dim=-1)
    # correctness = (item['labels']==item['logits'].argmax(dim=-1)).float().view(T, 1).to(device) # leakage test
    # item['inputs'] = torch.cat((gt_prob_mses, correctness, item['inputs'][:,:3]), dim=-1) # leakage test
    # item['inputs'] = item['inputs'][:,:2]
    # print(item['inputs']), item['inputs'][:,25:46]
    # item['inputs'][:, 0] = torch.linspace(0.5, 1, len(item['inputs']))
    # item['inputs'][:, 2] = 0
    # print(item['inputs'][:,1])
    # print(item['logits'].argmax(dim=-1).float())
    # print(item['labels'].float()); exit()
    # print(item['inputs'][-5:,:50])
    # exit()
    # add_noise_to_features(item)
 
def apply_temp_augmentation(item: Dict):
    item['inputs'][0][2] = 0.5
    mean_conf = item['inputs'][:, 0].mean()

    max_temp = 1 + torch.clamp(torch.tanh(4 * (mean_conf - 0.5)), min=0)
    temp = random.uniform(1,max_temp)
    item['logits'] /= temp
    # print(mean_conf.item(), temp.item()); exit()
    
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