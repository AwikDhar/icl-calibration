import random
from typing import Dict, List, Optional
import msgspec
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from utils.sampling_utils import get_similarities 

CLASS_EMBED = None
with open("calibration/models/transformer_config.json", "r") as file:
    config = msgspec.json.decode(file.read())
    MAX_NUM_CLASSES = config['context_length'] 

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
def get_batch(data: Dict, batch_size: Optional[int] = None):
    N = data["inputs"].shape[0] # dataset size
    if batch_size is None:
        return (
        data["inputs"],
        data["logits"],
        data["labels"],
    )
        
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
        volume_fraction: float = 1,
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
            corrupt_count = 0
            
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
                    data_size = int(len(sampling_split_data)*volume_fraction)
                    data_idxs = torch.randperm(len(sampling_split_data))[:data_size]
                    for idx in data_idxs:
                        shots_end = shots_end or len(sampling_split_data[idx]["inputs"])
                        sampling_split_data[idx]["inputs"] = torch.tensor(sampling_split_data[idx]["inputs"], dtype=torch.float32, device=device)[:shots_end+1, :]
                        sampling_split_data[idx]["logits"] = torch.tensor(sampling_split_data[idx]["logits"], dtype=torch.float32, device=device)[:shots_end+1, :]
                        sampling_split_data[idx]["labels"] = torch.tensor(sampling_split_data[idx]["labels"], dtype=torch.long, device=device)[:shots_end+1]

                        
                        # Check for NaN in inputs
                        if torch.isnan(sampling_split_data[idx]["logits"]).any():
                            has_nan_count += 1
                        else:
                            # Probabilistic augmentation: clone original and add as extra sample
                            # Only one augmentation active per clone even if both flags are true
                            augmentation = random.choices(
                                ['temp', 'label', None], 
                                [0.5 if temp_augment else 0, 0.3 if label_augment else 0 , 1 - 0.5*temp_augment - 0.3*label_augment], 
                                k=1
                            )[0]
                             
                            apply_temp = augmentation=='temp'
                            apply_label = augmentation=='label'
                            
                            if augmentation:
                                aug_item = {k: v.clone() if isinstance(v, torch.Tensor) else v
                                            for k, v in sampling_split_data[idx].items()}
                            
                            if task=='correctness_pred':
                                recalculate_features(sampling_split_data[idx], 
                                                    temp_augment=False, 
                                                    label_augment=False)
                            elif task=='bias_pred':
                                recalculate_surprise_features(sampling_split_data[idx], 
                                                            temp_augment=False, 
                                                            label_augment=False)
                            else:
                                raise NotImplementedError(f"Task specification for recalculating features not reccognised: {task}")
                                    
                            if sampling_split_data[idx].get('corrupt', False):
                                corrupt_count += 1
                            else:
                                filtered_split_data.append(sampling_split_data[idx])
                                
                            if augmentation:
                                if task=='correctness_pred':
                                    recalculate_features(aug_item,
                                                        temp_augment=apply_temp,
                                                        label_augment=apply_label)
                                elif task=='bias_pred':
                                    recalculate_surprise_features(aug_item,
                                                                temp_augment=apply_temp,
                                                                label_augment=apply_label)
                                
                                if aug_item.get('corrupt', False):
                                    corrupt_count += 1
                                else:
                                    filtered_split_data.append(aug_item)
                                
                    nan_percentage = (has_nan_count / total_items * 100) if total_items > 0 else 0
                    corrupt_percentage = (corrupt_count / total_items * 100) if total_items > 0 else 0
                    if nan_percentage>0:
                        print(f"{llm} | {dataset} | {split}: {nan_percentage:.2f}% items with NaN ({has_nan_count}/{total_items})")
                    if corrupt_percentage>0:
                        print(f"{llm} | {dataset} | {split}: {corrupt_percentage:.2f}% items with corrupt features ({corrupt_count}/{total_items})")
                        
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

def reshuffle_embeddings(data: Dict):
    """Reshuffle the embedding dimensions across all inputs in the data dict."""
    for llm in data:
        for dataset in data[llm]:
            for sampling_strategy in data[llm][dataset]:
                if "train" in data[llm][dataset][sampling_strategy]:
                    inputs = data[llm][dataset][sampling_strategy]["train"]["inputs"]
                    embedding_dim = inputs.shape[-1] - 5
                    idxs = torch.randperm(embedding_dim, device=inputs.device)
                    inputs[:, :, -embedding_dim:] = inputs[:, :, -embedding_dim:][..., idxs]

def create_class_embeddings(num_classes: int, device: str) -> torch.Tensor:
    A = torch.randn(num_classes, num_classes, device=device)
    Q, R = torch.linalg.qr(A)
    # Sign correction for Haar uniformity
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)
    return Q  

def recalculate_features(item: Dict, temp_augment=False, label_augment=False):
    device = item['inputs'].device
    # print(item['inputs'][:5,:3])
    if temp_augment:
        # print("temp augment"); exit()
        apply_temp_augmentation(item)    
        
    T, num_classes = item['logits'].shape
    probs = F.softmax(item['logits'], dim=-1)
    # print(probs[4])        
    pred_probs = probs.max(dim=-1).values

    item['inputs'][:,0] = pred_probs    
    sorted_probs = torch.sort(probs, dim=-1, descending=True).values
    second_highest_probs = sorted_probs[:, [1]]
    
    if label_augment:
        apply_label_augmentation(item)  
        item['inputs'][1:, 1] = (item['logits'][:-1].argmax(dim = -1) == item['labels'][:-1]).float()

    if temp_augment or label_augment:          
        item['inputs'][1:,2] = probs[torch.arange(T-1), item['labels'][:-1]]
    
    gt_prob_mses = torch.cat(
        (
            torch.tensor([[0.25]], device=device), 
            (1 - item['inputs'][1:,[2]])**2
        ), 
        dim=0)

    is_entropy = torch.tensor([
            [float(item['sampling_strategy']=='ENTROPY')]
            for _ in range(len(probs))
            ], device=device
        )

    permute_embeddings = True
    if permute_embeddings:
        embedding: torch.Tensor = item['inputs'][:, -128:]

        embedding_match = (item['inputs'][:, -128:]==item['inputs'][:, 46:]).float().mean()
        assert embedding_match == 1, f"{embedding_match} fraction of embeddings match the expected position"
        
        # embedding = embedding/embedding.norm(dim=-1, keepdim=True)
        
        # idxs = torch.randperm(128, device=item["inputs"].device)
        # permuted_embeddings = embedding[:, idxs]
        permuted_embeddings = embedding
        
        # class_embeddings = get_class_embeddings(item)
        
        item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], second_highest_probs, permuted_embeddings), dim=-1) # main/norm_embed
        # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], second_highest_probs, is_entropy, class_embeddings), dim=-1) # class_embed
        # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], second_highest_probs, is_entropy, class_embeddings, permuted_embeddings), dim=-1) # full embed
        # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], second_highest_probs, class_embeddings, permuted_embeddings), dim=-1) # full embed for similarity sampling
        # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], second_highest_probs, is_entropy, class_embeddings), dim=-1)
        # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], second_highest_probs, class_embeddings), dim=-1)
        pass
    else:
        # similarities = item['inputs'][:, 25:46].clone()
        # print(len(item['inputs'][0])); exit()
        # embeddings = item['inputs'][:, -128:].cpu().numpy()
        # similarities = torch.tril(torch.tensor(get_similarities(embeddings, embeddings)))
        # # print(embeddings[:5]) ; exit()
        # self_mask = torch.eye(MAX_NUM_CLASSES, dtype=torch.bool)[:len(similarities),:]
        # similarities[self_mask]=0
        # max_similarities = torch.max(similarities, dim=-1, keepdim=True).values
        # print(max_similarities[:10])
        
        # similarities = item['inputs'][:, 25:46].clone()
        # self_mask = torch.eye(MAX_NUM_CLASSES, dtype=torch.bool)[:len(similarities),:]
        # similarities[self_mask]=0
        # max_similarities = torch.max(similarities, dim=-1, keepdim=True).values
        # print(max_similarities[:10]); exit()
        # breakpoint()
        # print(item['inputs'][:5,:50], max_similarities, item['sampling_strategy']); exit()
        item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], second_highest_probs), dim=-1)
        # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], second_highest_probs, similarities), dim=-1)
        # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], second_highest_probs, is_entropy), dim=-1)
        # item['inputs'] = torch.cat((gt_prob_mses, item['inputs'][:,:3], second_highest_probs, max_similarities) , dim=-1)
        # item['inputs'] = item['inputs'][:,:2]
        
    # print(item['inputs'].shape); exit()
    # correctness = (item['labels']==item['logits'].argmax(dim=-1)).float().view(T, 1).to(device) # leakage test
    # item['inputs'] = torch.cat((gt_prob_mses, correctness, item['inputs'][:,:3]), dim=-1) # leakage test
    # print(item['inputs']), item['inputs'][:,25:46]
    # print(item['inputs'][-10:,:25])
    # print(item['inputs'][:5,:50], item['sampling_strategy'], temp_augment, label_augment)
 
def get_class_embeddings(item: Dict):
    global CLASS_EMBED
    
    if CLASS_EMBED is None or random.uniform(0,10)<2:
        CLASS_EMBED = create_class_embeddings(num_classes=MAX_NUM_CLASSES, device=item['inputs'].device)
        
    preds = item['logits'].argmax(dim=-1).tolist()
    sorted_preds = sorted(list(set(preds)))
    num_unique_classes = len(sorted_preds)
    
    # shuffled_class_embeddings =  CLASS_EMBED[:num_unique_classes, :]
    idxs = torch.randperm(MAX_NUM_CLASSES)[:num_unique_classes]
    shuffled_class_embeddings =  CLASS_EMBED[idxs, :]
    idxs = torch.randperm(MAX_NUM_CLASSES)
    shuffled_class_embeddings =  shuffled_class_embeddings[:, idxs]
    
    # print(preds[:3], sorted_preds)
    # print(shuffled_class_embeddings)
    class_embedding_map = {
        pred_class : shuffled_class_embeddings[i] for i, pred_class in enumerate(sorted_preds)
    }

    class_embeddings = torch.stack(
        [class_embedding_map[pred_class] for pred_class in preds]
    )
    
    return class_embeddings

    
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
    
# def apply_label_augmentation(item: Dict):
#     mean_conf = item['inputs'][:, 0].mean()
#     beta = 1 if mean_conf < 0.95 else 2
#     max_temp = 1 + torch.clamp(torch.tanh(beta*(mean_conf - 0.5)), min=0, max=0.3)
#     temp = random.uniform(1,max_temp)
#     logits = item['logits'] / temp
    
#     probs = F.softmax(logits, dim=-1)
#     preds = torch.multinomial(probs, 1).squeeze(-1)
#     # print(mean_conf, max_temp, temp, probs.max(dim=-1).values.mean())#; exit()
#     # print(item['labels'], '\n', logits.max(dim=-1).indices, '\n', preds); exit()
    
#     item['labels'] = preds

def apply_label_augmentation(item: Dict) -> None:
    mean_conf = item['inputs'][:, 0].mean()
    simulate_calibrated = random.random() < mean_conf/2
    
    if simulate_calibrated:
        beta = 1 if mean_conf < 0.95 else 2
        max_temp = 1 + torch.clamp(torch.tanh(beta * (mean_conf - 0.5)), min=0, max=0.3)
        temp = random.uniform(1, max_temp.item())
    else:
        beta = 1 if mean_conf < 0.95 else 2
        max_temp = 2 + 3*torch.clamp(torch.tanh(beta * (mean_conf - 0.4)), min=0.5)
        temp = random.uniform(2, max_temp.item())        
        
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

# def check_embeddings(embeddings: torch.Tensor):
    # permuted_embeddings = embedding
    # embeddings = embedding - embedding.mean(dim=0)
    # norms = embeddings.norm(dim=-1, keepdim=True)

    # low_norm_mask = norms.squeeze(-1) < 1e-2
    # if low_norm_mask.any():
    #     item['corrupt'] = True
    #     # print("Corrupt item")
    #     return
    #     # print(f"Low norm embeddings found: {low_norm_mask.sum().item()} / {len(norms)}")
    #     # print(f"Norms: {norms[low_norm_mask].squeeze()}")
    #     # print(f"Embeddings:\n{embeddings[low_norm_mask]}")
    #     # print(f"Original embeddings:\n{embedding[low_norm_mask]}")
    #     # print(f"probabilities: \n{pred_probs}")
    #     # print(f"Correctness labels: \n{item['inputs'][:, 1]}")
    #     # print(f"Labels: \n{item['labels']}")
    # assert torch.min(norms)>0.01