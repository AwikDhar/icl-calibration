from pathlib import Path
import numpy as np
import os
import torch
import random
from typing import List
from collections import Counter
from sampling_strategies import EntropyLevels

import logging
logger = logging.getLogger(__name__)

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
    elif isinstance(entropy_level, EntropyLevels) and num>=1:
        match entropy_level:
            case EntropyLevels.MAX:
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
            case EntropyLevels.RANDOM | EntropyLevels.RANDOM_SHARED:
                idxs = np.random.choice(len(labels), size=num, replace=False)
            case EntropyLevels.LABELSPIKE:
                test_label_idxs = [i for i, lb in enumerate(labels) if lb==test_label]
                other_label_idxs = [i for i, lb in enumerate(labels) if lb!=test_label]

                # Half of the training examples should have the same label as test input
                idxs = random.sample(test_label_idxs, round(num/2))
                rem_idxs = list(set(test_label_idxs)-set(idxs)) + other_label_idxs
                idxs.extend(random.sample(rem_idxs, num-len(idxs)))
                random.shuffle(idxs)  
            case EntropyLevels.LABELSUPPRESS:
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
        case EntropyLevels.MAX:
            if len(selected_labels)==1: # no "balanced distribution" for 1 shot, it'll be balanced on avg
                return
            label_counts = label_counts_map.values()
            target = int(len(selected_labels) > num_classes) # only when k-shot is more than num_classes there can be 1 extra ex in some class
            assert max(label_counts) - min(label_counts) <= target, f"Max entropy sampling logic seems to be wrong. label counts: {label_counts}, target:{target}"
        case EntropyLevels.LABELSPIKE:
            assert label_counts_map[test_label]>=round(len(selected_labels)/2), "Label ain't spiking"
        case EntropyLevels.LABELSUPPRESS:
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
    else:
        top_idxs = np.flip(top_idxs, axis=-1) # the test sentence stays at the last position instead of 1st
    
    from data_utils import SampledData 
        
    sampled_data = SampledData()
    sampled_data.sentences = [[sentences[j] for j in idx_row] for idx_row in top_idxs]
    sampled_data.labels = [[labels[j] for j in idx_row] for idx_row in top_idxs]

    # if num_shots==3: print(sentences[1], test_sentences[1]); exit()
    if return_embeddings:
        sampled_data.embeddings = np.array([[sentence_embeddings[j] for j in idx_row] for idx_row in top_idxs])
    return sampled_data

def get_similarities(test_embeddings: np.ndarray, sentence_embeddings: np.ndarray):
    test_embeddings = test_embeddings/np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    sentence_embeddings = sentence_embeddings/np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)

    similarities = np.dot(test_embeddings, sentence_embeddings.T)
    return similarities