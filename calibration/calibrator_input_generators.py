from collections import namedtuple
from typing import Dict, List
import numpy as np
from utils import get_similarities, get_results, get_embeddings, js_divergence

def generate_data_non_causal(params: Dict, sentences: List[str], embeddings: np.ndarray, labels: List[int]):
    data = {
        "inputs" : [],
        "logits": [],
        "labels": labels
    }
    zero_shot_probs = []
    # print(embeddings.shape, len(sentences), len(labels))
    # exit()
    similarity_vectors = np.tril(get_similarities(embeddings, embeddings)) # only upper triangular to make it causal

    # LLM's zero shot probs that will part of the input vector to the transformer at each token position
    zero_shot_probs, zero_shot_raw_logits = get_results(params, [[]]*len(sentences), [[]]*len(sentences), sentences)

    # Input to the calibration transformer is a concatenationo zer shot probs and the similarities
    # print(zero_shot_probs[1], similarity_vectors[1])    
    data['inputs'] = [np.concat((zero_shot_probs[sent_idx], similarity_vectors[sent_idx])).tolist() for sent_idx in range(len(sentences))]

    # LLM's logits(y|x, C) to be calibrated by transformer's output T
    train_sentences = [sentences[:sent_idx] for sent_idx in range(len(sentences))]
    train_labels = [labels[:sent_idx] for sent_idx in range(len(sentences))]
    test_sentences = [sentences[sent_idx] for sent_idx in range(len(sentences))]
    
    all_label_probs, all_label_raw_logits = get_results(params, train_sentences, train_labels, test_sentences)
    data['logits'] = all_label_raw_logits.tolist()
        
    return data

# Different feature set, made for causal temperature regression
def generate_data(params: Dict, sentences: List[str], embeddings: np.ndarray, labels: List[int]):
    data = {
        "inputs" : [],
        "logits": [],
        "labels": labels
    }
    
    num_classes = len(params['label_dict'])
    shifted_onehot_labels = np.zeros((len(labels), num_classes)) # T,num_classes
    shifted_onehot_labels[0] = np.array([1/num_classes]*num_classes)
    
    for i in range(len(labels)-1):
        shifted_onehot_labels[i+1][labels[i]] = 1
    
    similarity_vectors = np.tril(get_similarities(embeddings, embeddings)) # only lower triangular to make it causal
    
    # LLM's logits(y|x, C) to be calibrated by transformer's output T
    train_sentences = [sentences[:sent_idx] for sent_idx in range(len(sentences))]
    train_labels = [labels[:sent_idx] for sent_idx in range(len(sentences))]
    test_sentences = [sentences[sent_idx] for sent_idx in range(len(sentences))]
    
    all_label_probs, all_label_raw_logits = get_results(params, train_sentences, train_labels, test_sentences)
    # Input to the calibration transformer is a concatenation of t<k shot probs and the similarities
    data['inputs'] = [np.concat((all_label_probs[sent_idx], shifted_onehot_labels[sent_idx], similarity_vectors[sent_idx])).tolist() for sent_idx in range(len(sentences))]
    data['logits'] = all_label_raw_logits.tolist()

    # print(data['inputs'])
    # exit()
    return data

# Different feature set, made for causal temperature regression | added input embeddings
def generate_data_with_embeddings(params: Dict, sentences: List[str], embeddings: np.ndarray, labels: List[int]):
    data = {
        "inputs" : [],
        "logits": [],
        "labels": labels
    }
    
    num_classes = len(params['label_dict'])
    shifted_onehot_labels = np.zeros((len(labels), num_classes)) # T,num_classes
    shifted_onehot_labels[0] = np.array([1/num_classes]*num_classes)
    
    for i in range(len(labels)-1):
        shifted_onehot_labels[i+1][labels[i]] = 1
    
    # LLM's logits(y|x, C) to be calibrated by transformer's output T
    train_sentences = [sentences[:sent_idx] for sent_idx in range(len(sentences))]
    train_labels = [labels[:sent_idx] for sent_idx in range(len(sentences))]
    test_sentences = [sentences[sent_idx] for sent_idx in range(len(sentences))]
    
    all_label_probs, all_label_raw_logits = get_results(params, train_sentences, train_labels, test_sentences)
    # Input to the calibration transformer is a concatenation of t<k shot probs and the similarities
    data['inputs'] = [np.concat((all_label_probs[sent_idx], shifted_onehot_labels[sent_idx], embeddings[sent_idx])).tolist() for sent_idx in range(len(sentences))]
    data['logits'] = all_label_raw_logits.tolist()

    # print(np.array(data['inputs']).shape)
    # exit()
    return data

# Different feature set, made for causal temperature regression | class agnostic
def generate_data_class_agnostic(params: Dict, sentences: List[str], embeddings: np.ndarray, labels: List[int]):
    data = {
        "inputs" : [],
        "logits": [],
        "labels": labels
    }
    
    num_classes = len(params['label_dict'])

    all_label_probs, all_label_raw_logits = get_autoregressive_results(params, sentences, labels)
        
    row_sums = np.sum(all_label_probs, axis=-1, keepdims=True)
    if np.any(row_sums == 0):
        print("Zero probs in here! :", all_label_probs)
        print(sentences)
        print(labels)      
        
    probs = all_label_probs/row_sums
    preds = np.argmax(probs, axis=-1)
    pred_probs = probs[np.arange(len(preds)), preds]
    
    shifted_features = get_shifted_features(probs, preds, labels)

    normalized_entropies = -(probs@np.log(probs.T + 1e-9)).diagonal() / np.log(num_classes) # (B, num_classes) × (num_classes, B) --> (B, B) --> diagonal elements

    similarity_vectors = np.tril(get_similarities(embeddings, embeddings)) # only lower triangular to make it causal
    
    # Input to the calibration transformer is a concatenation of t<k shot probs, similarities and other features
    data['inputs'] = [
        np.concatenate((
            [pred_probs[sent_idx]], 
            [shifted_features.correctness[sent_idx]], 
            [shifted_features.gt_probs[sent_idx]], 
            [normalized_entropies[sent_idx]], 
            similarity_vectors[sent_idx]
        )).tolist()
        for sent_idx in range(len(sentences))
    ]
    data['logits'] = all_label_raw_logits.tolist()

    # with np.printoptions(precision=3, suppress=True):
        # print(np.array(data['inputs']))
        # print(np.array(data['logits']))
    # print(probs, sentences, labels)
    # exit()
    return data

def generate_data_class_agnostic_with_embeddings(params: Dict, sentences: List[str], embeddings: np.ndarray, labels: List[int]):
    data = {
        "inputs" : [],
        "logits": [],
        "labels": labels
    }
    
    num_classes = len(params['label_dict'])

    all_label_probs, all_label_raw_logits = get_autoregressive_results(params, sentences, labels)
        
    row_sums = np.sum(all_label_probs, axis=-1, keepdims=True)
    if np.any(row_sums == 0):
        print("Zero probs in here! :", all_label_probs)
        print(sentences)
        print(labels)      
        
    probs = all_label_probs/row_sums
    preds = np.argmax(probs, axis=-1)
    pred_probs = probs[np.arange(len(preds)), preds]
    
    shifted_features = get_shifted_features(probs, preds, labels)

    normalized_entropies = -(probs@np.log(probs.T + 1e-9)).diagonal() / np.log(num_classes) # (B, num_classes) × (num_classes, B) --> (B, B) --> diagonal elements

    input_similarity_vectors = np.tril(get_similarities(embeddings, embeddings)) # only lower triangular to make it causal
    
    lower_dim_embeddings = get_embeddings(params, sentences)
    
    T = len(sentences) # number of timesteps
    pred_similarity_vectors = np.zeros((T, T))
    for timestep in range(T):
        for prev_timestep in range(timestep + 1):  # only compute for j <= i (causal)
            js_div = js_divergence(probs[timestep], probs[prev_timestep])
            pred_similarity_vectors[timestep, prev_timestep] = 1 - (js_div / np.log(2))  # Maps [0, log(2)] → [1, 0]

    # Input to the calibration transformer is a concatenation of t<k shot probs, similarities and other features
    data['inputs'] = [
        np.concatenate((
            [pred_probs[sent_idx]], 
            [shifted_features.correctness[sent_idx]], 
            [shifted_features.gt_probs[sent_idx]], 
            [normalized_entropies[sent_idx]],
            pred_similarity_vectors[sent_idx], 
            input_similarity_vectors[sent_idx],
            lower_dim_embeddings[sent_idx]
        )).tolist()
        for sent_idx in range(len(sentences))
    ]
    data['logits'] = all_label_raw_logits.tolist()

    # with np.printoptions(precision=3, suppress=True):
    #     print(np.array(data['inputs'])[:, :4+2*T])
    #     # print(np.array(data['logits']))
    # for sent_idx in range(len(sentences)): 
    #     print("Sentence: ", sentences[sent_idx])
    #     print("Label: ", labels[sent_idx])
    #     print("Probs: ", probs[sent_idx],'\n')
    # print(np.array(similarity_vectors[-1]))
    # exit()
    return data

def get_autoregressive_results(params: Dict, sentences: List[str], labels: List[int]):
    # LLM's logits(y|x, C) to be calibrated by transformer's output T
    train_sentences = [sentences[:sent_idx] for sent_idx in range(len(sentences))]
    train_labels = [labels[:sent_idx] for sent_idx in range(len(sentences))]
    test_sentences = [sentences[sent_idx] for sent_idx in range(len(sentences))]
    
    probs, logits = get_results(params, train_sentences, train_labels, test_sentences)
    
    return probs, logits

def get_shifted_features(probs, preds, labels):
    shifted_gt_probs = np.concatenate((
        np.array([0.5]), 
        probs[np.arange(len(preds)-1), labels[:-1]]
    )) # exclude last position and shift
    
    shifted_correctness = np.zeros((len(labels),)) # num_classes, 
    shifted_correctness[0] = 0.5
    
    for i in range(len(labels)-1):
        shifted_correctness[i+1] = int(preds[i]==labels[i])
        
    shifted_features = namedtuple('shifted_features', ['gt_probs', 'correctness'])
    
    return shifted_features(shifted_gt_probs, shifted_correctness)