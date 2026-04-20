from collections import namedtuple
from typing import Dict, List
import numpy as np

from utils.sampling_utils import get_similarities 
from utils.gen_utils import get_llm_framework, js_divergence
from utils.run_utils import get_results, get_embeddings
from labels_trie import LabelsTrie
from llm_framework import LlmFramework

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
    
    # num_classes = len(params['label_dict'])

    probs, logits, lower_dim_embeddings = get_autoregressive_results_with_embeddings(params, sentences, labels) 
        
    preds = np.argmax(probs, axis=-1)
    pred_probs = probs[np.arange(len(preds)), preds]
    # print(preds.shape, probs.shape); exit()
    shifted_features = get_shifted_features(probs, preds, labels)
    
    # Input to the calibration transformer is a concatenation of t<k shot probs, embeddings and other features
    data['inputs'] = [
        np.concatenate((
            [pred_probs[sent_idx]], 
            [shifted_features.correctness[sent_idx]], 
            [shifted_features.gt_probs[sent_idx]], 
            lower_dim_embeddings[sent_idx]
        )).tolist()
        for sent_idx in range(len(sentences))
    ]
    data['logits'] = logits.tolist()

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

def get_autoregressive_results_with_embeddings(params: Dict, sentences: List[str], labels: List[int]):
    single_depth_label_trie = len(LabelsTrie(params['label_dict']).root.children) == len(params['label_dict'])
    in_context_logprobs = single_depth_label_trie and get_llm_framework(params['model'])==LlmFramework.HF
    
    if in_context_logprobs:
        train_sentences = [sentences[:-1]]
        train_labels = [labels[:-1]]
        test_sentences = [sentences[-1]]
        
        llm_result = get_results(params, train_sentences, train_labels, test_sentences, in_context_logprobs=in_context_logprobs)
        logprobs = llm_result.logprobs[0]
        in_context_logprobs = llm_result.in_context_logprobs[0]
        
        probs = logprobs.probs + in_context_logprobs.probs
        logits = logprobs.logits + in_context_logprobs.logits
        sentences = logprobs.hidden_features + in_context_logprobs.hidden_features
        # print(len(probs), len(logprobs.probs), len(in_context_logprobs.probs), len(sentences))
        embedding_model_name = "hidden_features"
    else:
        probs, logits = get_autoregressive_results(params, sentences, labels)
        embedding_model_name = "google/embeddinggemma300m"
        
    lower_dim_embeddings = get_embeddings(params, sentences, embedding_model_name=embedding_model_name)
    
    return np.array(probs), np.array(logits), lower_dim_embeddings
    
def get_autoregressive_results(params: Dict, sentences: List[str], labels: List[int]):
    # LLM's logits(y|x, C) to be calibrated by transformer's output T
    train_sentences = [sentences[:sent_idx] for sent_idx in range(len(sentences))]
    train_labels = [labels[:sent_idx] for sent_idx in range(len(sentences))]
    test_sentences = [sentences[sent_idx] for sent_idx in range(len(sentences))]
    
    llm_result = get_results(params, train_sentences, train_labels, test_sentences, in_context_logprobs=False)
    probs = np.array([llm_result.logprobs[i].probs[0] for i in range(len(test_sentences))])
    logits = np.array([llm_result.logprobs[i].logits[0] for i in range(len(test_sentences))])

    return probs, logits

# def get_autoregressive_results(params: Dict, sentences: List[str], labels: List[int]):
#     # LLM's logits(y|x, C) to be calibrated by transformer's output T
#     train_sentences = [sentences[:sent_idx] for sent_idx in range(len(sentences))]
#     train_labels = [labels[:sent_idx] for sent_idx in range(len(sentences))]
#     test_sentences = [sentences[sent_idx] for sent_idx in range(len(sentences))]
    
#     probs, logits = get_results(params, train_sentences[::-1], train_labels[::-1], test_sentences[::-1])
    
#     return probs[::-1], logits[::-1]

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