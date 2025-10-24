import re
from typing import Dict, List
import json

import datasets
from transformers import AutoTokenizer
from utils import ROOT_DIR
import numpy as np
from labels_trie import LabelsTrie

def load_dataset_with_embeddings(dataset:str, split:str):
    sentences, labels, embeddings = [], [], []

    with open(f"{ROOT_DIR}/data/{dataset}/{split}.json", 'r') as file:
        dataset = json.load(file)

    for item in dataset:
        if item['label']<0:
            continue
        sentences.append(item['sentence'])
        labels.append(item['label'])
        embeddings.append(item['sentence_embedding'])
        
    return sentences, labels, np.array(embeddings, dtype=np.float32)

def set_label_tokens(labels: List[str], params: Dict):
    params['label_dict'] = {}
    tokenizer = AutoTokenizer.from_pretrained(params['model'])
    
    for label_idx, label in enumerate(labels):
        tokens = tokenizer.tokenize(" " + label.lstrip())
        # tokens = [token.replace("Ġ", " ").replace('▁',' ') for token in tokens]
        tokens[0] = " " + tokens[0][1:]
        
        params['label_dict'][label_idx] = {
            "label":label,
            "tokens":tokens   
        }
    # print(params['label_dict'], len(params['label_dict']))
    # print(LabelsTrie(params['label_dict']).print_trie())
    # exit()
        
def set_prompt_params(params: Dict):
    """
    Set prompt construction and dataset related params
    """
    params["base_prompt"] = None
    
    if params['dataset'] == 'sst2':
        labels = ['Negative', 'Positive']
        params['prompt_prefix'] = f"Your task is to classify the sentiment for a given review as {', '.join(labels)}."
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        set_label_tokens(labels, params)

    elif params['dataset'] == 'sst5':
        labels = ['terrible', 'bad', 'neutral', 'good', 'great']
        params['prompt_prefix'] = f"Your task is to classify the sentiment for a given review as one of: {', '.join(labels)}."
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        set_label_tokens(labels, params)

    elif params['dataset'] == 'snli':
        labels = ['entails', 'neutral', 'contradiction']
        params['prompt_prefix'] = f"Your task is to classify a pair of premise and hypothesis as one of: {', '.join(labels)}."
        params["q_prefix"] = ""
        params["a_prefix"] = "Answer: "
        set_label_tokens(labels, params)

    elif params['dataset'] == 'qqp':
        labels = ['different', 'duplicate']
        params['prompt_prefix'] = f"Your task is to classify the pair of questions as one of: {', '.join(labels)}."
        params["q_prefix"] = ""
        params["a_prefix"] = "Answer: "
        set_label_tokens(labels, params)

    elif params['dataset'] == 'metatool':
        # first_word = lambda label: re.match(r'^[A-Z][a-z]+', label).group(0)
        
        with open("data/metatool/big_tool_des.json", 'r') as file:
            tool_desc = json.load(file)
        labels = sorted(list(tool_desc.keys()))
        
        base_prompt = f"Your task is to select which single tool addresses the given query, based on the provided tool list :\n"
        for tool in labels:
            base_prompt += f"\n{tool}: {tool_desc[tool]}\n"
             
        params['prompt_prefix'] = base_prompt + '\n'
        params["q_prefix"] = "Query: "
        params["a_prefix"] = "Tool: "

        # for i, label in enumerate(label_tokens):
        #     try:
        #         first_word(label)
        #     except:
        #         print(label, "ERRORED OUT LOADING LABELS")
        #         exit()
                
        # params['label_dict'] = {i:label for i, label in enumerate(label_tokens)}
        # params['prompt_label_dict'] = {i:label for i, label in enumerate(label_tokens)}
        set_label_tokens(labels, params)

    elif params['dataset']=='banking77':
        dataset = datasets.load_dataset('mteb/banking77')
        labels = sorted(dataset["train"].unique("label_text"))
        # labels = [label.replace("_"," ") for label in labels]
        
        labels_listed = ""
        for i, label in enumerate(labels):
            labels_listed += f"{i+1}. {label}\n"
            
        params['prompt_prefix'] = "Your task is to classify the given banking query as one of the following:\n\n" + labels_listed
        params["q_prefix"] = "Query: "
        params["a_prefix"] = "Type: "
        set_label_tokens(labels, params)
        
    elif params['dataset']=='wildguard':
        train_dataset = datasets.load_dataset("allenai/wildguardmix", "wildguardtrain")['train']
        labels = sorted(train_dataset.unique("subcategory")) # classes
                
        labels_listed = ""
        for i, label in enumerate(labels):
            labels_listed += f"{i+1}. {label}\n"
            
        params['prompt_prefix'] = "Your task is to classify the user prompt(for safety) as one of the following categories:\n\n" + labels_listed
        params["q_prefix"] = "User Prompt: "
        params["a_prefix"] = "Category: "
        set_label_tokens(labels, params)
        
    elif params['dataset'] == 'when2call':
        labels = ['tool_call', 'request_for_info', 'cannot_answer']
    
        params['prompt_prefix'] = ("Your task is to classify whether the given user question(request/query) can be answered " 
                                   f"by calling one of the provided tools({labels[0]}), or if you need to request for more info to anwer the question({labels[1]}), "
                                   f"or if the question cannot be answered with the provided set of tools({labels[2]}). Your response should be one of: {', '.join(labels)}")
        params["q_prefix"] = ""
        params["a_prefix"] = "Answer: "
        set_label_tokens(labels, params)
        
    elif params['dataset'] == 'toxic_chat':
        labels = ['benign', 'toxic']
                
        params['prompt_prefix'] = f"Your task is to classify the user prompt on toxicity as one of the following categories: {', '.join(labels)}"
        params["q_prefix"] = "User Prompt: "
        params["a_prefix"] = "Toxicity: "
        set_label_tokens(labels, params)
            
    elif params['dataset'] == 'snips':
        train_dataset = datasets.load_dataset("benayas/snips")['test']
        labels = sorted(train_dataset.unique("category")) # classes
                
        labels_listed = ""
        for i, label in enumerate(labels):
            labels_listed += f"{i+1}. {label}\n"
            
        params['prompt_prefix'] = "Your task is to classify the given user query(to a voice assistant) as one of the following intents:\n\n" + labels_listed
        params["q_prefix"] = "Query: "
        params["a_prefix"] = "Intent: "
        set_label_tokens(labels, params)
    
    elif params['dataset'] == 'massive_intent':
        train_dataset = datasets.load_dataset('mteb/amazon_massive_intent', "en")['train']
        labels = sorted(train_dataset.unique("label_text")) # classes
                
        labels_listed = ""
        for i, label in enumerate(labels):
            labels_listed += f"{i+1}. {label}\n"
            
        params['prompt_prefix'] = "Your task is to classify the given user query(to an assistant) as one of the following intents:\n\n" + labels_listed
        params["q_prefix"] = "Query: "
        params["a_prefix"] = "Intent: "
        set_label_tokens(labels, params)
                
    elif params['dataset'] == 'amazon_counterfactual':
        labels = ['not-counterfactual', 'counterfactual']
                
        params['prompt_prefix'] = f"Your task is to classify the given Amazon customer review as one of the following: {', '.join(labels)}"
        params["q_prefix"] = "User Prompt: "
        params["a_prefix"] = "Category: "
        set_label_tokens(labels, params)
    
    elif params['dataset'] == 'newsgroups':
        train_dataset = datasets.load_dataset('SetFit/20_newsgroups')['train']
        labels = sorted(train_dataset.unique("label_text")) # classes
                
        labels_listed = ""
        for i, label in enumerate(labels):
            labels_listed += f"{i+1}. {label}\n"
            
        params['prompt_prefix'] = "Your task is to classify the given newsgroup post as one of the following topics:\n\n" + labels_listed
        params["q_prefix"] = "Post: "
        params["a_prefix"] = "Topic: "
        set_label_tokens(labels, params)
        
    elif params['dataset'].startswith('dbpedia'):
        level = int(params['dataset'][-1])
        assert params['dataset'] in ('dbpedia_l1', 'dbpedia_l2'), "only dbpedia_l1 and dbpedia_l2 datasets names allowed (9/20 classes)"
        
        dataset = datasets.load_dataset('DeveloperOats/DBPedia_Classes')
        labels = sorted(dataset['train'].unique(f'l{level}')) # classes
        
        labels_listed = ""
        for i, label in enumerate(labels):
            labels_listed += f"{i+1}. {label}\n"
        
        params['prompt_prefix'] = f"Your task is to classify the given Wikipedia article excerpt as one of the following categories:\n\n" + labels_listed
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Category: "
        set_label_tokens(labels, params)

    elif params['dataset'] == 'wikitoxic':
        labels = ['benign', 'toxic']
        
        params['prompt_prefix'] = f"Your task is to classify the given Wikipedia comment as one of the following: {', '.join(labels)}"
        params["q_prefix"] = "Comment: "
        params["a_prefix"] = "Toxicity: "
        set_label_tokens(labels, params)

    elif params['dataset'] == 'goemotions':
        dataset = datasets.load_dataset('mrm8488/goemotions')['train']
        feature_names = list(dataset.features.keys())
        labels = [col for col in feature_names if col not in 
                    [ 'text',
                        'id',
                        'author',
                        'subreddit',
                        'link_id',
                        'parent_id',
                        'created_utc',
                        'rater_id',
                        'example_very_unclear']
        ]
        
        labels_listed = ""
        for i, label in enumerate(labels):
            labels_listed += f"{i+1}. {label}\n"
        
        params['prompt_prefix'] = "Your task is to classify the emotion expressed in the given reddit comment as one of the following:\n\n" + labels_listed
        params["q_prefix"] = "Comment: "
        params["a_prefix"] = "Answer: "
        set_label_tokens(labels, params)
           
    elif params['dataset'] == 'strategy_qa':
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {1: 'True', 0: 'False'}

    elif params['dataset'] == 'worldtree':
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    elif params['dataset'] == 'open_book_qa':
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    elif params['dataset'] == 'commonsense_qa':
        labels= ['A', 'B', 'C', 'D', 'E']
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer: "
        set_label_tokens(labels, params)

    elif params['dataset'] == 'agnews':
        labels = ['World', 'Sports', 'Business', 'Technology']
        params['prompt_prefix'] = f"Classify the news article as one from the categories: {', '.join(labels)}.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Category: "
        set_label_tokens(labels, params)

    elif params['dataset'] == 'trec':
        labels = ['Number', 'Location', 'Person', 'Description', 'Entity', 'Abbreviation']
        params['prompt_prefix'] = f"Classify the questions based on whether their answer type is one of: {', '.join(labels)}.\n\n"
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer Type: "
        set_label_tokens(labels, params)

    elif params['dataset'] == 'rte':
        labels = ['False', 'True']
        params['prompt_prefix'] = ""
        params["q_prefix"] = " "
        params["a_prefix"] = "answer: "
        set_label_tokens(labels, params)

    elif params['dataset'] == 'cb':
        params['prompt_prefix'] = ""
        params["q_prefix"] = ""
        params["a_prefix"] = "answer: "
        params['label_dict'] = {0: 'false', 1: 'neither', 2: 'true'}

    elif params['dataset'] == 'dbpedia':
        params['prompt_prefix'] = "Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {
            0: 'Company', 1: 'School', 2: 'Artist', 3: 'Ath', 4: 'Polit',
            5: 'Transportation', 6: 'Building', 7: 'Nature', 8: 'Village',
            9: 'Animal', 10: 'Plant', 11: 'Album', 12: 'Film', 13: 'Book'
        }

    else:
        raise NotImplementedError
