from typing import List
import pandas as pd
import datasets
from datasets import DatasetDict, concatenate_datasets
import random
import json
import numpy as np
from sklearn.model_selection import train_test_split
from utils import ROOT_DIR

path_tests = {'strategy_qa' : 'data/strategy_qa/strategyqa_train.json', 'commonsense_qa': 'data/commonsense_qa/dev_rand_split.json', 'open_book_qa': 'data/open_book_qa/test.jsonl', 'worldtree': 'data/worldtree/test.json'}

def load_sst2():
    def process_raw_data_sst(lines):
        """from lines in dataset to two lists of sentences and labels respectively"""
        labels = []
        sentences = []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(line[2:].strip())
        return sentences, labels

    with open(f"{ROOT_DIR}/data/sst2/stsa.binary.train", "r") as f:
        train_lines = f.readlines()
    with open(f"{ROOT_DIR}/data/sst2/stsa.binary.test", "r") as f:
        test_lines = f.readlines()
    train_sentences, train_labels = process_raw_data_sst(train_lines)
    test_sentences, test_labels = process_raw_data_sst(test_lines)
    return train_sentences, train_labels, test_sentences, test_labels

def load_commonsense_qa_old(name='commonsense_qa', label_dict={'A': 0, 'B': 1, 'C':2, 'D':3, 'E':4}):
    path = 'data/thoughtsource_100.json'
    path_test = path_tests[name]
    train_sentences, train_labels, test_sentences, test_labels = [], [], [], []
    with open(path, 'r') as file:
        file = file.read()
        json_data = json.loads(file)

        if name in ['open_book_qa', 'worldtree']:
            data = json_data[name]['test']
        else:
            data = json_data[name]['validation']        

        for i, json_data in enumerate(data):
            question = json_data['question']
            choice = json_data['choices']
            answer = json_data['answer']
            label = choice.index(answer[0])
            candidate = ''
            for k in range(len(label_dict)):
                candidate += '\n' + label_dict[k] + ' ' + choice[k] 
            train_labels.append(label) #label_dict[label]
            train_sentences.append(question + ' ' + candidate)
    with open(path_test, 'r') as file:
        # file = file.read()
        # data = json.loads(file)
        if path_test.endswith('.jsonl'):
            data = [json.loads(line) for line in file]
        elif path_test.endswith('.json'):
            data = json.load(file)
        for json_data in data:
            question = json_data['question']
            answer = json_data['answerKey']

            inv_label_dict = {'A': 0, 'B': 1, 'C':2, 'D':3, 'E':4}
            if answer not in inv_label_dict.keys():
                continue

            test_labels.append(inv_label_dict[answer])

            choice = question['choices']
            stem = question['stem'] 
            candidate = ''

            if 'worldtree' in path_test:
                candidate = ''
                for l in inv_label_dict.keys():
                    pre_l = '(' + l + ')'
                    stem = stem.replace(pre_l, '\n'+l)
            else:
                for k in range(len(choice)):
                    l, t = choice[k]['label'], choice[k]['text']
                    candidate += '\n' + l + ' ' + t
            test_sentences.append(stem + candidate)

    return train_sentences, train_labels, test_sentences, test_labels 

def load_commonsense_qa(inv_label_dict):
    dataset = datasets.load_dataset("tau/commonsense_qa")
    dataset.pop('test') # has no labels, not useful 
    
    def format_example(example):
        prompt = example['question']

        for label, text in zip(example['choices']['label'], example['choices']['text']):
            prompt += '\n' + label + ' ' + text    # Ex: '\nA bank'

        example['text'] = prompt
        example['answerLabel'] = inv_label_dict[example['answerKey']]

        return example 

    dataset = dataset.map(format_example)
    train_sentences = list(dataset['train']['text'])
    train_labels = list(dataset['train']['answerLabel'])
    test_sentences = list(dataset['validation']['text'])
    test_labels = list(dataset['validation']['answerLabel'])

    return train_sentences, train_labels, test_sentences, test_labels

def load_strategy_qa(name='strategy_qa', num=100, ratio=0.25, label_dict={'True': 1, 'False': 0}):
    path_cot = 'data/thoughtsource_100.json'
    path_test = path_tests[name]
    train_sentences, train_labels, test_sentences, test_labels = [], [], [], []
    with open(path_cot, 'r') as file:
        file = file.read()
        json_data = json.loads(file)
        data = json_data[name]['train']
        num = min(num, len(data))
        for i, json_data in enumerate(data[:num]):
            question = json_data['question']
            train_labels.append(label_dict[json_data['answer'][0]])
            train_sentences.append(question)

    with open(path_test, 'r') as file:
        file = file.read()
        data = json.loads(file)
        num = min(num, len(data))

        for i, json_data in enumerate(data):
            question = json_data['question']
            test_labels.append(int(json_data['answer']))
            test_sentences.append(question)

    return train_sentences, train_labels, test_sentences, test_labels 

def load_agnews():
    train_data = pd.read_csv(f'{ROOT_DIR}/data/agnews/train.csv')
    test_data = pd.read_csv(f'{ROOT_DIR}/data/agnews/test.csv')

    train_sentences = train_data['Title'] + ". " + train_data['Description']
    train_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in train_sentences]) # some basic cleaning
    train_labels = list(train_data['Class Index'])
    test_sentences = test_data['Title'] + ". " + test_data['Description']
    test_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in test_sentences]) # some basic cleaning
    test_labels = list(test_data['Class Index']) 
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4
    test_labels = [l - 1 for l in test_labels]
    return train_sentences, train_labels, test_sentences, test_labels

def load_trec():
    inv_label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}
    train_sentences = []
    train_labels = []
    with open(f'{ROOT_DIR}/data/trec/train.txt', 'r') as train_data:
        for line in train_data:
            train_label = line.split(' ')[0].split(':')[0]
            train_label = inv_label_dict[train_label]
            train_sentence = ' '.join(line.split(' ')[1:]).strip()
            # basic cleaning
            train_sentence = train_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            train_labels.append(train_label)
            train_sentences.append(train_sentence)

    test_sentences = []
    test_labels = []
    with open(f'{ROOT_DIR}/data/trec/test.txt', 'r') as test_data:
        for line in test_data:
            test_label = line.split(' ')[0].split(':')[0]
            test_label = inv_label_dict[test_label]
            test_sentence = ' '.join(line.split(' ')[1:]).strip()
            test_sentence = test_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            test_labels.append(test_label)
            test_sentences.append(test_sentence)
    return train_sentences, train_labels, test_sentences, test_labels

def get_cb():
    train_questions = []
    train_answers = []
    with open(f"{ROOT_DIR}/data/cb/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            curr_label = myjson['label']
            if curr_label == 'contradiction':
                train_answers.append(0)
            elif curr_label == 'neutral':
                train_answers.append(1)
            elif curr_label == 'entailment':
                train_answers.append(2)
            # being a bit lazy here. We put the "question: " into the input and treat it like single sentence classification.
            train_questions.append(p.strip() + '\n' + 'question: ' + q + '. true, false, or neither?')

    test_questions = []
    test_answers = []
    with open(f"{ROOT_DIR}/data/cb/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'contradiction':
                test_answers.append(0)
            elif myjson['label'] == 'neutral':
                test_answers.append(1)
            elif myjson['label'] == 'entailment':
                test_answers.append(2)
            else:
                exit('answer')
            test_questions.append(p.strip() + '\n' + 'question: ' + q + '. true, false, or neither?')

    return train_questions, train_answers, test_questions, test_answers

def load_rte():
    train_questions = []
    train_answers = []
    with open("data/rte/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                train_answers.append(0)
            elif myjson['label'] == 'entailment':
                train_answers.append(1)
            else:
                exit('answer')
            train_questions.append(p + '\n' + 'Question: ' + q + ' True or False?')

    test_questions = []
    test_answers = []
    with open("data/rte/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                test_answers.append(0)
            elif myjson['label'] == 'entailment':
                test_answers.append(1)
            else:
                exit('answer')
            test_questions.append(p + '\n' + 'Question: ' + q + ' True or False?')

    return train_questions, train_answers, test_questions, test_answers

def load_snli():
    dataset = datasets.load_dataset('stanfordnlp/snli')
    dataset['train'] = dataset['train'].filter(lambda row: row['label']!=-1).shuffle(seed=42).select(range(10**5)) # 100k examples
    dataset['test'] = dataset['test'].filter(lambda row: row['label']!=-1)

    def format_example(example):
        example['text'] = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
        return example
    dataset = dataset.map(format_example)
    train_sentences = list(dataset['train']['text'])
    train_labels = list(dataset['train']['label'])
    test_sentences = list(dataset['test']['text'])
    test_labels = list(dataset['test']['label'])
    return train_sentences, train_labels, test_sentences, test_labels

def load_sst5():
    dataset = datasets.load_dataset('SetFit/sst5')

    train_sentences = list(dataset['train']['text'])
    train_labels = list(dataset['train']['label'])
    test_sentences = list(dataset['test']['text'])
    test_labels = list(dataset['test']['label'])
    return train_sentences, train_labels, test_sentences, test_labels

def load_qqp():
    dataset = datasets.load_dataset("SetFit/qqp")
    dataset.pop('test') # has no labels, not useful 
    dataset['train'] = dataset['train'].select(range(10**5)) # first 100k rows
    dataset['validation'] = dataset['validation'].select(range(2*10**4)) # first 20k rows
    
    def format_example(example):
        example['text'] = f"1. {example['text1']}\n2. {example['text2']}"

        return example 

    dataset = dataset.map(format_example)
    train_sentences = list(dataset['train']['text'])
    train_labels = list(dataset['train']['label'])
    test_sentences = list(dataset['validation']['text'])
    test_labels = list(dataset['validation']['label'])
    
    return train_sentences, train_labels, test_sentences, test_labels
    
def load_metatool():
    with open("data/metatool/big_tool_des.json", 'r') as file:
        tool_desc = json.load(file)    # has renamed tools
    with open("data/metatool/tool_rename_map.json", 'r') as file:
        tool_rename_map = json.load(file)
        unchanged_names = set(tool_desc.keys())-set(tool_rename_map.values())
        for tool in unchanged_names:
            tool_rename_map[tool] = tool
        
    # print(tool_rename_map, len(tool_rename_map)); exit()
    # assert set(tool_desc.keys())==set(tool_rename_map.values())
    label_map = {tool:i for i, tool in enumerate( sorted(tool_desc.keys()) )} 
    
    df = pd.read_csv("data/metatool/single_tool.csv")
    df = df[df["Tool"].isin(tool_rename_map.keys())]
    df["Tool"] = df["Tool"].map(lambda tool: label_map[tool_rename_map[tool]]) # rename tool and then map to the label idxs
    
    X = df['Query'].to_list()
    y = df['Tool'].to_list()
    
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=1) # fixed random_state to ensure reproducible shuffling. Avoids data leakage in case this dataset is used for training 
    # print(len(train_sentences))
    return train_sentences, train_labels, test_sentences, test_labels
    
def load_banking77():
    dataset = datasets.load_dataset('mteb/banking77')
    labels = sorted(dataset["train"].unique("label_text"))
    label_idx_map = {label:idx for idx, label in enumerate(labels)}
    
    def format_example(example):
        example['updated_label'] = label_idx_map[example['label_text']]
        return example 

    dataset = dataset.map(format_example)
    
    train_sentences = list(dataset['train']['text'])
    train_labels = list(dataset['train']['updated_label'])
    test_sentences = list(dataset['test']['text'])
    test_labels = list(dataset['test']['updated_label'])
    
    return train_sentences, train_labels, test_sentences, test_labels
    
def load_wildguard():
    train_dataset = datasets.load_dataset("allenai/wildguardmix", "wildguardtrain")['train'].shuffle(seed=42)
    test_dataset = datasets.load_dataset("allenai/wildguardmix", "wildguardtest")['test'].shuffle(seed=42)

    train_dataset = train_dataset.select(range(50000))
    
    labels = sorted(train_dataset.unique("subcategory")) # classes
    label_idx_map = {label:idx for idx, label in enumerate(labels)}

    def format_example(example):
        example['label'] = label_idx_map[example['subcategory']] # label idxs
        return example 

    train_dataset = train_dataset.map(format_example)
    test_dataset = test_dataset.map(format_example)
    # print(label_idx_map); print(train_dataset[0]); exit()

    train_sentences = list(train_dataset['prompt'])
    train_labels = list(train_dataset['label'])
    test_sentences = list(test_dataset['prompt'])
    test_labels = list(test_dataset['label'])
    
    return train_sentences, train_labels, test_sentences, test_labels

def load_when2call():
    dataset = datasets.load_dataset('nvidia/When2Call')['mcq'].shuffle(seed=42)
    train_size = int(len(dataset) * 0.8)

    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))

    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    label_idx_map = {'tool_call':0, 'request_for_info':1, 'cannot_answer':2}
    
    
    def format_tools(tools: List[str]):
        formatted_tools = []
        
        for tool in tools:
            tool = json.loads(tool)

            for prop_key, prop_value in tool['parameters']['properties'].items():
                del prop_value['description']
            
            formatted_tools.append(tool)
        
        return formatted_tools
    
    def format_example(example):
        example['updated_label'] = label_idx_map[example['correct_answer']]
        
        text = f"Question: {example['question']}\nTools: "
        formatted_tools = format_tools(example['tools'])
        text += json.dumps(formatted_tools, indent=4)
        
        example['text'] = text
        return example 

    dataset = dataset.map(format_example)
    
    train_sentences = list(dataset['train']['text'])
    train_labels = list(dataset['train']['updated_label'])
    test_sentences = list(dataset['test']['text'])
    test_labels = list(dataset['test']['updated_label'])
    print(train_sentences[0], train_labels[0])

    return train_sentences, train_labels, test_sentences, test_labels

def load_toxic_chat():
    dataset = datasets.load_dataset('lmsys/toxic-chat', 'toxicchat0124').shuffle(seed=42)
       
    for split in dataset.keys():
        data_split = dataset[split]
        
        toxic = data_split.filter(lambda row: row['toxicity'] == 1)
        non_toxic = data_split.filter(lambda row: row['toxicity'] == 0)
        
        n_non_toxic = len(non_toxic) // 3
        non_toxic = non_toxic.shuffle(seed=42).select(range(n_non_toxic))
        
        dataset[split] = concatenate_datasets([toxic, non_toxic])
        
    train_sentences = list(dataset['train']['user_input'])
    train_labels = list(dataset['train']['toxicity'])
    test_sentences = list(dataset['test']['user_input'])
    test_labels = list(dataset['test']['toxicity'])
    
    return train_sentences, train_labels, test_sentences, test_labels    

def load_snips():
    dataset = datasets.load_dataset('benayas/snips').shuffle(seed=42)
    
    labels = sorted(dataset['train'].unique("category")) # classes
    label_idx_map = {label:idx for idx, label in enumerate(labels)}

    def format_example(example):
        example['label'] = label_idx_map[example['category']] # label idxs
        return example 

    dataset = dataset.map(format_example)
    
    train_sentences = list(dataset['train']['text'])
    train_labels = list(dataset['train']['label'])
    test_sentences = list(dataset['test']['text'])
    test_labels = list(dataset['test']['label'])
    
    return train_sentences, train_labels, test_sentences, test_labels  

def load_massive_intent():
    dataset = datasets.load_dataset('mteb/amazon_massive_intent', "en").shuffle(seed=42)
    
    labels = sorted(dataset['train'].unique("label_text")) # classes
    label_idx_map = {label:idx for idx, label in enumerate(labels)}

    def format_example(example):
        example['label'] = label_idx_map[example['label_text']] # label idxs
        return example 

    dataset = dataset.map(format_example)
    
    train_sentences = list(dataset['train']['text'])
    train_labels = list(dataset['train']['label'])
    test_sentences = list(dataset['test']['text'])
    test_labels = list(dataset['test']['label'])
    
    return train_sentences, train_labels, test_sentences, test_labels  

def load_amazon_counterfactual():
    dataset = datasets.load_dataset('mteb/amazon_counterfactual', "en-ext").shuffle(seed=42)
    
    train_sentences = list(dataset['train']['text'])
    train_labels = list(dataset['train']['label'])
    test_sentences = list(dataset['test']['text'])
    test_labels = list(dataset['test']['label'])
    
    return train_sentences, train_labels, test_sentences, test_labels  

def load_newsgroups():
    dataset = datasets.load_dataset('SetFit/20_newsgroups').shuffle(seed=42)
    dataset = dataset.filter(lambda example: len(example['text'])<3000) # some examples are tens of thousads of chars long
    
    labels = sorted(dataset['train'].unique("label_text")) # classes
    label_idx_map = {label:idx for idx, label in enumerate(labels)}

    def format_example(example):
        example['label'] = label_idx_map[example['label_text']] # label idxs
        return example 

    dataset = dataset.map(format_example)
        
    train_sentences = list(dataset['train']['text'])
    train_labels = list(dataset['train']['label'])
    test_sentences = list(dataset['test']['text'])
    test_labels = list(dataset['test']['label'])
    
    return train_sentences, train_labels, test_sentences, test_labels  

def load_dbpedia(level: int):
    dataset = datasets.load_dataset('DeveloperOats/DBPedia_Classes').shuffle(seed=42)
    dataset = dataset.filter(lambda example: len(example)<3000) # some examples are too long

    label_text = f"l{level}"
    labels = sorted(dataset['train'].unique(label_text)) # classes
    label_idx_map = {label:idx for idx, label in enumerate(labels)}

    def format_example(example):
        example['label'] = label_idx_map[example[label_text]] # label idxs
        return example 

    dataset = dataset.map(format_example)
        
    train_sentences = list(dataset['train']['text'])
    train_labels = list(dataset['train']['label'])
    test_sentences = list(dataset['test']['text'])
    test_labels = list(dataset['test']['label'])
    
    return train_sentences, train_labels, test_sentences, test_labels

def load_wikitoxic():
    dataset = datasets.load_dataset(
        "csv",
        data_files={
            "train": "hf://datasets/OxAISH-AL-LLM/wiki_toxic/train.csv",
            "validation": "hf://datasets/OxAISH-AL-LLM/wiki_toxic/validation.csv",
            "test": "hf://datasets/OxAISH-AL-LLM/wiki_toxic/test.csv"
        }
    ).shuffle(seed=42)
    
    for split in dataset.keys():
        data_split = dataset[split]
        
        toxic = data_split.filter(lambda row: row['label'] == 1)
        non_toxic = data_split.filter(lambda row: row['label'] == 0)
        
        n_non_toxic = len(non_toxic) // 3
        non_toxic = non_toxic.shuffle(seed=42).select(range(n_non_toxic))
        
        dataset[split] = concatenate_datasets([toxic, non_toxic])
        
    train_sentences = list(dataset['train']['comment_text'])
    train_labels = list(dataset['train']['label'])
    test_sentences = list(dataset['test']['comment_text'])
    test_labels = list(dataset['test']['label'])
    
    return train_sentences, train_labels, test_sentences, test_labels  
    
def load_goemotions():
    dataset = datasets.load_dataset('mrm8488/goemotions')
    feature_names = list(dataset['train'].features.keys())
    labels = [col for col in feature_names if col not in 
              ['text', 'id', 'author', 'subreddit', 'link_id', 
               'parent_id', 'created_utc', 'rater_id', 'example_very_unclear']
    ]
    label_idx_map = {label: idx for idx, label in enumerate(labels)}
    
    # Convert to pandas for easier aggregation
    df = dataset['train'].to_pandas()
    
    # Aggregate by text - sum up votes for each label across all raters
    agg_dict = {'text': 'first'}  # Keep the text
    agg_dict.update({label: 'sum' for label in labels})  # Sum votes for each label
    
    df_agg = df.groupby('text', as_index=False).agg(agg_dict)
    
    # Keep only texts where exactly one label has the most votes (clear winner)
    def has_single_winner(row):
        label_counts = [row[label] for label in labels]
        max_count = max(label_counts)
        return label_counts.count(max_count) == 1 and max_count > 0
    
    df_agg = df_agg[df_agg.apply(has_single_winner, axis=1)]
    
    # Convert back to dataset
    dataset = datasets.Dataset.from_pandas(df_agg)
    
    # Shuffle and select
    dataset = dataset.shuffle(seed=42).select(range(min(10**5, len(dataset))))
    dataset = dataset.train_test_split(train_size=0.8, seed=42)
    
    def format_example(examples):
        formatted_labels = []
        for i in range(len(examples['text'])):
            # Find label with highest count
            label_idx = max(label_idx_map.items(), 
                          key=lambda x: examples[x[0]][i])[1]
            formatted_labels.append(label_idx)
        examples['label'] = formatted_labels
        return examples
    
    dataset = dataset.map(format_example, batched=True, batch_size=32)
    
    train_sentences = list(dataset['train']['text'])
    train_labels = list(dataset['train']['label'])
    test_sentences = list(dataset['test']['text'])
    test_labels = list(dataset['test']['label'])
    
    return train_sentences, train_labels, test_sentences, test_labels

def load_dataset(params):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """
    params["base_prompt"] = None
    if params['dataset'] == 'sst2':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst2()
        params['prompt_prefix'] = "Your task is to classify the sentiment for a given review as ."
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
        
    elif params['dataset'] == 'sst5':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst5()
        label_tokens = ['terrible', 'bad', 'neutral', 'good', 'great']
        params['prompt_prefix'] = f"Your task is to classify the sentiment for a given review as one of {', '.join(label_tokens)}."
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {i:label for i, label in enumerate(label_tokens)}

    elif params['dataset'] == 'snli':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_snli()
        label_tokens = ['entails', 'neutral', 'contradiction']
        params['prompt_prefix'] = f"Your task is to classify a pair of premise and hypothesis as one of: {', '.join(label_tokens)}."
        params["q_prefix"] = ""
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {i: label for i, label in enumerate(label_tokens)}

    elif params['dataset'] == 'qqp':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_qqp()
        label_tokens = ['different', 'duplicate']
        params['prompt_prefix'] = f"Your task is to classify the pair of questions as one of {', '.join(label_tokens)}."
        params["q_prefix"] = ""
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {i:label for i, label in enumerate(label_tokens)}

    elif params['dataset'] == 'metatool':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_metatool()

    elif params['dataset'] == 'banking77':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_banking77()
                
    elif params['dataset'] == 'wildguard':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_wildguard()
                
    elif params['dataset'] == 'when2call':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_when2call()
    
    elif params['dataset'] == 'toxic_chat':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_toxic_chat()
    
    elif params['dataset'] == 'snips':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_snips()
    
    elif params['dataset'] == 'massive_intent':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_massive_intent()
        
    elif params['dataset'] == 'amazon_counterfactual':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_amazon_counterfactual()

    elif params['dataset'] == 'newsgroups':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_newsgroups()
    
    elif params['dataset'].startswith('dbpedia'):
        level = int(params['dataset'][-1])
        assert params['dataset'] in ('dbpedia_l1', 'dbpedia_l2'), "only dbpedia_l1 and dbpedia_l2 datasets names allowed (9/70 classes)"
        
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_dbpedia(level) # dbpedia9 or dbpedia20
    
    elif params['dataset'] == 'wikitoxic':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_wikitoxic()
        
    elif params['dataset'] == 'goemotions':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_goemotions()

    elif params['dataset'] == 'strategy_qa':
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {1: ['True'], 0: ['False']}
        params['inv_label_dict'] = {'True': 1, 'False': 0}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_strategy_qa(name='strategy_qa', label_dict=params['inv_label_dict'])
    
    elif params['dataset'] == 'worldtree':
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0:'A', 1:'B', 2:'C', 3:'D'}
        params['inv_label_dict'] = {0:'A', 1:'B', 2:'C', 3:'D'}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_commonsense_qa(name='worldtree', label_dict=params['inv_label_dict'])
    
    elif params['dataset'] == 'open_book_qa':
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0:'A', 1:'B', 2:'C', 3:'D'}#{'A': 0, 'B': 1, 'C':2, 'D':3, 'E':4}
        params['inv_label_dict'] = {0:'A', 1:'B', 2:'C', 3:'D'}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_commonsense_qa(name='open_book_qa', label_dict=params['inv_label_dict'])
    
    elif params['dataset'] == 'commonsense_qa':
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}#{'A': 0, 'B': 1, 'C':2, 'D':3, 'E':4}
        params['inv_label_dict'] = {v:k for k,v in params['label_dict'].items()}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_commonsense_qa(inv_label_dict=params['inv_label_dict'])

    elif params['dataset'] == 'agnews':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_agnews()
        params['prompt_prefix'] = "Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['World'], 1: ['Sports'], 2: ['Business'], 3: ['Technology']}
        params['inv_label_dict'] = {'World': 0, 'Sports': 1, 'Business': 2, 'Technology': 3, } # notice index start from 1 here
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'trec':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_trec()
        params['prompt_prefix'] = "Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n"
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer Type: "
        params['label_dict'] = {0: ['Number'], 1: ['Location'], 2: ['Person'], 3: ['Description'], 4: ['Entity'], 5: ['Ab']}
        params['inv_label_dict'] = {'Number': 0, 'Location': 1, 'Person': 2, 'Description': 3, 'Entity': 4, 'Ab': 5}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'rte':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_rte()
        params['prompt_prefix'] = ""
        params["q_prefix"] = " "
        params["a_prefix"] = "answer: "
        params['label_dict'] = {0: ['False'], 1: ['True']}
        params['inv_label_dict'] = {'False': 0, 'True': 1}
        params['num_user_input'] = 2
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'cb':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = get_cb()
        params['prompt_prefix'] = ""
        params["q_prefix"] = ""
        params["a_prefix"] = "answer: "
        params['label_dict'] = {0: ['false'], 1: ['neither'], 2: ['true']}
        params['inv_label_dict'] = {'false': 0, 'neither': 1, 'true': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    else:
        raise NotImplementedError
    
    return orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels
