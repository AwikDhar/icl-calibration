import argparse
import os
from transformers import AutoTokenizer
from utils import convert_to_list
from data_utils.load_data import set_prompt_params

def main(models, datasets):
    for model in models:
        for dataset in datasets:
            params={'dataset':dataset}
            set_prompt_params(params)
            print(params['label_dict'])
            
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, cache_dir=os.environ['HF_HOME'])
            
            for id, label in params['label_dict'].items():       
                tokens = tokenizer.tokenize(label)
                if len(tokens)>1:
                    print(f"Label: {label} is split into multi-tokens: {tokens}")
                    alt_tokens = tokenizer.tokenize(" "+label)
                    if len(alt_tokens)<len(tokens):
                        print(f"After adding space:{alt_tokens}\n")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--models', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')

    args = parser.parse_args()
    args = vars(args)

    args['models'] = convert_to_list(args['models'])
    args['datasets'] = convert_to_list(args['datasets'])
    
    main(**args)