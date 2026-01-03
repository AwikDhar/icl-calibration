from google.oauth2 import service_account
from google import genai
from google.genai import types, Client

import json
from pathlib import Path

def setup_gemini_client():
    parent_dir = Path(__file__).parent
    
    creds = service_account.Credentials.from_service_account_file(
        filename=parent_dir/'creds.json', 
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    
    config_path =  parent_dir / 'gemma_config.json'
    with open(config_path, 'r') as f:
        gemma_config = json.load(f)
    
    return genai.Client(
        vertexai=True,
        project=gemma_config['project_id'],
        location=gemma_config['location'],
        credentials=creds
    )
class GeminiModel():
    def __init__(self, model_name):
        self.gemini_client = setup_gemini_client()
        self.model_name = model_name
    
    def generate(self, 
            prompt, 
            temperature=0.0,
            max_output_tokens=100,
            top_p=1.0,
            top_k=1,
            num_log_probs=None,
            logprobs=None
        ):
        

        if logprobs is None:
            logprobs = num_log_probs if num_log_probs else None
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            response_logprobs=True,
            logprobs=num_log_probs,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(
                    # thinking_level=types.ThinkingLevel.LOW  # For faster and lower-latency responses
                    thinking_budget=0  # For faster and lower-latency responses
            )
        )
    
        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        return response
    
class GeminiTokenizer():
    def __init__(self, model_name):
        self.gemini_client = setup_gemini_client()
        self.model_name = model_name
        
    def tokenize(self, prompt):
        tokens_response = self.gemini_client.models.compute_tokens(
                model=self.model_name,
                contents=prompt,
            )

        tokens = tokens_response.tokens_info[0].tokens
        tokens = [token.decode('utf-8') for token in tokens]
        
        return tokens