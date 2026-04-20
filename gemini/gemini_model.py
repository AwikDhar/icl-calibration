import time
from google.oauth2 import service_account
from google import genai
from google.genai import types, Client
from google.genai.errors import ClientError, APIError

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
    def __init__(self, model_name, retry_attempt=3):
        self.gemini_client = setup_gemini_client()
        self.model_name = model_name
        self.retry_attempt = retry_attempt
        
    def generate(self, 
            prompt, 
            temperature=0.0,
            max_output_tokens=1,
            top_p=1.0,
            top_k=1,
            num_log_probs=20,
        ):
    
        if "gemini-3" in self.model_name:
            thinking_config = types.ThinkingConfig(
                    # thinking_level replaces thinking_budget for gemini 3 
                    thinking_level=types.ThinkingLevel.MINIMAL
                    # thinking_level="MINIMAL"  
            )
        else:
            thinking_config = types.ThinkingConfig(
                    thinking_budget=0 
            )
            
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            response_logprobs=True,
            logprobs=num_log_probs,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=thinking_config,
            # Filters off for toxicity classification
            safety_settings=[types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_UNSPECIFIED, threshold=types.HarmBlockThreshold.OFF),
                             types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.OFF),
                             types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=types.HarmBlockThreshold.OFF),
                             types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_JAILBREAK, threshold=types.HarmBlockThreshold.OFF),
                             types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.OFF),
                             types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.OFF),
                             types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.OFF)]
            )
    
        total_attempts = self.retry_attempt + 1
        
        for attempt in range(total_attempts):
            try:
                if attempt>0:
                    print(f"Retry attempt {attempt}/{self.retry_attempt}")
                    
                response = self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )

                return response

            except ClientError as ce:
                if attempt==self.retry_attempt-1:
                    raise ce
                
                sleep_time = (attempt + 1)*7
                print(f"{ce.__repr__()}\nRetrying in {sleep_time} sec...")
                time.sleep(sleep_time)
            except Exception as e:
                if attempt==self.retry_attempt-1:
                    raise e
                
                sleep_time = (attempt + 1)*7
                print(f"{e.__repr__()}\nRetrying in {sleep_time} sec...")
                time.sleep(sleep_time)
                
                continue

    
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