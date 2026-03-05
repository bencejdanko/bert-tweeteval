import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import backoff

# Prompt Templates
PROMPT_1_MINIMAL = """Classify the following tweet into one of these emotions: anger, joy, optimism, sadness.
Tweet: {text}
Emotion:"""

PROMPT_2_STRUCTURED = """Task: Sentiment classification for tweets.
Labels:
- anger: The tweet expresses frustration, resentment, or rage.
- joy: The tweet expresses happiness, pleasure, or satisfaction.
- optimism: The tweet expresses hopefulness, confidence about the future, or positive anticipation.
- sadness: The tweet expresses sorrow, disappointment, or unhappiness.

Instructions:
1. Read the tweet provided below.
2. Select the most appropriate label from the list above.
3. Output ONLY the label name. Do not include any other text or explanation.

Tweet: {text}

Label:"""

LABELS = ["anger", "joy", "optimism", "sadness"]

class LLMEvaluator:
    def __init__(self, openai_api_key=None, hf_model_name="Qwen/Qwen3-4B-Instruct-2507"):
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.hf_model_name = hf_model_name
        self.hf_model = None
        self.hf_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_hf_model(self):
        print(f"Loading HF model: {self.hf_model_name} on {self.device}...")
        self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def call_openai_stream(self, prompt, model="gpt-4o-mini"):
        """Utility for OpenAI calls with backoff handled by wrapper."""
        @backoff.on_exception(backoff.expo, Exception, max_tries=5)
        def _call():
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            return response.choices[0].message.content.strip().lower()
        return _call()

    def call_hf(self, prompt):
        inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.hf_model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.hf_tokenizer.eos_token_id
            )
        response = self.hf_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip().lower()

    def evaluate(self, df, model_type, prompt_template):
        eval_df = df

        predictions_and_times = []

        def process_row(row_data):
            idx, row = row_data
            prompt = prompt_template.format(text=row['text'])
            try:
                req_start = time.time()
                if model_type == "openai":
                    pred = self.call_openai_stream(prompt)
                elif model_type == "hf":
                    pred = self.call_hf(prompt)
                req_time = time.time() - req_start
                
                # Simple cleaning
                pred_clean = "unknown"
                for label in LABELS:
                    if label in pred:
                        pred_clean = label
                        break
                return pred_clean, req_time
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                return "error", 0.0

        if model_type == "openai":
            from concurrent.futures import ThreadPoolExecutor
            if not self.openai_client:
                raise ValueError("OpenAI client not initialized. Provide API key.")
            
            # Using 10 workers to stay within standard rate limits while being much faster
            with ThreadPoolExecutor(max_workers=10) as executor:
                predictions_and_times = list(tqdm(
                    executor.map(process_row, eval_df.iterrows()), 
                    total=len(eval_df), 
                    desc=f"Evaluating {model_type} (Parallel)"
                ))
        else:
            for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"Evaluating {model_type}"):
                predictions_and_times.append(process_row((idx, row)))
        
        predictions = [pt[0] for pt in predictions_and_times]
        times = [pt[1] for pt in predictions_and_times]
        
        # Metrics
        y_true = [LABELS[i] for i in eval_df['label']]
        acc = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, average='macro')
        
        avg_time_per_request = sum(times) / len(times) if times else 0
        time_per_100 = avg_time_per_request * 100

        return {
            "Accuracy": acc,
            "Macro-F1": f1,
            "Time_per_100": time_per_100,
            "Predictions": predictions,
            "True_Labels": y_true
        }
