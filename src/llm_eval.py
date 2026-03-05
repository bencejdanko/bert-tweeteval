import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from openai import OpenAI, AsyncOpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import backoff
import asyncio

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
    def __init__(self, openai_api_key=None, base_url="https://openrouter.ai/api/v1", hf_model_name="Qwen/Qwen3-4B-Instruct-2507"):
        self.openai_api_key = openai_api_key
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/bencejdanko/bert-tweeteval",
                "X-Title": "BERT TweetEval Research",
            }
        ) if openai_api_key else None
        self.openai_async_client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/bencejdanko/bert-tweeteval",
                "X-Title": "BERT TweetEval Research",
            }
        ) if openai_api_key else None
        self.hf_model_name = hf_model_name
        self.hf_model = None
        self.hf_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_hf_model(self):
        print(f"Loading HF model: {self.hf_model_name} on {self.device}...")
        self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        self.hf_tokenizer.padding_side = "left"
        if self.hf_tokenizer.pad_token is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
        
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def _call_openai_async(self, prompt, model="openai/gpt-4o-mini"):
        response = await self.openai_async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        return response.choices[0].message.content.strip().lower()

    async def _evaluate_batch_openai(self, prompts, model="openai/gpt-4o-mini"):
        semaphore = asyncio.Semaphore(20)
        async def sem_call(p):
            async with semaphore:
                return await self._call_openai_async(p, model=model)
        
        tasks = [sem_call(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def call_hf_batch(self, prompts):
        """Process a batch of prompts using the HF model."""
        inputs = self.hf_tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.hf_model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.hf_tokenizer.pad_token_id
            )
        
        responses = []
        for i in range(len(prompts)):
            decoded = self.hf_tokenizer.decode(outputs[i][input_length:], skip_special_tokens=True)
            responses.append(decoded.strip().lower())
        return responses

    def evaluate(self, df, model_type, prompt_template, batch_size=100):
        all_predictions = []
        batch_times = []
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        for i in tqdm(range(0, len(df), batch_size), desc=f"Evaluating {model_type} (Batch size {batch_size})"):
            batch_df = df.iloc[i : i + batch_size]
            prompts = [prompt_template.format(text=row['text']) for _, row in batch_df.iterrows()]
            
            start_time = time.time()
            if model_type == "openai":
                if loop.is_running():
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                    except ImportError:
                        print("Warning: nest_asyncio not found. Async calls in Jupyter might fail. Run '!pip install nest_asyncio'")
                preds = loop.run_until_complete(self._evaluate_batch_openai(prompts))
            elif model_type == "hf":
                preds = self.call_hf_batch(prompts)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            end_time = time.time()
            batch_duration = end_time - start_time
            normalized_time = (batch_duration / len(batch_df)) * 100
            batch_times.append(normalized_time)
            
            for pred in preds:
                pred_clean = "unknown"
                for label in LABELS:
                    if label in pred:
                        pred_clean = label
                        break
                all_predictions.append(pred_clean)

        y_true = [LABELS[i] for i in df['label']]
        acc = accuracy_score(y_true[:len(all_predictions)], all_predictions)
        f1 = f1_score(y_true[:len(all_predictions)], all_predictions, average='macro')
        
        avg_time_per_100 = sum(batch_times) / len(batch_times) if batch_times else 0

        return {
            "Accuracy": acc,
            "Macro-F1": f1,
            "Time_per_100": avg_time_per_100,
            "Predictions": all_predictions,
            "True_Labels": y_true
        }
