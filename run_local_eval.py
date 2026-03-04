import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath('src'))

from download import download_and_split_dataset
from llm_eval import LLMEvaluator, PROMPT_1_MINIMAL, PROMPT_2_STRUCTURED, LABELS

def main():
    train_df, val_df, test_df = download_and_split_dataset()
    # Use a small subset for quick local evaluation
    num_samples = 20
    test_subset = test_df.sample(num_samples, random_state=42)
    
    evaluator = LLMEvaluator(hf_model_name="Qwen/Qwen3.5-0.8B")
    evaluator.load_hf_model()
    
    print("\nEvaluating Qwen (Minimal Prompt)...")
    res_p1 = evaluator.evaluate(test_subset, "hf", PROMPT_1_MINIMAL, num_samples=None)
    
    print("\nEvaluating Qwen (Structured Prompt)...")
    res_p2 = evaluator.evaluate(test_subset, "hf", PROMPT_2_STRUCTURED, num_samples=None)
    
    results = {
        "Qwen-0.8B (Minimal)": res_p1,
        "Qwen-0.8B (Structured)": res_p2
    }
    
    for name, res in results.items():
        print(f"\nResults for {name}:")
        print(f"Accuracy: {res['Accuracy']:.4f}")
        print(f"Macro-F1: {res['Macro-F1']:.4f}")
        print(f"Time/100: {res['Time_per_100']:.2f}s")

if __name__ == "__main__":
    main()
