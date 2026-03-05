import pandas as pd
import numpy as np

def print_samples(df, labels, n=5):
    from IPython.display import display
    
    sample_df = df.sample(n).copy()
    sample_df['Label'] = sample_df['label'].apply(lambda x: labels[x])
    sample_df['Tweet Text'] = sample_df['text'].str.replace('\n', ' ')
    
    display_df = sample_df[['Label', 'Tweet Text']]
    display(display_df)

def print_distribution(train_df, test_df, labels):
    print(f"\nData Split")
    print(f"Training set: {len(train_df)}")
    print(f"Testing set:  {len(test_df)}")

    print("\nClass Distribution (Counts)")
    train_counts = train_df['label'].map(labels).value_counts()
    test_counts = test_df['label'].map(labels).value_counts()

    dist_df = pd.DataFrame({
        'Train Count': train_counts,
        'Test Count': test_counts,
        'Train %': (train_counts / len(train_df) * 100).round(2)
    })
    print(dist_df)

def calculate_ece(y_true, y_preds, confidences, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    total_samples = len(y_true)
    accuracies = (y_preds == y_true)
    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)
    return ece

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def loss_plot(log_history, model_name):
    train_data = [e for e in log_history if 'loss' in e]
    eval_data = [e for e in log_history if 'eval_loss' in e]
    
    df_train = pd.DataFrame(train_data)
    df_eval = pd.DataFrame(eval_data)
    
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    if not df_train.empty:
        plt.plot(df_train['epoch'], df_train['loss'], 'o-', label='Train Loss', color='#1f77b4')
    if not df_eval.empty:
        plt.plot(df_eval['epoch'], df_eval['eval_loss'], 's-', label='Validation Loss', color='#ff7f0e')
        
    plt.title(f"Training & Validation Loss: {model_name}", fontsize=14, pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(frameon=True, facecolor='white')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def show_tokenization(text):
    from zero_shot import DistilBERT_zero_shot_pipeline, DistilRoBERTa_zero_shot_pipeline
    tokenizer_bert = DistilBERT_zero_shot_pipeline.tokenizer
    tokenizer_rob = DistilRoBERTa_zero_shot_pipeline.tokenizer
    
    tokens_bert = tokenizer_bert.tokenize(text)
    tokens_rob = tokenizer_rob.tokenize(text)
    
    return [
        {"Tokenizer": "WordPiece", "Model": "DistilBERT", "Tokens": tokens_bert},
        {"Tokenizer": "BPE", "Model": "DistilRoBERTa", "Tokens": tokens_rob}
    ]