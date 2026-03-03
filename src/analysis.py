import pandas as pd
import numpy as np

def print_samples(df, labels, n=5):
    print(f"{'Label':<12} | {'Tweet Text'}")
    print("-" * 50)
    for _, row in df.sample(n).iterrows():
        label_name = labels[row['label']]
        text = row['text'].replace('\n', ' ')
        print(f"{label_name:<12} | {text[:80]}...")

def print_distribution(train_df, test_df, labels):
    print(f"\n--- Data Split Summary ---")
    print(f"Training set: {len(train_df)}")
    print(f"Testing set:  {len(test_df)}")

    print("\n--- Class Distribution (Counts) ---")
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