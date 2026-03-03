from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

def download_and_split_dataset():
    dataset = load_dataset("tweet_eval", "emotion")

    full_df = pd.concat([pd.DataFrame(dataset[split]) for split in dataset.keys()])

    sample_size = min(12000, len(full_df))
    subset_df = full_df.sample(n=sample_size, random_state=15179996).reset_index(drop=True)

    train_df, test_df = train_test_split(
        subset_df,
        test_size=0.2,
        random_state=15179996,
        stratify=subset_df["label"]
    )

    return train_df, test_df