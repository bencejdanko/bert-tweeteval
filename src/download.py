from datasets import load_dataset
import pandas as pd

def download_and_split_dataset():
    dataset = load_dataset("tweet_eval", "emotion")

    train_df = pd.DataFrame(dataset["train"])
    val_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])

    return train_df, val_df, test_df