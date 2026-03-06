import pandas as pd
import re

def get_subset_by_feature(df, feature_type, present=True):
    if feature_type == 'mentions':
        pattern = r'@user|@\w+'
    elif feature_type == 'links':
        pattern = r'http\S+|http'
    elif feature_type == 'hashtags':
        pattern = r'#\w+'
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    mask = df['text'].str.contains(pattern, case=False, regex=True, na=False)
    
    if present:
        return df[mask].copy()
    else:
        return df[~mask].copy()

def create_shift_ablation_sets(df):
    shifts = {
        "full_test": df,
        "with_mentions": get_subset_by_feature(df, 'mentions', True),
        "no_mentions": get_subset_by_feature(df, 'mentions', False),
        "with_links": get_subset_by_feature(df, 'links', True),
        "no_links": get_subset_by_feature(df, 'links', False),
        "with_hashtags": get_subset_by_feature(df, 'hashtags', True),
        "no_hashtags": get_subset_by_feature(df, 'hashtags', False),
    }
    
    # Filter out empty sets just in case
    return {k: v for k, v in shifts.items() if len(v) > 0}

def print_shift_stats(shifts):
    print(f"{'Shift Name':<20} | {'Count':<6} | {'% of Original'}")
    print("-" * 45)
    total = len(shifts["full_test"])
    for name, df in shifts.items():
        count = len(df)
        pct = (count / total) * 100
        print(f"{name:<20} | {count:<6} | {pct:>5.1f}%")
