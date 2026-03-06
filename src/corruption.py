import random
import re
import string

def introduce_typos(text, prob=0.1):
    """
    Randomly introduces typos into a given text
    """
    words = text.split()
    new_words = []
    
    for word in words:
        if len(word) > 3 and random.random() < prob:
            typo_type = random.choice(['swap', 'delete', 'insert'])
            idx = random.randint(0, len(word) - 2)
            
            if typo_type == 'swap':
                word_list = list(word)
                word_list[idx], word_list[idx+1] = word_list[idx+1], word_list[idx]
                word = "".join(word_list)
            elif typo_type == 'delete':
                idx = random.randint(0, len(word) - 1)
                word = word[:idx] + word[idx+1:]
            elif typo_type == 'insert':
                idx = random.randint(0, len(word))
                char = random.choice(string.ascii_lowercase)
                word = word[:idx] + char + word[idx:]
        new_words.append(word)
        
    return " ".join(new_words)

def split_hashtags(text, prob=0.5):
    """
    Identifies hashtags and splits them into constituent 
    words based on CamelCase or removing the hashtag.
    """
    def repl(match):
        if random.random() < prob:
            hashtag = match.group(1)
            # Split CamelCase
            splitted = re.sub('([a-z0-9])([A-Z])', r'\1 \2', hashtag)
            return splitted
        return match.group(0)

    # Find hashtags: #followed by alphanumeric characters
    return re.sub(r'#(\w+)', repl, text)

def remove_emojis(text):
    """
    Removes emojis from the text.
    """
    # Simple regex for emojis (covers most common ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U0001f900-\U0001f9ff"  # supplemental symbols and pictographs
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def apply_corruptions(text, corruptions):
    if 'typos' in corruptions:
        text = introduce_typos(text)
    if 'hashtag_split' in corruptions:
        text = split_hashtags(text)
    if 'emoji_removal' in corruptions:
        text = remove_emojis(text)
    return text

def create_corruption_ablations(df):
    """
    Creates a dictionary of corrupted dataframes for ablation.
    """
    ablations = {
        "original": df.copy(),
        "corruption_typos": df.copy(),
        "corruption_hashtags": df.copy(),
        "corruption_emojis": df.copy(),
        "corruption_all": df.copy()
    }
    
    ablations["corruption_typos"]["text"] = ablations["corruption_typos"]["text"].apply(lambda x: introduce_typos(x))
    ablations["corruption_hashtags"]["text"] = ablations["corruption_hashtags"]["text"].apply(lambda x: split_hashtags(x))
    ablations["corruption_emojis"]["text"] = ablations["corruption_emojis"]["text"].apply(lambda x: remove_emojis(x))
    ablations["corruption_all"]["text"] = ablations["corruption_all"]["text"].apply(
        lambda x: apply_corruptions(x, ['typos', 'hashtag_split', 'emoji_removal'])
    )
    
    return ablations
