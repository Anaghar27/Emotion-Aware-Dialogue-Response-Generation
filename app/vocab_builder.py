import os
import pickle
import pandas as pd
from collections import Counter

# Build a vocabulary dictionary from a list of token lists.
def build_vocab(token_lists, min_freq=2):
    counter = Counter()
    for tokens in token_lists:
        if isinstance(tokens, list):
            counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

# Build and save a vocabulary from MELD and DailyDialog tokenized datasets.
def save_vocab(meld_pkl_path, dd_pkl_path, save_path, min_freq=2):
    meld_df = pd.read_pickle(meld_pkl_path)
    dd_df = pd.read_pickle(dd_pkl_path)

    meld_tokens = meld_df['tokens'].tolist()
    dd_tokens = dd_df['Tokens'].tolist()

    all_tokens = dd_tokens
    
    vocab = build_vocab(all_tokens, min_freq=min_freq)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)

    print(f"Saved vocab of size {len(vocab)} to {save_path}")