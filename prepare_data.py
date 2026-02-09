"""
Prepare OpenWebText for pre-training.
Downloads the dataset, tokenizes with GPT-2 BPE, saves as binary files.
Run: python prepare_data.py
"""
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

os.makedirs('data', exist_ok=True)

num_proc = 8
enc = tiktoken.get_encoding('gpt2')

# Download OpenWebText
print("Downloading OpenWebText...")
dataset = load_dataset('openwebtext', num_proc=num_proc)

# Split into train/val (only has 'train' by default)
split = dataset['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split['val'] = split.pop('test')

print(f"Train: {len(split['train']):,} documents")
print(f"Val:   {len(split['val']):,} documents")


# Tokenize
def tokenize(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)  # 50256
    return {'ids': ids, 'len': len(ids)}


tokenized = split.map(tokenize, remove_columns=['text'], desc='Tokenizing', num_proc=num_proc)

# Write to binary files
for name, dset in tokenized.items():
    total_tokens = np.sum(dset['len'], dtype=np.uint64)
    print(f"\n{name}: {total_tokens:,} tokens")

    path = f'data/{name}.bin'
    arr = np.memmap(path, dtype=np.uint16, mode='w+', shape=(total_tokens,))

    idx = 0
    for batch_idx in tqdm(range(1024), desc=f'Writing {path}'):
        batch = dset.shard(num_shards=1024, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx:idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

print("\nDone. Files saved to data/")
