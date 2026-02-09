"""
Prepare OpenAssistant oasst1 for SFT.
Downloads the dataset, formats conversations, tokenizes, saves as binary files.
Run: python prepare_sft_data.py
"""
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from collections import defaultdict

os.makedirs('data', exist_ok=True)

enc = tiktoken.get_encoding('gpt2')
EOT = enc.eot_token       # 50256 <|endoftext|>
USER = 50257              # <|user|>
ASSISTANT = 50258         # <|assistant|>
CONTEXT_LENGTH = 1024

# ============================================================================
# DOWNLOAD
# ============================================================================
print("Downloading OpenAssistant oasst1...")
dataset = load_dataset('OpenAssistant/oasst1')

# Combine all messages
all_messages = list(dataset['train']) + list(dataset['validation'])
print(f"Total messages: {len(all_messages):,}")

# ============================================================================
# BUILD CONVERSATION TREES
# ============================================================================
by_id = {m['message_id']: m for m in all_messages}
children = defaultdict(list)
roots = []

for m in all_messages:
    if m['parent_id'] is None:
        roots.append(m)
    else:
        children[m['parent_id']].append(m)


def extract_paths(node):
    """Extract all root-to-leaf conversation paths."""
    kids = children[node['message_id']]
    if not kids:
        return [[node]]
    paths = []
    for child in kids:
        for path in extract_paths(child):
            paths.append([node] + path)
    return paths


# ============================================================================
# FORMAT AND TOKENIZE
# ============================================================================
conversations = []
for root in roots:
    if root['lang'] != 'en':
        continue
    for path in extract_paths(root):
        tokens = []
        mask = []
        for msg in path:
            role_token = USER if msg['role'] == 'prompter' else ASSISTANT
            text_ids = enc.encode_ordinary(msg['text'])

            # Role marker (not trained on)
            tokens.append(role_token)
            mask.append(0)

            # Text content
            tokens.extend(text_ids)
            if msg['role'] == 'assistant':
                mask.extend([1] * len(text_ids))
            else:
                mask.extend([0] * len(text_ids))

            # End of turn (trained on for assistant turns)
            tokens.append(EOT)
            mask.append(1 if msg['role'] == 'assistant' else 0)

        # Skip short conversations
        if sum(mask) < 20:
            continue

        # Truncate to context length
        tokens = tokens[:CONTEXT_LENGTH]
        mask = mask[:CONTEXT_LENGTH]

        conversations.append((tokens, mask))

print(f"Conversations: {len(conversations):,}")

# ============================================================================
# SPLIT AND SAVE
# ============================================================================
np.random.seed(42)
indices = np.random.permutation(len(conversations))
val_size = max(500, len(conversations) // 20)
val_idx = indices[:val_size]
train_idx = indices[val_size:]

for name, idxs in [('sft_train', train_idx), ('sft_val', val_idx)]:
    all_tokens = []
    all_masks = []
    for i in idxs:
        tok, msk = conversations[i]
        # Pad to context_length
        pad_len = CONTEXT_LENGTH - len(tok)
        tok = tok + [EOT] * pad_len
        msk = msk + [0] * pad_len
        all_tokens.extend(tok)
        all_masks.extend(msk)

    tokens_arr = np.array(all_tokens, dtype=np.uint16)
    masks_arr = np.array(all_masks, dtype=np.uint8)

    tokens_arr.tofile(f'data/{name}.bin')
    masks_arr.tofile(f'data/{name}_mask.bin')

    n_convs = len(idxs)
    n_tokens = int(masks_arr.sum())
    print(f"{name}: {n_convs:,} conversations, {n_tokens:,} trained tokens")

print("\nDone. Files saved to data/")
