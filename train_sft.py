"""
SFT fine-tuning on OpenAssistant conversations.
Loads the pre-trained checkpoint and fine-tunes with masked loss.
Run: python train_sft.py
"""
import torch
import numpy as np
import time
import math
import os
import csv
import matplotlib.pyplot as plt
from model import GPT, GPTConfig

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
print(f'Device: {device}')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ============================================================================
# CONFIG
# ============================================================================
config = GPTConfig()
batch_size = 32
n_epochs = 5
eval_interval = 100
eval_iters = 50

learning_rate = 2e-5
min_lr = 2e-6
weight_decay = 0.1
grad_clip = 1.0
betas = (0.9, 0.95)

# ============================================================================
# DATA
# ============================================================================
train_tokens = np.fromfile('data/sft_train.bin', dtype=np.uint16)
train_masks = np.fromfile('data/sft_train_mask.bin', dtype=np.uint8)
val_tokens = np.fromfile('data/sft_val.bin', dtype=np.uint16)
val_masks = np.fromfile('data/sft_val_mask.bin', dtype=np.uint8)

n_train = len(train_tokens) // config.context_length
n_val = len(val_tokens) // config.context_length
max_steps = n_epochs * (n_train // batch_size)

print(f"Train conversations: {n_train:,}")
print(f"Val conversations:   {n_val:,}")
print(f"Max steps:           {max_steps:,}")


def get_batch(split):
    tokens = train_tokens if split == 'train' else val_tokens
    masks = train_masks if split == 'train' else val_masks
    n = len(tokens) // config.context_length

    ix = torch.randint(n, (batch_size,))
    x_list, y_list, m_list = [], [], []
    for i in ix:
        start = i * config.context_length
        end = start + config.context_length
        chunk = tokens[start:end].astype(np.int64)
        mask_chunk = masks[start:end].astype(np.int64)
        x_list.append(torch.from_numpy(chunk[:-1]))
        y_list.append(torch.from_numpy(chunk[1:]))
        m_list.append(torch.from_numpy(mask_chunk[1:]))

    x = torch.stack(x_list)
    y = torch.stack(y_list)
    m = torch.stack(m_list)
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        m = m.pin_memory().to(device, non_blocking=True)
    else:
        x, y, m = x.to(device), y.to(device), m.to(device)
    return x, y, m


def get_lr(step):
    decay_ratio = step / max_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, M = get_batch(split)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(X, Y, loss_mask=M)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ============================================================================
# MODEL (load pretrained)
# ============================================================================
checkpoint = torch.load('checkpoints/pretrained.pt', map_location=device, weights_only=False)
model = GPT(config).to(device)
# Strip _orig_mod. prefix added by torch.compile
state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
model.load_state_dict(state_dict)
model = torch.compile(model)
print(f"Loaded pretrained model (val loss: {checkpoint['val_loss']:.4f})")
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

optimizer = model.configure_optimizers(learning_rate, betas, weight_decay, device_type)

# ============================================================================
# TRAINING
# ============================================================================
train_losses = []
val_losses = []
steps_log = []
start_time = time.time()

for step in range(max_steps):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if step % eval_interval == 0:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        epoch = step * batch_size / n_train
        print(f"step {step:5d} | epoch {epoch:.1f} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.0f}s")
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        steps_log.append(step)

    X, Y, M = get_batch('train')
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _, loss = model(X, Y, loss_mask=M)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

# Final eval
losses = estimate_loss()
total_time = time.time() - start_time
print(f"\nFinal | train {losses['train']:.4f} | val {losses['val']:.4f} | total {total_time:.0f}s")
train_losses.append(losses['train'])
val_losses.append(losses['val'])
steps_log.append(max_steps)

# ============================================================================
# SAVE
# ============================================================================
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.__dict__,
    'step': max_steps,
    'train_loss': train_losses[-1],
    'val_loss': val_losses[-1],
    'total_time': total_time,
}, 'checkpoints/sft.pt')
print("Saved checkpoints/sft.pt")

with open('plots/sft_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'train_loss', 'val_loss'])
    for i in range(len(steps_log)):
        writer.writerow([steps_log[i], train_losses[i], val_losses[i]])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(steps_log, train_losses, label='Train')
ax.plot(steps_log, val_losses, label='Val')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('SFT Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/sft_loss.png', dpi=150)
print("Saved plots/sft_loss.png")
