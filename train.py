"""
Pre-train GPT-2 Small on OpenWebText.
Run: python train.py
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

# tf32 for free speed on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ============================================================================
# CONFIG
# ============================================================================
config = GPTConfig()
batch_size = 64
max_steps = 140000
eval_interval = 500
eval_iters = 200
log_interval = 100

# Optimizer
learning_rate = 6e-4
min_lr = 6e-5
warmup_steps = 2000
weight_decay = 0.1
grad_clip = 1.0
betas = (0.9, 0.95)

# ============================================================================
# DATA
# ============================================================================
train_data = np.memmap('data/train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('data/val.bin', dtype=np.uint16, mode='r')
print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens:   {len(val_data):,}")


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + config.context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + config.context_length + 1].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(step):
    if step < warmup_steps:
        return learning_rate * (step + 1) / (warmup_steps + 1)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ============================================================================
# MODEL
# ============================================================================
model = GPT(config).to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
model = torch.compile(model)

optimizer = model.configure_optimizers(learning_rate, betas, weight_decay, device_type)

# ============================================================================
# TRAINING
# ============================================================================
train_losses = []
val_losses = []
steps_log = []
start_time = time.time()

for step in range(max_steps):
    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate
    if step % eval_interval == 0:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(f"step {step:6d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.0f}s")
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        steps_log.append(step)

    # Log step time
    if step % log_interval == 0 and step > 0:
        elapsed = time.time() - start_time
        tokens_per_sec = step * batch_size * config.context_length / elapsed
        if step % eval_interval != 0:
            print(f"step {step:6d} | {tokens_per_sec:.0f} tok/s | lr {lr:.2e}")

    # Forward + backward
    xb, yb = get_batch('train')
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

# Final evaluation
losses = estimate_loss()
total_time = time.time() - start_time
print(f"\nFinal | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | total {total_time:.0f}s")
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
}, 'checkpoints/pretrained.pt')
print("Saved checkpoints/pretrained.pt")

# Save metrics
with open('plots/pretrain_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'train_loss', 'val_loss'])
    for i in range(len(steps_log)):
        writer.writerow([steps_log[i], train_losses[i], val_losses[i]])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(steps_log, train_losses, label='Train')
ax.plot(steps_log, val_losses, label='Val')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('Pre-training Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/pretrain_loss.png', dpi=150)
print("Saved plots/pretrain_loss.png")
