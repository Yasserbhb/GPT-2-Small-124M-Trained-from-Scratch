import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    context_length: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embed // config.n_head
        self.qkv = nn.Linear(config.n_embed, 3 * config.n_embed, bias=False)
        self.proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.resid_dropout(self.proj(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=False)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa_head = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed, bias=False)
        self.ln2 = nn.LayerNorm(config.n_embed, bias=False)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embedding = nn.Embedding(config.context_length, config.n_embed)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed, bias=False)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight tying
        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            if loss_mask is not None:
                loss_mask_flat = loss_mask.view(-1).float()
                loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                loss = (loss * loss_mask_flat).sum() / loss_mask_flat.sum()
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)
        else:
            # inference: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, lr, betas, weight_decay, device_type):
        """Separate weight decay for 2D params (matmuls/embeds) vs 1D params (biases/norms)."""
        decay = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
        fused = device_type == 'cuda'
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_ctx = idx[:, -self.config.context_length:]
            logits, _ = self(idx_ctx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
