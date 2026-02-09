"""
Streamlit demo for GPT-2 Small.
Chat with the SFT model or use text completion with token probability visualization.
Run:  streamlit run streamlit_app.py
"""
import streamlit as st
import torch
import tiktoken
import time
import os
from torch.nn import functional as F
from model import GPT, GPTConfig

# ============================================================================
# CONFIG
# ============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding('gpt2')
EOT = enc.eot_token
USER = 50257
ASSISTANT = 50258


def decode_safe(ids):
    return enc.decode([t for t in ids if t < enc.n_vocab])


def decode_token_display(token_id):
    """Decode a single token for display in prob bars (handles partial UTF-8)."""
    if token_id >= enc.n_vocab:
        return f'<{token_id}>'
    b = enc.decode_single_token_bytes(token_id)
    try:
        return b.decode('utf-8')
    except UnicodeDecodeError:
        return '<0x' + b.hex().upper() + '>'


@st.cache_resource
def load_model(path):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint['config'])
    model = GPT(config).to(device)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, checkpoint.get('val_loss', None)


# ============================================================================
# FIND AVAILABLE MODELS
# ============================================================================
available = {}
for name, paths in [
    ('SFT (Chat)', ['checkpoints/sft.pt', 'sft.pt']),
    ('Pretrained (Completion)', ['checkpoints/pretrained.pt', 'pretrained.pt']),
]:
    for p in paths:
        if os.path.exists(p):
            available[name] = p
            break


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="GPT-2 Small", page_icon="", layout="wide")

st.title("GPT-2 Small (124M)")
st.caption("Pre-trained on OpenWebText, fine-tuned on OpenAssistant. Trained from scratch.")

if not available:
    st.error("No model checkpoints found. Place `pretrained.pt` or `sft.pt` in `checkpoints/`.")
    st.stop()


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Chat", "Completion"], horizontal=True)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
    top_k = st.slider("Top-k", 1, 200, 40, 1)
    max_tokens = st.slider("Max tokens", 10, 500, 200, 10)

    st.divider()
    st.header("Model")
    if mode == "Chat" and 'SFT (Chat)' in available:
        model_key = 'SFT (Chat)'
    elif 'Pretrained (Completion)' in available:
        model_key = 'Pretrained (Completion)'
    else:
        model_key = list(available.keys())[0]
    model_key = st.selectbox("Checkpoint", list(available.keys()), index=list(available.keys()).index(model_key))
    model, config, val_loss = load_model(available[model_key])
    if val_loss:
        st.caption(f"Val loss: {val_loss:.4f}")
    st.caption(f"Device: {device}")
    st.caption(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    if mode == "Completion":
        st.divider()
        show_probs = st.checkbox("Show token probabilities", value=True)
        delay = st.slider("Delay per token (s)", 0.0, 2.0, 0.3, 0.1)


# ============================================================================
# GENERATION HELPERS
# ============================================================================
@torch.no_grad()
def generate_step(model, config, idx, temperature, top_k, context_ids=None):
    """One generation step. Returns chosen_idx and top-10 with display text."""
    idx_ctx = idx[:, -config.context_length:]
    logits, _ = model(idx_ctx)
    logits = logits[:, -1, :] / temperature

    probs = F.softmax(logits, dim=-1).squeeze(0)
    top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)))
    top_probs_norm = top_probs / top_probs.sum()
    sample_idx = torch.multinomial(top_probs_norm, num_samples=1)
    chosen_idx = top_indices[sample_idx].item()

    # Top-10 for display — show what text each token would add
    display_probs, display_indices = torch.topk(probs, min(10, probs.size(-1)))
    base_text = decode_safe(context_ids) if context_ids else ""
    top_tokens = []
    for p, i in zip(display_probs.tolist(), display_indices.tolist()):
        if context_ids is not None:
            candidate_text = decode_safe(context_ids + [i])
            visible = candidate_text[len(base_text):]
            tok = visible if visible else decode_token_display(i)
        else:
            tok = decode_token_display(i)
        top_tokens.append((tok, p, i == chosen_idx))

    return chosen_idx, top_tokens


# ============================================================================
# CHAT MODE
# ============================================================================
if mode == "Chat":
    # Session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    # Chat input
    if prompt := st.chat_input("Type a message..."):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.write(prompt)

        # Build token sequence
        tokens = []
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                tokens.append(USER)
                tokens.extend(enc.encode_ordinary(msg['content']))
                tokens.append(EOT)
            elif msg['role'] == 'assistant':
                tokens.append(ASSISTANT)
                tokens.extend(enc.encode_ordinary(msg['content']))
                tokens.append(EOT)
        tokens.append(ASSISTANT)

        # Truncate from left
        max_ctx = config.context_length - max_tokens
        if len(tokens) > max_ctx:
            tokens = tokens[-max_ctx:]

        # Generate with streaming
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        generated_ids = []

        with st.chat_message('assistant'):
            placeholder = st.empty()
            for _ in range(max_tokens):
                chosen_idx, _ = generate_step(model, config, idx, temperature, top_k, context_ids=generated_ids)
                if chosen_idx == EOT:
                    break
                generated_ids.append(chosen_idx)
                idx = torch.cat([idx, torch.tensor([[chosen_idx]], device=device)], dim=1)
                response = decode_safe(generated_ids)
                placeholder.markdown(response + "")

            response = decode_safe(generated_ids)
            placeholder.markdown(response)

        st.session_state.messages.append({'role': 'assistant', 'content': response})

    # Clear button
    if st.session_state.messages:
        if st.sidebar.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ============================================================================
# COMPLETION MODE
# ============================================================================
else:
    prompt = st.text_area("Prompt", placeholder="The meaning of life is", height=100)

    if st.button("Generate", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Enter a prompt.")
            st.stop()

        tokens = enc.encode_ordinary(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        all_ids = list(tokens)

        text_placeholder = st.empty()
        text_placeholder.markdown(prompt)

        if show_probs:
            st.divider()
            prob_placeholder = st.empty()

        for step_i in range(max_tokens):
            chosen_idx, top_tokens = generate_step(model, config, idx, temperature, top_k, context_ids=all_ids)

            if chosen_idx == EOT:
                break

            all_ids.append(chosen_idx)
            idx = torch.cat([idx, torch.tensor([[chosen_idx]], device=device)], dim=1)
            generated = decode_safe(all_ids)

            text_placeholder.markdown(generated)

            if show_probs:
                chosen_tok = next((t for t, _, c in top_tokens if c), top_tokens[0])
                html = (
                    f'<div style="padding:10px;background:#f0f2f6;border-radius:8px;font-family:monospace;font-size:13px;">'
                    f'<div style="font-weight:bold;margin-bottom:6px;font-size:15px;">'
                    f'Token {step_i + 1}: <span style="color:#22c55e;font-size:16px;">{repr(chosen_tok[0])[1:-1]}</span></div>'
                )
                for tok, prob, is_chosen in top_tokens:
                    tok_display = repr(tok)[1:-1]
                    width = max(int(prob * 100), 1)
                    if is_chosen:
                        html += (
                            f'<div style="background:linear-gradient(90deg,#22c55e {width}%,transparent {width}%);'
                            f'padding:4px 8px;margin:2px 0;border-radius:4px;border:1px solid #22c55e;">'
                            f'<b>{tok_display}</b> {prob:.1%}</div>'
                        )
                    else:
                        html += (
                            f'<div style="background:linear-gradient(90deg,#6366f1 {width}%,transparent {width}%);'
                            f'padding:4px 8px;margin:2px 0;border-radius:4px;opacity:0.5;">'
                            f'{tok_display} {prob:.1%}</div>'
                        )
                html += '</div>'
                prob_placeholder.html(html)

            if delay > 0:
                time.sleep(delay)

        st.success("Generation complete.")
