"""
Generate text from a trained model.
Usage:
  python generate.py                              # pretrained, text completion
  python generate.py --model checkpoints/sft.pt   # SFT, chat mode
  python generate.py --chat                        # force chat mode
"""
import argparse
import torch
import tiktoken
from model import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='checkpoints/pretrained.pt')
parser.add_argument('--chat', action='store_true', help='Chat mode (auto-detected for SFT)')
parser.add_argument('--prompt', default=None, help='Prompt for completion mode')
parser.add_argument('--max_tokens', type=int, default=200)
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--top_k', type=int, default=40)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding('gpt2')
EOT = enc.eot_token  # 50256
USER = 50257
ASSISTANT = 50258

# Load model
checkpoint = torch.load(args.model, map_location=device, weights_only=False)
config = GPTConfig(**checkpoint['config'])
model = GPT(config).to(device)
# Strip _orig_mod. prefix added by torch.compile
state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded {args.model} (loss: {checkpoint.get('val_loss', '?'):.4f})")

is_sft = 'sft' in args.model or args.chat


def decode_safe(ids):
    """Decode token IDs, skipping special tokens."""
    return enc.decode([t for t in ids if t < enc.n_vocab])


if is_sft:
    # Chat mode
    print("\nChat mode. Type 'quit' to exit.\n")
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ('quit', 'exit', 'q'):
            break

        # Build token sequence
        tokens = []
        for role, text in history:
            role_id = USER if role == 'user' else ASSISTANT
            tokens.append(role_id)
            tokens.extend(enc.encode_ordinary(text))
            tokens.append(EOT)

        tokens.append(USER)
        tokens.extend(enc.encode_ordinary(user_input))
        tokens.append(EOT)
        tokens.append(ASSISTANT)

        # Truncate from left if too long
        if len(tokens) > config.context_length - args.max_tokens:
            tokens = tokens[-(config.context_length - args.max_tokens):]

        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        output = model.generate(idx, args.max_tokens, args.temperature, args.top_k)
        new_tokens = output[0, len(tokens):].tolist()

        # Stop at EOT
        if EOT in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(EOT)]

        response = decode_safe(new_tokens)
        print(f"GPT: {response}\n")

        history.append(('user', user_input))
        history.append(('assistant', response))
else:
    # Completion mode
    if args.prompt:
        tokens = enc.encode_ordinary(args.prompt)
    else:
        tokens = [EOT]

    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    output = model.generate(idx, args.max_tokens, args.temperature, args.top_k)
    print(decode_safe(output[0].tolist()))
