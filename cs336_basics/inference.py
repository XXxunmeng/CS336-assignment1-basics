import torch
from cs336_basics.transformer import TransformerLM, softmax
from cs336_basics.bpe import Tokenizer
from cs336_basics.train import load_checkpoint
import pickle

def generate(model, tokenizer, prompt, max_token=512, temperature=0.8, top_p = 0.9):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 编码prompt
    tokens = tokenizer.encode(prompt)
    # (1, seq_len)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    model.to(device)
    end_idx = tokenizer.encode("<|endoftext|>")[0]
    with torch.no_grad():
        for i in range(max_token):
            # (1, seq_len, vocab_size)
            hidden = model(tokens)
            # (1, vocab_size)
            logits = hidden[:, -1, :]
            logits = logits / temperature

            probs = softmax(logits, -1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)

            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[0] = 0

            sorted_probs[sorted_indices_to_remove] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            sample_idx = torch.multinomial(sorted_probs, num_samples=1)

            nxt_token = sorted_indices.gather(-1, sample_idx)
            tokens = torch.cat([tokens, nxt_token], dim=-1)

            if nxt_token.item() == end_idx:
                break
    return tokenizer.decode(tokens.squeeze(0).tolist())

if __name__ == "__main__":
    prompt = ""
    ckpt_path = "checkpoints/ckpt.pt"
    model = TransformerLM(vocab_size=10000, context_length=512, num_layers=4, d_model=512, num_heads=16, d_ff=1344, theta=10000.0)
    checkpoint = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    with open("cs336_basics/TinyStories_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("cs336_basics/TinyStories_merges.pkl", "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    # 4. 开始生成！
    output = generate(model, tokenizer, prompt="Once upon a time, there was a little girl named Lily.")
    print(output)