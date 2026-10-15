import torch
from torch import nn
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        # weight: (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        # Xavier初始化
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return x @ self.weight.T
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(self.d_model, dtype=dtype, device=device))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps) ** 0.5
        result = x / rms * self.weight
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.w3(x)
        gate = self.silu(self.w1(x))
        hidden = gate * up
        return self.w2(hidden)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_model = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        # (d_k // 2)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        # (max_seq_len)
        positions = torch.arange(max_seq_len, device=device)
        # (max_seq_len, d_k // 2)
        freqs = torch.outer(positions, freqs).float()
        self.register_buffer("cos_cache", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cache", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 按照奇偶分两组
        x_0 = x[..., 0::2]
        x_1 = x[..., 1::2]
        # (seq_len, d_k // 2)
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        result1 = cos * x_0 - sin * x_1
        result2 = sin * x_0 + cos * x_1
        result = torch.stack([result1, result2], dim=-1)
        return result.flatten(-2)

def softmax(x: torch.tensor, dim: int):
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    d_k = query.shape[-1]
    score = query @ key.transpose(-2, -1) / d_k ** 0.5
    if mask is not None:
        score = score.masked_fill(mask == 0, float('-inf'))
    score = softmax(score, dim=-1)
    return score @ value

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, num_heads * self.head_dim)
        self.k_proj = Linear(d_model, num_heads * self.head_dim)
        self.v_proj = Linear(d_model, num_heads * self.head_dim)
        self.output_proj = Linear(num_heads * self.head_dim, d_model)
        self.rope = None
        if theta is not None:
            self.rope = RoPE(theta, self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        result = scaled_dot_product_attention(q, k , v, mask)
        result = result.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.output_proj(result)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float | None = None, max_seq_len: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        result = x + self.attn(self.ln1(x))
        result = result + self.ffn(self.ln2(result))
        return result

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, theta: float | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, theta, context_length) for _ in range(num_layers)]
        )
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embeddings(inputs)
        for layer in self.layers:
            hidden = layer(hidden)
        result = self.lm_head(self.ln_final(hidden))
        return result