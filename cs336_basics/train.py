from torch import nn
import torch
import math
import numpy as np
from typing import Iterable
from cs336_basics.transformer import TransformerLM
import os
import argparse

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    # inputs: [..., vocab_size]
    # targets: [batch_size]
    maxn = inputs.max(dim=-1, keepdim=True).values
    inputs = inputs - maxn
    # [batch_size]
    target_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    tmp = torch.log(torch.sum(torch.exp(inputs), dim=-1))
    loss = (tmp - target_logits).mean()
    return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                state["step"] += 1
                t = state["step"]

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                lr_new = group['lr'] * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
                p.data = p.data - group['lr'] * group["weight_decay"] * p.data
                p.data = p.data - lr_new * m / (v ** 0.5 + group['eps'])
                state['m'] = m
                state['v'] = v

def lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        tmp = math.cos(((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) + 1
        result = min_learning_rate + 0.5 * tmp * (max_learning_rate - min_learning_rate)
        return result

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    l2_norm = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        l2_norm += (p.grad.data ** 2).sum().item()
    l2_norm = l2_norm ** 0.5

    if l2_norm > max_l2_norm:
        tmp = max_l2_norm / (l2_norm + eps)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data = p.grad.data * tmp

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

def get_batch(dataset, batch_size, context_length, device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x, y = [], []
    for start in starts:
        tmp = dataset[start:start + context_length + 1]
        x.append(tmp[:-1])
        y.append(tmp[1:])

    x = np.stack(x)
    y = np.stack(y)
    # Embedding需要的是LongTensor，需要转成uint64
    x = torch.from_numpy(x.astype(np.int64)).to(device)
    y = torch.from_numpy(y.astype(np.int64)).to(device)
    return x, y


def train(train_data_src: str, eval_data_src: str, vocab_size: int, context_length: int, num_layers: int, d_model: int, 
            num_heads: int, d_ff: int, theta: float | None = None, lr: float = 1e-3, weight_decay: float = 0.01,
            steps: int = 1000, warmup_steps = 100, max_lr: float = 1e-3, min_lr : float = 1e-5, batch_size: int = 8, 
            max_l2_norm: float = 1.0, ckpt_path: str = None, ckpt_interval: int = 100, eval_interval: int = 100):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    # 加载数据
    dataset_train = np.load(train_data_src)
    dataset_eval = np.load(eval_data_src)

    # 模型初始化，优化器
    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta).to(device)
    optimizer = AdamW(model.parameters(), lr, weight_decay)

    start_step = -1
    if os.path.exists(ckpt_path):
        start_step = load_checkpoint(ckpt_path, model, optimizer)

    # 训练循环
    for step in range(start_step + 1, steps):
        model.train()
        # 更新学习率
        lr = lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, steps)
        for group in optimizer.param_groups:
            group['lr'] = lr

        # 取数据
        x, y = get_batch(dataset_train, batch_size, context_length, device)

        # 前向传播
        logits = model(x)
        loss = cross_entropy(logits, y)

        # 梯度裁剪，更新参数
        optimizer.zero_grad()
        
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm)
        optimizer.step()

        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                eval_x, eval_y = get_batch(dataset_eval, batch_size, context_length, device)
                eval_logits = model(eval_x)
                eval_loss = cross_entropy(eval_logits, eval_y)
                print(f"step: {step}, lr: {lr}, loss: {eval_loss.item()}")
            model.train()
        if step % ckpt_interval == 0:
            save_checkpoint(model, optimizer, step, ckpt_path)

    if ckpt_path is not None:
        final_step = steps - 1
        save_checkpoint(model, optimizer, final_step, ckpt_path)
        print(f"训练完成，最终模型已保存至 {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM on TinyStories")

    # ================= 1. 路径与数据 =================
    parser.add_argument("--train_data", type=str, default="data/TinyStoriesV2-GPT4-train_ids.npy", 
                        help="Path to training data (.npy)")
    parser.add_argument("--eval_data", type=str, default="data/TinyStoriesV2-GPT4-valid_ids.npy", 
                        help="Path to validation data (.npy)")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/ckpt.pt", 
                        help="Path to save/load checkpoint")

    # ================= 2. 模型超参数 (基线配置) =================
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=512, help="Context length (seq_len)")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer blocks")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344, help="Feed-forward inner dimension")
    parser.add_argument("--theta", type=float, default=10000.0, help="RoPE theta parameter")

    # ================= 3. 训练超参数 =================
    parser.add_argument("--steps", type=int, default=20000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps for LR schedule")
    
    parser.add_argument("--max_lr", type=float, default=1e-3, help="Max learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Min learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_l2_norm", type=float, default=1.0, help="Gradient clipping threshold")

    # ================= 4. 日志与保存频率 =================
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--ckpt_interval", type=int, default=1000, help="Save checkpoint every N steps")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 确保保存 checkpoint 的文件夹存在
    os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
    
    print("========== 开始训练 ==========")
    print(f"模型配置: {args.num_layers} 层, d_model={args.d_model}, {args.num_heads} 头")
    print(f"训练配置: {args.steps} 步, batch_size={args.batch_size}, 序列长度={args.context_length}")
    print(f"数据路径: {args.train_data}")
    print("==============================")

    train(
        train_data_src=args.train_data,
        eval_data_src=args.eval_data,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        lr=args.max_lr, 
        weight_decay=args.weight_decay,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        batch_size=args.batch_size,
        max_l2_norm=args.max_l2_norm,
        ckpt_path=args.ckpt_path,
        ckpt_interval=args.ckpt_interval,
        eval_interval=args.eval_interval
    )