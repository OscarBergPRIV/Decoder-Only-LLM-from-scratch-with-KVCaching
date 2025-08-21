import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import textwrap
from transformers import GPT2LMHeadModel
import tiktoken


# --- DecoderOnly Model ---
class KVCache(nn.Module):
    def __init__(self, cache_size, num_heads, head_dim, max_batch=1):
        super().__init__()
        self.max_batch = max_batch
        self.cache_size = cache_size
        self.register_buffer("cache_K", torch.zeros((max_batch, num_heads, cache_size, head_dim)), persistent=False)
        self.register_buffer("cache_V", torch.zeros((max_batch, num_heads, cache_size, head_dim)), persistent=False)
        self.index = 0

    def push(self, k, v):
        b, h, seq, d = k.size()
        assert b <= self.max_batch, f"Batch size {b} exceeds max_batch {self.max_batch}"
        assert self.index + seq <= self.cache_size, f"Cache overflow: index {self.index} + seq {seq} > {self.cache_size}"
        self.cache_K[:b, :, self.index:self.index + seq] = k
        self.cache_V[:b, :, self.index:self.index + seq] = v
        self.index += seq
        return self.cache_K[:b, :, :self.index], self.cache_V[:b, :, :self.index]

    def reset(self):
        self.index = 0
        self.cache_K.zero_()
        self.cache_V.zero_()

    def to(self, device):
        self.cache_K = self.cache_K.to(device)
        self.cache_V = self.cache_V.to(device)
        return super().to(device)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, block_size, max_batch=1):
        super().__init__()
        assert dim % num_heads == 0
        self.in_map = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.kvcache = KVCache(block_size, num_heads, self.head_dim, max_batch)

    def forward(self, x, caching=False):
        b, seq, d = x.size()
        q, k, v = self.in_map(x).split(self.dim, dim=2)
        q = q.view(b, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if caching:
            k, v = self.kvcache.push(k, v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, seq, d)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, block_size, max_batch=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.attn = Attention(dim, num_heads, block_size, max_batch)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=True)
        self.mlp = MLP(dim)

    def forward(self, x, caching=False):
        x = x + self.attn(self.ln1(x), caching=caching)
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderOnly(nn.Module):
    def __init__(self, vocab_size=50257, dim=768, num_layers=12, block_size=1024, num_heads=12, max_batch=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_heads = num_heads
        self.max_batch = max_batch
        self.wte = nn.Embedding(vocab_size, dim)
        self.wpe = nn.Embedding(block_size, dim)
        self.layers = nn.ModuleList([DecoderLayer(dim, num_heads, block_size, max_batch) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(dim, elementwise_affine=True)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.wte.weight = self.head.weight

    def forward(self, x, labels=None, caching=False):
        device = x.device
        b, seq = x.size()
        assert seq <= self.block_size, f"Number of tokens in sequence must be <= block_size"
        assert b <= self.max_batch, f"Batch size {b} exceeds max_batch {self.max_batch}"
        start_pos = self.layers[0].attn.kvcache.index if caching else 0
        pos = torch.arange(start_pos, start_pos + seq, dtype=torch.long, device=device)
        pos = pos % self.block_size
        tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        for layer in self.layers:
            x = layer(x, caching=caching)
        x = self.ln(x)
        loss = None
        if labels is not None:
            out = self.head(x)
            loss = F.cross_entropy(out.view(-1, out.size(-1)), labels.view(-1), ignore_index=-1)
        else:
            out = self.head(x[:, [-1], :])
        return out, loss

    def reset_cache(self):
        for layer in self.layers:
            layer.attn.kvcache.reset()

    def to(self, device):
        for layer in self.layers:
            layer.attn.kvcache.to(device)
        return super().to(device)

    def translate_state_dict(self, hf_state_dict):
        sd = self.state_dict()
        sd_keys = list(sd.keys())
        hf_keys = list(hf_state_dict.keys())
        assert len(hf_keys) == len(sd_keys), f"Mismatch: {len(hf_keys)} HuggingFace vs {len(sd_keys)} local"
        transposed = ["attn.in_map.weight", "attn.out.weight", "mlp.fc1.weight", "mlp.fc2.weight"]
        for k_hf, k_local in zip(hf_keys, sd_keys):
            if any(k_local.endswith(w) for w in transposed):
                assert hf_state_dict[k_hf].shape[::-1] == sd[k_local].shape
                with torch.no_grad():
                    sd[k_local].copy_(hf_state_dict[k_hf].t())
            else:
                assert hf_state_dict[k_hf].shape == sd[k_local].shape, f"Shape mismatch: {k_hf}"
                with torch.no_grad():
                    sd[k_local].copy_(hf_state_dict[k_hf])
        self.load_state_dict(sd)
        print("✅ State dict successfully loaded (with Conv1D → Linear transpositions)")

    def benchmark_generate(self, prompt_tokens, max_new_tokens, caching=False,
                           temperature=1.0, top_k=0, top_p=1.0):
        device = prompt_tokens.device
        generated = prompt_tokens.clone()
        prefill_time = None
        decode_time = None
        tok_per_sec = 0.0
        eos_token_id = 50256

        def sample_next_token(logits, temperature, top_k, top_p):
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            # Top-k filtering
            if top_k > 0:
                top_k = min(top_k, probs.size(-1))
                values, indices = torch.topk(probs, top_k)
                probs[probs < values[:, [-1]]] = 0
                probs = probs / probs.sum()

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumulative_probs > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_probs[cutoff] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                return sorted_indices.gather(-1, next_token)

            return torch.multinomial(probs, num_samples=1)  # Shape: [batch_size, 1]

        if caching:
            self.reset_cache()
            start_time = time.perf_counter()
            with torch.no_grad():
                logits, _ = self(generated, caching=True)
            if device == 'cuda':
                torch.cuda.synchronize()
            prefill_time = time.perf_counter() - start_time

            for _ in range(max_new_tokens):
                next_token_id = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
                generated = torch.cat([generated, next_token_id], dim=1)
                if next_token_id.item() == eos_token_id:
                    break
                start = time.perf_counter()
                with torch.no_grad():
                    logits, _ = self(generated[:, -1:], caching=True)
                if device == 'cuda':
                    torch.cuda.synchronize()
                decode_time = (decode_time or 0) + (time.perf_counter() - start)
        else:
            self.reset_cache()
            start_time = time.perf_counter()
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits, _ = self(generated, caching=False)
                next_token_id = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
                generated = torch.cat([generated, next_token_id], dim=1)
                if next_token_id.item() == eos_token_id:
                    break
            if device == 'cuda':
                torch.cuda.synchronize()
            decode_time = time.perf_counter() - start_time

        total_new_tokens = generated.size(1) - prompt_tokens.size(1)
        tok_per_sec = total_new_tokens / decode_time if decode_time and decode_time > 0 else 0.0
        text = enc.decode(generated[0].cpu().tolist())
        return text, tok_per_sec, prefill_time, decode_time


# --- CLI helper ---
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with DecoderOnly GPT-2")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation")
    parser.add_argument("--kvcaching", action="store_true", help="Enable key-value caching during generation")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu",
                        help="Device to run inference on (cpu or cuda)")
    parser.add_argument("--max_tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling (1.0 = disabled)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Device handling
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load HuggingFace GPT-2 model weights
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_state_dict = hf_model.state_dict()

    # Initialize local model
    model = DecoderOnly(max_batch=1)
    model.translate_state_dict(hf_state_dict)
    model.eval().to(device)

    # Tokenizer
    global enc
    enc = tiktoken.get_encoding("gpt2")

    # Encode prompt
    tokens = enc.encode(args.prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Run generation
    text, tok_per_sec, prefill_time, decode_time = model.benchmark_generate(
        tokens,
        max_new_tokens=args.max_tokens,
        caching=args.kvcaching,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    # Output results
    clean_text = ' '.join(text.strip().splitlines())
    wrapped_text = '\n'.join(textwrap.wrap(clean_text, width=100))
    print("\n" + "=" * 80)
    print("Generated text:")
    print("-" * 80)
    print(wrapped_text)
    print("-" * 80)
    if prefill_time is not None:
        print(f"Prefill time: {prefill_time:.4f} seconds")
    print(f"Autoregressive tokens per second: {tok_per_sec:.2f}")
    print(f"Decode time: {decode_time:.4f} seconds")
    print("=" * 80)
