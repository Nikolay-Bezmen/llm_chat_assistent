import torch
import torch.nn as nn

from typing import Optional, Tuple
from torch.nn import functional as F

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=512, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=256):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
    
    
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    b = [position_ids.shape, cos.shape, q.shape, k.shape]
    cos = cos[position_ids.int()] # [bs, seq_len, dim]

    sin = sin[position_ids.int()]  # [bs, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryEmbedding(head_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3, x.shape
        
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_emb(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd: int, num_heads: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(
            n_embd=n_embd,
            num_heads=n_head,
            head_size=head_size,
            block_size=block_size,
            dropout=dropout
        )
        self.feed_forward = FeedFoward(n_embd, dropout)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm_1(x + self.self_attention(x))
        x = self.layer_norm_2(x + self.feed_forward(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        block_size: int,
        n_layer: int,
        dropout: float,
        device: str,
        ignore_index: int = -100
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.block_size = block_size
        self.device = device

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.final_linear_layer = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_tokens.shape

        token_embedding = self.token_embedding_table(input_tokens)
        x = token_embedding
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.final_linear_layer(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(
                logits, targets, ignore_index=self.ignore_index)

        return logits, loss

    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates new tokens from the model.

        Args:
            input_tokens: The initial input tokens.
            max_new_tokens: The maximum number of tokens to generate.

        Returns:
            The generated tokens.
        """
        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -self.block_size:]
            logits, _ = self(cropped_input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)
        return input_tokens

    def advanced_generation(
        self,
        input_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generates new tokens from the model.

        Args:
            input_tokens: The initial input tokens.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: Controls randomness (higher = more random).
            top_k: Limits generation to the top-k most likely tokens.
            top_p: Limits generation to tokens with cumulative probability <= top_p.

        Returns:
            The generated tokens.
        """
        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -self.block_size:]
            logits, _ = self(cropped_input)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[...,
                                         1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(logits).scatter_(
                    1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            idx_next = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)

        return input_tokens