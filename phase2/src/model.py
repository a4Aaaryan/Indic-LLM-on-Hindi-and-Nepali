import math
import torch
import torch.nn as nn
from .config import LMConfig

class InputEmbeddings(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input_ids):
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds maximum sequence length {self.max_seq_len}")
            
        device = input_ids.device
        position_ids = torch.arange(0, T, dtype=torch.long, device=device)
        
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError(f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})")
            
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.max_seq_len = config.max_seq_len
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask register_buffer ensures it moves to correct device
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
            1, 1, config.max_seq_len, config.max_seq_len
        )
        self.register_buffer("bias_mask", mask)
        
    def forward(self, x, return_attention=False):
        B, T, C = x.size() # B, T, d_model
        
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds maximum sequence length {self.max_seq_len}")
            
        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Causal mask applied BEFORE softmax
        att = att.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax
        att = torch.nn.functional.softmax(att, dim=-1)
        
        # Attention probabilities dropout
        att_dropped = self.attn_dropout(att)
        
        # Multiply with values
        y = att_dropped @ v
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.out_proj(y))
        
        if return_attention:
            return y, att
        return y


class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=config.bias),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x, return_attention=False):
        # Pre-LayerNorm: x = x + Attention(LayerNorm(x))
        if return_attention:
            attn_out, att_weights = self.attn(self.ln_1(x), return_attention=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.ln_1(x))

        # Pre-LayerNorm: x = x + FFN(LayerNorm(x))
        x = x + self.ffn(self.ln_2(x))

        if return_attention:
            return x, att_weights
        return x


class DecoderOnlyLM(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config

        # Input representation
        self.embedding = InputEmbeddings(config)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.d_model)

        # LM head: weight is tied to the token embedding, but has its own bias
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=config.bias)

        # Weight tying: LM head weight IS the token embedding weight
        self.lm_head.weight = self.embedding.token_embedding.weight

    def forward(self, input_ids, return_attention=False):
        # input_ids: [B, T]
        x = self.embedding(input_ids)

        attentions = [] if return_attention else None

        for block in self.blocks:
            if return_attention:
                x, att = block(x, return_attention=True)
                attentions.append(att)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        if return_attention:
            return logits, attentions
        return logits
