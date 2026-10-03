from dataclasses import dataclass

@dataclass
class LMConfig:
    vocab_size: int = 10000
    d_model: int = 448
    n_layers: int = 8
    n_heads: int = 7
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = True
