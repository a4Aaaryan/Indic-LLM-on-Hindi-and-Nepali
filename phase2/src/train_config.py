"""
Phase 2 — Stage 6: Training Configuration

Separate from the model architecture (LMConfig) so that hyperparameters
can be tuned without touching the model definition.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    # -----------------------------------------------------------------------
    # Identity
    # -----------------------------------------------------------------------
    language: str = "hindi"          # "hindi" or "nepali"
    run_name: str = "run"            # used for log/checkpoint naming

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------
    seed: int = 42

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    context_length: int = 512
    num_workers: int = 0             # DataLoader workers (0 = main process)

    # -----------------------------------------------------------------------
    # Batch / accumulation
    # -----------------------------------------------------------------------
    batch_size: int = 16             # micro-batch size per step
    grad_accum_steps: int = 1        # effective_batch = batch_size * grad_accum_steps

    # -----------------------------------------------------------------------
    # Optimiser
    # -----------------------------------------------------------------------
    learning_rate: float = 3e-4
    min_lr: float = 3e-5             # cosine decay floor
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # -----------------------------------------------------------------------
    # Schedule
    # -----------------------------------------------------------------------
    max_steps: int = 100_000         # total optimiser steps
    warmup_steps: int = 2_000        # linear warmup duration

    # -----------------------------------------------------------------------
    # Regularisation / gradient
    # -----------------------------------------------------------------------
    grad_clip: float = 1.0          # 0.0 disables clipping

    # -----------------------------------------------------------------------
    # Mixed precision
    # -----------------------------------------------------------------------
    use_amp: bool = True             # disabled automatically on CPU

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------
    output_dir: Path = Path("checkpoints")
    checkpoint_interval: int = 1_000   # save every N optimiser steps
    keep_latest_n: int = 3            # how many step-checkpoints to retain

    # -----------------------------------------------------------------------
    # Logging / evaluation
    # -----------------------------------------------------------------------
    log_interval: int = 10           # print/log training metrics every N steps
    eval_interval: int = 500         # run validation every N steps
    eval_batches: int = 50           # number of batches for each validation pass

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)

        if self.language not in ("hindi", "nepali"):
            raise ValueError(f"language must be 'hindi' or 'nepali', got {self.language!r}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be < max_steps")
        if not (0.0 < self.learning_rate):
            raise ValueError("learning_rate must be positive")
        if not (0.0 <= self.min_lr < self.learning_rate):
            raise ValueError("min_lr must satisfy 0 <= min_lr < learning_rate")
        if not (0.0 <= self.grad_clip):
            raise ValueError("grad_clip must be >= 0")
