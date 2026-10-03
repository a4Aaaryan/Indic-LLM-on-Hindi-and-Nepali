"""
Phase 2 — Stage 6: Training Infrastructure

Implements:
  - seed_everything
  - build_optimizer
  - build_scheduler  (linear warmup + cosine decay)
  - compute_loss / perplexity
  - training_step
  - evaluate
  - save_checkpoint / load_checkpoint
  - JsonlLogger
  - train  (main training loop)
"""

import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .train_config import TrainConfig


# ---------------------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch CPU, and CUDA (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 2. Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """
    Build AdamW with separate weight-decay groups.

    Parameters without meaningful weight decay (biases, LayerNorm weights)
    are placed in a no-decay group.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude 1-D parameters (biases, LN weights/biases) from WD
        if param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
    )
    return optimizer


# ---------------------------------------------------------------------------
# 3. LR Scheduler  (linear warmup + cosine decay to min_lr)
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainConfig) -> float:
    """
    Return the learning rate for a given optimiser step.

    Schedule:
      [0, warmup_steps)          → linear 0 → lr
      [warmup_steps, max_steps]  → cosine decay lr → min_lr
      > max_steps                → min_lr (clamped)
    """
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step / max(1, cfg.warmup_steps))

    if step >= cfg.max_steps:
        return cfg.min_lr

    # Cosine decay phase
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + cosine * (cfg.learning_rate - cfg.min_lr)


def build_scheduler(optimizer: torch.optim.AdamW, cfg: TrainConfig):
    """
    Return a LambdaLR scheduler wrapping the get_lr function.

    scheduler.step() must be called AFTER optimizer.step().
    The lambda receives the current step count (maintained internally by PyTorch).
    """
    # LambdaLR multiplies base_lr by the returned factor.
    # Our get_lr already returns the absolute LR, so we divide by base lr.
    def lr_lambda(step: int) -> float:
        return get_lr(step, cfg) / cfg.learning_rate

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


# ---------------------------------------------------------------------------
# 4. Loss / metrics
# ---------------------------------------------------------------------------

def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss for language modelling.

    logits:  [B, T, vocab_size]
    targets: [B, T]              (int64)

    Returns scalar loss.
    """
    B, T, V = logits.shape
    loss = nn.functional.cross_entropy(
        logits.view(B * T, V),
        targets.view(B * T).long(),
    )
    return loss


def perplexity(loss: float) -> float:
    """
    Perplexity = exp(loss).  Clamps to avoid inf for very large losses.
    """
    try:
        return math.exp(min(loss, 20.0))   # exp(20) ≈ 485M — sensible cap
    except OverflowError:
        return float("inf")


# ---------------------------------------------------------------------------
# 5. AMP helper
# ---------------------------------------------------------------------------

def _amp_available(cfg: TrainConfig, device: torch.device) -> bool:
    """True only when AMP is requested AND a CUDA device is active."""
    return cfg.use_amp and device.type == "cuda"


# ---------------------------------------------------------------------------
# 6. Training step
# ---------------------------------------------------------------------------

def training_step(
    model: nn.Module,
    batch_iter,
    optimizer: torch.optim.AdamW,
    scheduler,
    scaler: "torch.cuda.amp.GradScaler",
    cfg: TrainConfig,
    device: torch.device,
    use_amp: bool,
) -> dict:
    """
    Execute one *optimiser* step, which may span cfg.grad_accum_steps micro-batches.

    Returns a dict with:
        loss        – mean loss over accum steps
        grad_norm   – gradient global norm (before clipping), or None
        tokens      – number of tokens processed in this optimiser step
        lr          – learning rate after this step
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    accum_loss = 0.0
    total_tokens = 0

    for micro_step in range(cfg.grad_accum_steps):
        x, y = next(batch_iter)
        x = x.to(device)
        y = y.to(device)
        total_tokens += x.numel()

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            # Scale loss by grad_accum_steps so gradients average over all microbatches
            loss = compute_loss(logits, y) / cfg.grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += loss.item()

    # Gradient clipping
    grad_norm = None
    if cfg.grad_clip > 0.0:
        if use_amp:
            scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        ).item()

    # Optimiser step
    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    # Scheduler step (AFTER optimiser step)
    scheduler.step()

    return {
        "loss": accum_loss,          # already averaged by /grad_accum_steps
        "grad_norm": grad_norm,
        "tokens": total_tokens,
        "lr": scheduler.get_last_lr()[0],
    }


# ---------------------------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Dataset,
    cfg: TrainConfig,
    device: torch.device,
    use_amp: bool,
) -> dict:
    """
    Evaluate model on up to cfg.eval_batches random batches from dataset.

    Saves and restores model training/eval state.

    Returns dict with:
        val_loss  – mean cross-entropy
        val_ppl   – perplexity
    """
    was_training = model.training
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = compute_loss(logits, y)
        total_loss += loss.item()
        n_batches += 1
        if n_batches >= cfg.eval_batches:
            break

    model.train(was_training)

    mean_loss = total_loss / max(n_batches, 1)
    return {
        "val_loss": mean_loss,
        "val_ppl": perplexity(mean_loss),
    }


# ---------------------------------------------------------------------------
# 8. Random-state helpers
# ---------------------------------------------------------------------------

def get_rng_state() -> dict:
    state = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and "cuda_rng" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


# ---------------------------------------------------------------------------
# 9. Checkpointing
# ---------------------------------------------------------------------------

CHECKPOINT_VERSION = 1


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    scheduler,
    cfg: TrainConfig,
    model_cfg,
    best_val_loss: Optional[float] = None,
    rng_state: Optional[dict] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Path:
    """
    Atomically save a checkpoint.

    Writes to a temp file, then renames to avoid corruption on interruption.

    Saves:
    - model state_dict
    - optimizer state_dict
    - scheduler state_dict
    - step
    - TrainConfig (as dict)
    - LMConfig (as dict)
    - best_val_loss
    - rng_state
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": CHECKPOINT_VERSION,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_cfg": vars(cfg).copy(),
        "model_cfg": vars(model_cfg),
        "best_val_loss": best_val_loss,
        "rng_state": rng_state,
    }
    
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()

    # Serialise Path → str so torch.save can handle it
    payload["train_cfg"]["output_dir"] = str(payload["train_cfg"]["output_dir"])

    step_name = f"step_{step:08d}.pt"
    final_path = output_dir / step_name

    # Write to temp file, then rename for atomicity
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        os.close(fd)
        torch.save(payload, tmp_path)
        shutil.move(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    # Update latest symlink / pointer file (plain text, always safe)
    latest_path = output_dir / "latest.txt"
    fd, tmp_latest = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        os.close(fd)
        with open(tmp_latest, "w") as f:
            f.write(step_name)
        shutil.move(tmp_latest, latest_path)
    except Exception:
        if os.path.exists(tmp_latest):
            os.remove(tmp_latest)
        raise

    return final_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.AdamW] = None,
    scheduler=None,
    scaler=None,
    device: Optional[torch.device] = None,
    restore_rng: bool = True,
) -> dict:
    """
    Load a checkpoint and restore model (and optionally optimizer/scheduler) state.

    Returns the payload dict so callers can access step, config, etc.

    Raises ValueError for version mismatches.
    """
    if device is None:
        device = torch.device("cpu")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version mismatch. "
            f"Expected {CHECKPOINT_VERSION}, got {payload.get('version')}"
        )

    model.load_state_dict(payload["model_state"])

    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])

    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
        
    if scaler is not None and "scaler_state" in payload:
        scaler.load_state_dict(payload["scaler_state"])

    if restore_rng and payload.get("rng_state") is not None:
        set_rng_state(payload["rng_state"])

    return payload


def get_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Return the path pointed to by latest.txt, or None if not found."""
    latest_file = Path(output_dir) / "latest.txt"
    if not latest_file.exists():
        return None
    step_name = latest_file.read_text().strip()
    ckpt = Path(output_dir) / step_name
    return ckpt if ckpt.exists() else None


# ---------------------------------------------------------------------------
# 10. JSONL Logger
# ---------------------------------------------------------------------------

class JsonlLogger:
    """Append-only JSONL logger for training and evaluation metrics."""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        record = dict(record)
        record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 11. DataLoader factory
# ---------------------------------------------------------------------------

class _IterableWrapper:
    """Infinite iterator over a random-sampling DataLoader."""

    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int):
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._loader = self._make_loader()
        self._iter = iter(self._loader)

    def _make_loader(self):
        return DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            drop_last=True,
            pin_memory=False,
        )

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._make_loader())
            return next(self._iter)


# ---------------------------------------------------------------------------
# 12. Main training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    cfg: TrainConfig,
    model_cfg,
    resume_from: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Main training loop.

    Args:
        model:         DecoderOnlyLM (or any nn.Module with [B,T] → [B,T,V] forward)
        train_dataset: TokenDataset for training split
        val_dataset:   TokenDataset for validation split
        cfg:           TrainConfig
        model_cfg:     LMConfig (saved into checkpoints)
        resume_from:   Optional path to a checkpoint to resume from
        device:        torch.device; defaults to CUDA if available, else CPU
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    seed_everything(cfg.seed)

    use_amp = _amp_available(cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    start_step = 0
    best_val_loss = float("inf")

    # Resume from checkpoint
    if resume_from is not None:
        payload = load_checkpoint(
            resume_from, model, optimizer, scheduler, scaler, device, restore_rng=True
        )
        start_step = payload["step"]
        best_val_loss = payload.get("best_val_loss") or float("inf")
        print(f"Resumed from step {start_step}")

    # Advance scheduler to match resumed step without modifying parameters
    # (LambdaLR state is already restored from checkpoint scheduler_state_dict)

    logger = JsonlLogger(cfg.output_dir / "train_log.jsonl")
    batch_iter = _IterableWrapper(train_dataset, cfg.batch_size, cfg.num_workers)

    total_tokens = 0
    t_start = time.time()

    print(
        f"Training {cfg.language} | device={device} | amp={use_amp} | "
        f"steps={cfg.max_steps} | batch={cfg.batch_size} | accum={cfg.grad_accum_steps}"
    )

    for step in range(start_step, cfg.max_steps):
        step_metrics = training_step(
            model, batch_iter, optimizer, scheduler, scaler, cfg, device, use_amp
        )
        total_tokens += step_metrics["tokens"]
        current_step = step + 1

        # ---- Logging ----
        if current_step % cfg.log_interval == 0:
            elapsed = time.time() - t_start
            record = {
                "type": "train",
                "step": current_step,
                "loss": round(step_metrics["loss"], 6),
                "lr": step_metrics["lr"],
                "grad_norm": step_metrics["grad_norm"],
                "tokens": total_tokens,
                "elapsed_s": round(elapsed, 2),
            }
            logger.log(record)
            print(
                f"step {current_step:6d} | loss {step_metrics['loss']:.4f} "
                f"| lr {step_metrics['lr']:.2e} | tokens {total_tokens:,}"
            )

        # ---- Evaluation ----
        if current_step % cfg.eval_interval == 0:
            eval_metrics = evaluate(model, val_dataset, cfg, device, use_amp)
            record = {
                "type": "eval",
                "step": current_step,
                "val_loss": round(eval_metrics["val_loss"], 6),
                "val_ppl": round(eval_metrics["val_ppl"], 4),
            }
            logger.log(record)
            print(
                f"  [eval] step {current_step} | val_loss {eval_metrics['val_loss']:.4f} "
                f"| ppl {eval_metrics['val_ppl']:.2f}"
            )
            if eval_metrics["val_loss"] < best_val_loss:
                best_val_loss = eval_metrics["val_loss"]

        # ---- Checkpointing ----
        if current_step % cfg.checkpoint_interval == 0:
            rng_state = get_rng_state()
            ckpt_path = save_checkpoint(
                cfg.output_dir,
                current_step,
                model,
                optimizer,
                scheduler,
                cfg,
                model_cfg,
                best_val_loss,
                rng_state,
                scaler=scaler,
            )
            print(f"  [ckpt] saved {ckpt_path}")

    print(f"Training complete. Total tokens: {total_tokens:,}")
