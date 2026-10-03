"""
Production runner for Phase 2 language model training.
"""

import argparse
import sys
from pathlib import Path
import torch

from src.config import LMConfig
from src.model import DecoderOnlyLM
from src.data import TokenDataset
from src.train_config import TrainConfig
import src.trainer as trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 Training Runner")
    parser.add_argument("--language", choices=["hindi", "nepali"], required=True, help="Language to train")
    parser.add_argument("--train-bin", type=Path, required=True, help="Path to the training .bin file")
    parser.add_argument("--val-bin", type=Path, required=True, help="Path to the validation .bin file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save checkpoints and logs")
    
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in output-dir")
    
    # Overrides
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per step")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--max-steps", type=int, default=100000, help="Total number of optimizer steps")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Number of linear warmup steps")
    parser.add_argument("--eval-interval", type=int, default=500, help="Steps between evaluations")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Steps between checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Fail clearly if required .bin files are missing
    if not args.train_bin.exists():
        print(f"Error: Training file not found at {args.train_bin}", file=sys.stderr)
        sys.exit(1)
    if not args.val_bin.exists():
        print(f"Error: Validation file not found at {args.val_bin}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve locked architecture
    lm_cfg = LMConfig()
    
    # Construct train config
    cfg = TrainConfig(
        language=args.language,
        run_name=f"run_{args.language}",
        context_length=lm_cfg.max_seq_len,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print(f"Initializing TokenDatasets (context_length={cfg.context_length})...")
    train_ds = TokenDataset(args.train_bin, context_length=cfg.context_length)
    val_ds = TokenDataset(args.val_bin, context_length=cfg.context_length)

    print("Instantiating model...")
    model = DecoderOnlyLM(lm_cfg)
    
    resume_ckpt = None
    if args.resume:
        latest = trainer.get_latest_checkpoint(cfg.output_dir)
        if latest is not None:
            resume_ckpt = latest
            print(f"Will resume from checkpoint: {resume_ckpt}")
        else:
            print("No latest checkpoint found to resume from, starting fresh.")

    print("Starting training...")
    trainer.train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg=cfg,
        model_cfg=lm_cfg,
        resume_from=resume_ckpt
    )

if __name__ == "__main__":
    main()
