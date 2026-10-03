"""
Phase 3 Fine-tuning Script

Loads pretrained Phase 2 model and fine-tunes on Phase 3 JSONL dataset.
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from phase2.src.config import LMConfig
from phase2.src.model import DecoderOnlyLM
from phase2.src.train_config import TrainConfig
from phase2.src.trainer import train, load_checkpoint
from phase3.src.dataset import Phase3ReasoningDataset

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Fine-tuning")
    parser.add_argument('--config', type=str, required=True, help="Path to finetune_config.yaml")
    parser.add_argument('--resume_from', type=str, default=None, help="Path to Phase 3 checkpoint to resume from")
    parser.add_argument('--dry-run', type=int, default=0, help="If >0, runs N steps with random weights for testing")
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
        
    # Extract Phase 2 Model architecture parameters
    model_cfg = LMConfig(
        vocab_size=cfg_dict.get('vocab_size', 10000),
        d_model=cfg_dict.get('d_model', 448),
        n_layers=cfg_dict.get('n_layers', 8),
        n_heads=cfg_dict.get('n_heads', 7),
        d_ff=cfg_dict.get('d_ff', 2048),
        max_seq_len=cfg_dict.get('max_seq_len', 512),
        dropout=cfg_dict.get('dropout', 0.1),
        bias=cfg_dict.get('bias', True)
    )
    
    # Extract TrainConfig parameters
    train_cfg = TrainConfig(
        language=cfg_dict.get('language', 'hindi'),
        seed=cfg_dict.get('seed', 42),
        context_length=cfg_dict.get('context_length', 512),
        batch_size=cfg_dict.get('batch_size', 16),
        grad_accum_steps=cfg_dict.get('grad_accum_steps', 1),
        learning_rate=float(cfg_dict.get('learning_rate', 5e-5)),
        min_lr=float(cfg_dict.get('min_lr', 5e-6)),
        weight_decay=float(cfg_dict.get('weight_decay', 0.1)),
        max_steps=cfg_dict.get('max_steps', 3000),
        warmup_steps=cfg_dict.get('warmup_steps', 300),
        grad_clip=float(cfg_dict.get('grad_clip', 1.0)),
        use_amp=cfg_dict.get('use_amp', True),
        output_dir=Path(cfg_dict.get('output_dir', 'results/phase3_finetune/')),
        checkpoint_interval=cfg_dict.get('checkpoint_interval', 1000),
        log_interval=cfg_dict.get('log_interval', 10),
        eval_interval=cfg_dict.get('eval_interval', 500),
        eval_batches=cfg_dict.get('eval_batches', 50)
    )
    
    # Checkpoint path
    pretrained_ckpt = cfg_dict.get('pretrained_checkpoint')
    tokenizer_path = cfg_dict.get('tokenizer_path')
    
    if args.dry_run > 0:
        print("\n" + "="*50)
        print("DRY RUN / RANDOM INITIALIZATION — NOT A PRETRAINED BASELINE")
        print("="*50 + "\n")
        train_cfg.max_steps = args.dry_run
        train_cfg.eval_interval = 1
        train_cfg.eval_batches = 1
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Model
    model = DecoderOnlyLM(model_cfg)
    
    if args.dry_run == 0 and args.resume_from is None:
        if not os.path.exists(pretrained_ckpt):
            print(f"[ERROR] Pretrained Phase 2 checkpoint not found at {pretrained_ckpt}")
            print("Cannot start Phase 3 fine-tuning without the pretrained weights.")
            sys.exit(1)
        print(f"Loading pretrained Phase 2 checkpoint from {pretrained_ckpt}")
        # The phase 2 load_checkpoint function expects state_dict in payload['model_state']
        # which it assigns to the model.
        payload = load_checkpoint(Path(pretrained_ckpt), model, device=torch.device('cpu'), restore_rng=False)
        print("Pretrained checkpoint loaded.")
        
    # Load Datasets
    print(f"Loading train data from {cfg_dict['train_data']}")
    train_dataset = Phase3ReasoningDataset(cfg_dict['train_data'], tokenizer_path, max_length=train_cfg.context_length)
    
    print(f"Loading val data from {cfg_dict['val_data']}")
    val_dataset = Phase3ReasoningDataset(cfg_dict['val_data'], tokenizer_path, max_length=train_cfg.context_length)
    
    # Run Training
    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=train_cfg,
        model_cfg=model_cfg,
        resume_from=Path(args.resume_from) if args.resume_from else None,
        device=device
    )
    
if __name__ == '__main__':
    main()
