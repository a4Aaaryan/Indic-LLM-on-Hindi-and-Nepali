"""Stage 5: Document-level Train/Validation/Test splitting for Hindi.

Splits the deduplicated Hindi corpus into:
- Train split (default 90%)
- Validation split (default 5%)
- Test split (default 5%)
with fixed random seeds to guarantee reproducibility and prevent data leakage.
"""

import argparse
import sys
import random
import json
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_yaml_config, resolve_path, ensure_dir
from common.utils import setup_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate reproducible train/val/test splits for Hindi.")
    parser.add_argument(
        "--config",
        type=str,
        default="hindi/configs/phase1_config.yaml",
        help="Path to Phase 1 configuration YAML",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    logger = setup_logger("hindi_split")
    
    seed_val = config.get("splitting", {}).get("random_seed", 42)
    set_seed(seed_val)
    # Seed python random explicitly
    random.seed(seed_val)

    # Use interim (deduplicated) directory as input
    interim_dir = resolve_path(config["paths"]["data_interim_dir"])
    splits_dir = ensure_dir(config["paths"]["data_splits_dir"])
    
    train_dir = ensure_dir(splits_dir / "train")
    val_dir = ensure_dir(splits_dir / "val")
    test_dir = ensure_dir(splits_dir / "test")

    logger.info("Hindi splitting pipeline configured.")
    logger.info("Input dir (deduplicated): %s", interim_dir)
    logger.info("Splits directory: %s", splits_dir)
    
    train_ratio = config.get("splitting", {}).get("train_ratio", 0.9)
    val_ratio = config.get("splitting", {}).get("val_ratio", 0.05)
    
    files = list(interim_dir.glob("*.jsonl"))
    if not files:
        logger.warning("No interim data files found in %s", interim_dir)
        return

    stats = {
        "train": {"docs": 0, "words": 0, "sources": defaultdict(lambda: {"docs": 0, "words": 0})},
        "val":   {"docs": 0, "words": 0, "sources": defaultdict(lambda: {"docs": 0, "words": 0})},
        "test":  {"docs": 0, "words": 0, "sources": defaultdict(lambda: {"docs": 0, "words": 0})},
        "total": {"docs": 0, "words": 0}
    }

    for filepath in files:
        source_name = filepath.stem
        logger.info("Splitting %s...", filepath.name)
        
        train_path = train_dir / filepath.name
        val_path = val_dir / filepath.name
        test_path = test_dir / filepath.name
        
        # Reset seed per file to ensure deterministic per-source behavior
        # regardless of file processing order
        random.seed(f"{seed_val}_{source_name}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as fin, \
                 open(train_path, 'w', encoding='utf-8') as ftrain, \
                 open(val_path, 'w', encoding='utf-8') as fval, \
                 open(test_path, 'w', encoding='utf-8') as ftest:
                 
                for line in fin:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                        words = len(text.split())
                        
                        r = random.random()
                        if r < train_ratio:
                            split = "train"
                            ftrain.write(line)
                        elif r < train_ratio + val_ratio:
                            split = "val"
                            fval.write(line)
                        else:
                            split = "test"
                            ftest.write(line)
                            
                        stats[split]["docs"] += 1
                        stats[split]["words"] += words
                        stats[split]["sources"][source_name]["docs"] += 1
                        stats[split]["sources"][source_name]["words"] += words
                        
                        stats["total"]["docs"] += 1
                        stats["total"]["words"] += words
                        
                    except Exception as e:
                        continue
        except Exception as e:
            logger.error("Failed to split %s: %s", filepath.name, e)

    logger.info("--- Splitting Complete ---")
    logger.info("Total Input Docs: %d", stats["total"]["docs"])
    logger.info("Total Input Words: %d", stats["total"]["words"])
    
    for split in ["train", "val", "test"]:
        logger.info("--- SPLIT: %s ---", split.upper())
        logger.info("Total %s docs: %d", split, stats[split]["docs"])
        logger.info("Total %s words: %d", split, stats[split]["words"])
        for src in stats[split]["sources"]:
            logger.info("  %s -> docs: %d, words: %d", src, 
                        stats[split]["sources"][src]["docs"],
                        stats[split]["sources"][src]["words"])

    # Output stats file for the report
    with open(splits_dir / "split_stats.json", "w", encoding="utf-8") as fs:
        json.dump(stats, fs, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
