"""Stage 5b: Deterministic Downsampling of HPLT for Assignment Compliance.

This script downsamples the `indic_hplt_v2_ne.jsonl` files from the existing
`splits/` directory to meet an exact word target (118,000,000 words) while
preserving the 90/5/5 train/val/test structure.

It uses a two-pass byte-offset sampling method to guarantee a uniform random
selection of documents across the entire file without excessive memory usage.
Manual files are hardlinked/copied untouched.
"""

import argparse
import sys
import random
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_yaml_config, resolve_path, ensure_dir
from common.utils import setup_logger, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Downsample HPLT to assignment target.")
    parser.add_argument("--config", default="nepali/configs/phase1_config.yaml")
    return parser.parse_args()

def process_manual_files(split_in_dir, split_out_dir, logger):
    """Copy manual files to the final directory unmodified."""
    manual_files = [f for f in split_in_dir.glob("*.jsonl") if "hplt" not in f.name.lower()]
    for f in manual_files:
        out_path = split_out_dir / f.name
        if not out_path.exists():
            shutil.copy2(f, out_path)
            logger.info("Copied manual file %s", f.name)

def downsample_file(in_path, out_path, target_words, seed, logger):
    """Two-pass deterministic sampling using byte offsets."""
    if not in_path.exists():
        logger.warning("File %s does not exist.", in_path)
        return

    logger.info("Pass 1: Indexing %s...", in_path.name)
    offsets = []

    with open(in_path, 'rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                # Quick word count without full json parse for speed
                text_part = json.loads(line.decode('utf-8')).get("text", "")
                words = len(text_part.split())
                if words > 0:
                    offsets.append((offset, words))
            except Exception:
                pass

    logger.info("Indexed %d documents. Total words available: %d",
                len(offsets), sum(w for _, w in offsets))

    logger.info("Shuffling indices with seed %d...", seed)
    rng = random.Random(seed)
    rng.shuffle(offsets)

    logger.info("Pass 2: Accumulating %d words...", target_words)
    selected_offsets = []
    accumulated_words = 0

    for offset, words in offsets:
        selected_offsets.append(offset)
        accumulated_words += words
        if accumulated_words >= target_words:
            break

    # Sort offsets so we read sequentially from the disk in pass 3
    selected_offsets.sort()

    logger.info("Pass 3: Writing %d selected documents (Target: %d, Actual: %d words)",
                len(selected_offsets), target_words, accumulated_words)

    with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
        for offset in selected_offsets:
            fin.seek(offset)
            fout.write(fin.readline())

def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    logger = setup_logger("nepali_downsample")

    seed = 42
    set_seed(seed)

    splits_dir = resolve_path(config["paths"]["data_splits_dir"])
    final_dir = ensure_dir(resolve_path("nepali/data/final"))

    # 118M total HPLT words * split ratio
    targets = {
        "train": int(118_000_000 * 0.90), # 106.2M
        "val": int(118_000_000 * 0.05),   # 5.9M
        "test": int(118_000_000 * 0.05)   # 5.9M
    }

    for split_name, target_words in targets.items():
        logger.info("--- Processing Split: %s ---", split_name.upper())
        split_in = splits_dir / split_name
        split_out = ensure_dir(final_dir / split_name)

        process_manual_files(split_in, split_out, logger)

        hplt_in = split_in / "indic_hplt_v2_ne.jsonl"
        hplt_out = split_out / "indic_hplt_v2_ne.jsonl"

        downsample_file(hplt_in, hplt_out, target_words, seed, logger)

    logger.info("Downsampling complete. Final corpus is in %s", final_dir)

if __name__ == "__main__":
    main()
