"""Stage 4: Cross-source and within-source deduplication for Hindi.

Performs:
- Exact document deduplication (SHA-256 hash)
- Preservation of surviving document provenance
- Removal logging and metrics tracking
"""

import argparse
import hashlib
import sys
import json
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_yaml_config, resolve_path, ensure_dir
from common.utils import setup_logger


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text for exact deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def parse_args():
    parser = argparse.ArgumentParser(description="Deduplicate Hindi corpus.")
    parser.add_argument(
        "--config",
        type=str,
        default="hindi/configs/phase1_config.yaml",
        help="Path to Phase 1 configuration YAML",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Run only on a sample of N documents per file for testing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    logger = setup_logger("hindi_dedup")

    cleaned_dir = resolve_path(config["paths"]["data_cleaned_dir"])
    interim_dir = ensure_dir(config["paths"]["data_interim_dir"])
    
    logger.info("Hindi deduplication configured for corpus in: %s", cleaned_dir)
    logger.info("Outputting deduplicated files to: %s", interim_dir)

    files = list(cleaned_dir.glob("*.jsonl"))
    if not files:
        logger.warning("No cleaned data files found in %s", cleaned_dir)
        return

    # Global set to track seen hashes
    seen_hashes = set()
    
    # Statistics tracking
    stats = {
        "input_docs": 0,
        "unique_docs": 0,
        "global_duplicates": 0,
        "removed_per_source": defaultdict(int),
        "input_words": 0,
        "output_words": 0,
        "input_chars": 0,
        "output_chars": 0
    }

    for filepath in files:
        source_name = filepath.stem
        out_path = interim_dir / filepath.name
        
        logger.info("Processing %s...", filepath.name)
        
        source_in_docs = 0
        source_out_docs = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as fin, \
                 open(out_path, 'w', encoding='utf-8') as fout:
                 
                for line in fin:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                        if not text: continue
                        
                        words = len(text.split())
                        chars = len(text)
                        
                        stats["input_docs"] += 1
                        stats["input_words"] += words
                        stats["input_chars"] += chars
                        source_in_docs += 1
                        
                        h = compute_hash(text)
                        
                        if h in seen_hashes:
                            stats["global_duplicates"] += 1
                            stats["removed_per_source"][source_name] += 1
                            continue
                            
                        seen_hashes.add(h)
                        
                        stats["unique_docs"] += 1
                        stats["output_words"] += words
                        stats["output_chars"] += chars
                        source_out_docs += 1
                        
                        fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                        
                        if stats["input_docs"] % 100000 == 0:
                            logger.info("  Processed %d documents globally so far...", stats["input_docs"])
                            
                        if args.sample and source_in_docs >= args.sample:
                            break
                            
                    except Exception as e:
                        continue
        except Exception as e:
            logger.error("Failed to process %s: %s", filepath.name, e)
            
        logger.info("  Source %s complete: %d in -> %d out (%d removed)", 
                    source_name, source_in_docs, source_out_docs, stats["removed_per_source"][source_name])

    logger.info("--- Deduplication Complete ---")
    logger.info("Total Input Docs: %d", stats["input_docs"])
    logger.info("Total Unique Docs: %d", stats["unique_docs"])
    logger.info("Total Global Duplicates Removed: %d", stats["global_duplicates"])
    for src, count in stats["removed_per_source"].items():
        logger.info("  Removed from %s: %d", src, count)
        
    logger.info("Input Words: %d | Output Words: %d", stats["input_words"], stats["output_words"])
    logger.info("Input Chars: %d | Output Chars: %d", stats["input_chars"], stats["output_chars"])


if __name__ == "__main__":
    main()
