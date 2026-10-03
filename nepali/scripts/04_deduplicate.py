"""Stage 4: Cross-source and within-source exact deduplication for Nepali.

Performs:
- Exact document deduplication using global SHA-256 hash.
- Strict deterministic file/source processing order to prioritize manual corpora.
- Preservation of surviving document provenance.
- Detailed tracking of cross-source duplicates.
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
from common.utils import setup_logger, save_json

FILE_PRIORITY = [
    "nepali_wikipedia.jsonl",
    "nepali_wikisource.jsonl",
    "nepali_news_scraped.jsonl",
    "indic_hplt_v2_ne.jsonl"
]

def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of text for exact deduplication."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

def parse_args():
    parser = argparse.ArgumentParser(description="Deduplicate Nepali corpus.")
    parser.add_argument(
        "--config",
        type=str,
        default="nepali/configs/phase1_config.yaml",
        help="Path to Phase 1 configuration YAML",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    logger = setup_logger("nepali_dedup")

    cleaned_dir = resolve_path(config["paths"]["data_cleaned_dir"])
    interim_dir = ensure_dir(config["paths"]["data_interim_dir"])
    stats_dir = ensure_dir(config["paths"]["statistics_dir"])

    logger.info("Nepali deduplication configured for corpus in: %s", cleaned_dir)
    logger.info("Outputting deduplicated files to: %s", interim_dir)

    all_files = list(cleaned_dir.glob("*.jsonl"))
    if not all_files:
        logger.warning("No cleaned data files found in %s", cleaned_dir)
        return

    # Deterministic sorting
    ordered_files = []
    for pref in FILE_PRIORITY:
        for f in all_files:
            if f.name == pref:
                ordered_files.append(f)
    for f in sorted(all_files, key=lambda x: x.name):
        if f not in ordered_files:
            ordered_files.append(f)

    logger.info("Deterministic File Processing Order:")
    for i, f in enumerate(ordered_files, 1):
        logger.info("  %d. %s", i, f.name)

    # Dictionary to track seen hashes to their original source_name
    seen_hashes = {}

    # Statistics tracking
    stats = {
        "global": {
            "input_docs": 0,
            "retained_docs": 0,
            "duplicate_docs": 0,
            "duplicate_percentage": 0.0,
            "input_words": 0,
            "retained_words": 0,
            "removed_words": 0,
            "input_chars": 0,
            "retained_chars": 0,
            "removed_chars": 0
        },
        "per_source": {}
    }

    for filepath in ordered_files:
        out_path = interim_dir / filepath.name
        logger.info("Processing %s...", filepath.name)

        try:
            with open(filepath, 'r', encoding='utf-8') as fin, \
                 open(out_path, 'w', encoding='utf-8') as fout:

                for line in fin:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                        source_name = record.get("source_name", filepath.stem)

                        if not text: continue

                        if source_name not in stats["per_source"]:
                            stats["per_source"][source_name] = {
                                "input_docs": 0,
                                "unique_docs": 0,
                                "exact_duplicates": 0,
                                "input_words": 0,
                                "retained_words": 0,
                                "removed_words": 0,
                                "input_chars": 0,
                                "retained_chars": 0,
                                "removed_chars": 0,
                                "cross_source_duplicates": defaultdict(int)
                            }

                        s = stats["per_source"][source_name]
                        g = stats["global"]

                        words = len(text.split())
                        chars = len(text)

                        g["input_docs"] += 1
                        g["input_words"] += words
                        g["input_chars"] += chars

                        s["input_docs"] += 1
                        s["input_words"] += words
                        s["input_chars"] += chars

                        h = compute_hash(text)

                        if h in seen_hashes:
                            g["duplicate_docs"] += 1
                            g["removed_words"] += words
                            g["removed_chars"] += chars

                            s["exact_duplicates"] += 1
                            s["removed_words"] += words
                            s["removed_chars"] += chars

                            original_src = seen_hashes[h]
                            if original_src != source_name:
                                s["cross_source_duplicates"][original_src] += 1

                            continue

                        seen_hashes[h] = source_name

                        g["retained_docs"] += 1
                        g["retained_words"] += words
                        g["retained_chars"] += chars

                        s["unique_docs"] += 1
                        s["retained_words"] += words
                        s["retained_chars"] += chars

                        fout.write(json.dumps(record, ensure_ascii=False) + '\n')

                        if g["input_docs"] % 100000 == 0:
                            logger.info("  Processed %d documents globally so far... (%d duplicates)", g["input_docs"], g["duplicate_docs"])

                    except Exception as e:
                        logger.warning("Malformed record in %s: %s", filepath.name, e)
                        continue
        except Exception as e:
            logger.error("Failed to process %s: %s", filepath.name, e)

    # Final percentage calculations
    if stats["global"]["input_docs"] > 0:
        stats["global"]["duplicate_percentage"] = (stats["global"]["duplicate_docs"] / stats["global"]["input_docs"]) * 100

    report_path = stats_dir / "nepali_deduplication.json"
    save_json(stats, report_path)

    logger.info("--- Global Deduplication Complete ---")
    g = stats["global"]
    logger.info("Global Input Docs: %d", g["input_docs"])
    logger.info("Global Retained Docs: %d", g["retained_docs"])
    logger.info("Global Duplicates: %d (%.2f%%)", g["duplicate_docs"], g["duplicate_percentage"])
    logger.info("Global Input Words: %d | Retained: %d | Removed: %d", g["input_words"], g["retained_words"], g["removed_words"])
    logger.info("Report saved to %s", report_path)

if __name__ == "__main__":
    main()
