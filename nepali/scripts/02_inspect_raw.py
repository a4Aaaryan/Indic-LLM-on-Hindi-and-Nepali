"""Stage 2: Raw corpus inspection for Nepali.

Measures initial document counts, script distribution, encoding validation,
and contamination patterns across raw sources.
"""

import argparse
import sys
import json
import gzip
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_yaml_config, resolve_path
from common.utils import setup_logger, save_json

DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]+')

def calculate_stats(filepath: Path) -> dict:
    """Calculates stats grouped by source_name within the file."""
    stats = {}

    file_size = filepath.stat().st_size

    try:
        if filepath.suffix == '.gz':
            f = gzip.open(filepath, 'rt', encoding='utf-8')
        else:
            f = open(filepath, 'r', encoding='utf-8')

        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if not text: continue

                source_name = data.get("source_name", filepath.stem)

                if source_name not in stats:
                    stats[source_name] = {
                        "source_name": source_name,
                        "file_name": filepath.name,
                        "docs": 0,
                        "words": 0,
                        "chars": 0,
                        "devanagari_chars": 0,
                        "file_size_bytes": file_size, # Only accurate if 1 source per file
                    }

                s = stats[source_name]
                s["docs"] += 1
                s["words"] += len(text.split())

                char_count = len(text)
                s["chars"] += char_count

                # Count Devanagari characters using regex
                dev_chars = sum(len(match) for match in DEVANAGARI_RANGE.findall(text))
                s["devanagari_chars"] += dev_chars

            except Exception:
                continue

        f.close()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    for s in stats.values():
        if s["chars"] > 0:
            s["devanagari_ratio"] = s["devanagari_chars"] / s["chars"]
        else:
            s["devanagari_ratio"] = 0.0

    return stats

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect raw Nepali corpus files.")
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
    logger = setup_logger("nepali_inspect")

    raw_dir = resolve_path(config["paths"]["data_raw_dir"])
    stats_dir = resolve_path(config["paths"]["statistics_dir"])
    stats_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Inspecting raw Nepali files in: %s", raw_dir)

    raw_files = list(raw_dir.glob("*.jsonl")) + list(raw_dir.glob("*.jsonl.gz"))
    if not raw_files:
        logger.info("No raw data files found in %s (expected during Stage 0 setup).", raw_dir)
        return

    logger.info("Found %d raw data files to inspect.", len(raw_files))

    all_stats = {}
    for filepath in raw_files:
        if "repaired" in filepath.name:
            # Skip the temporary repaired file to avoid double counting
            continue
        logger.info("Inspecting %s...", filepath.name)
        file_stats = calculate_stats(filepath)
        for src, s in file_stats.items():
            all_stats[src] = s
            logger.info("Stats for %s: Docs=%d, Words=%d, DevRatio=%.4f", src, s["docs"], s["words"], s["devanagari_ratio"])

    output_path = stats_dir / "nepali_raw_inspection.json"
    save_json(all_stats, output_path)
    logger.info("Saved inspection results to %s", output_path)

    # Print Concise Summary
    print("\n" + "="*50)
    print("NEPALI RAW INSPECTION SUMMARY")
    print("="*50)

    total_manual_words = 0
    total_hplt_words = 0
    total_docs = 0

    for src, s in all_stats.items():
        print(f"\nSource: {src}")
        print(f"File: {s['file_name']} (Size: {s['file_size_bytes']/1024/1024:.2f} MB)")
        print(f"Docs: {s['docs']:,}")
        print(f"Words: {s['words']:,}")
        print(f"Devanagari Ratio: {s['devanagari_ratio']:.4f}")

        total_docs += s["docs"]
        if "hplt" in src.lower():
            total_hplt_words += s["words"]
        else:
            total_manual_words += s["words"]

    print("\n" + "-"*50)
    print(f"Total Documents: {total_docs:,}")
    print(f"Total Manual Words: {total_manual_words:,}")
    print(f"Total HPLT Words: {total_hplt_words:,}")
    print(f"Total Words (Combined): {total_manual_words + total_hplt_words:,}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
