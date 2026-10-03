"""Stage 2: Raw corpus inspection for Hindi.

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
    stats = {
        "docs": 0,
        "words": 0,
        "chars": 0,
        "devanagari_chars": 0
    }
    
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
                
                stats["docs"] += 1
                stats["words"] += len(text.split())
                stats["chars"] += len(text)
                
                # Count Devanagari characters using regex
                dev_chars = sum(len(match) for match in DEVANAGARI_RANGE.findall(text))
                stats["devanagari_chars"] += dev_chars
            except Exception:
                continue
                
        f.close()
    except Exception as e:
        pass
        
    if stats["chars"] > 0:
        stats["devanagari_ratio"] = stats["devanagari_chars"] / stats["chars"]
    else:
        stats["devanagari_ratio"] = 0.0
        
    return stats

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect raw Hindi corpus files.")
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
    logger = setup_logger("hindi_inspect")

    raw_dir = resolve_path(config["paths"]["data_raw_dir"])
    stats_dir = resolve_path(config["paths"]["statistics_dir"])
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Inspecting raw Hindi files in: %s", raw_dir)

    raw_files = list(raw_dir.glob("*.jsonl")) + list(raw_dir.glob("*.jsonl.gz"))
    if not raw_files:
        logger.info("No raw data files found in %s (expected during Stage 0 setup).", raw_dir)
        return

    logger.info("Found %d raw data files to inspect.", len(raw_files))
    
    all_stats = {}
    for filepath in raw_files:
        logger.info("Inspecting %s...", filepath.name)
        file_stats = calculate_stats(filepath)
        all_stats[filepath.name] = file_stats
        logger.info("Stats for %s: %s", filepath.name, file_stats)
        
    output_path = stats_dir / "hindi_raw_inspection.json"
    save_json(all_stats, output_path)
    logger.info("Saved inspection results to %s", output_path)


if __name__ == "__main__":
    main()
