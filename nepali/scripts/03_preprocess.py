"""Stage 3: Preprocessing and normalization pipeline for Nepali.

Applies:
- Unicode normalization (NFKC)
- HTML/markup stripping
- Deterministic whitespace normalization
- Filtering documents below minimum character length
"""

import argparse
import sys
import unicodedata
import json
import gzip
import re
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_yaml_config, resolve_path, ensure_dir
from common.utils import setup_logger

HTML_TAG_RE = re.compile(r'<[^>]+>')
WHITESPACE_RE = re.compile(r'\s+')

def normalize_text(text: str, form: str = "NFKC") -> str:
    """Normalize Unicode text according to specified form."""
    return unicodedata.normalize(form, text)

def clean_text(text: str) -> str:
    """Apply HTML stripping and deterministic whitespace normalization."""
    # Strip HTML tags
    text = HTML_TAG_RE.sub(' ', text)
    # Normalize unicode
    text = normalize_text(text)
    # Normalize whitespace (replace multiple spaces/newlines with single space)
    # Preserve paragraph boundaries (newlines).
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = WHITESPACE_RE.sub(' ', line).strip()
        if line:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and normalize Nepali corpus.")
    parser.add_argument(
        "--config",
        type=str,
        default="nepali/configs/phase1_config.yaml",
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
    logger = setup_logger("nepali_preprocess")

    raw_dir = resolve_path(config["paths"]["data_raw_dir"])
    cleaned_dir = ensure_dir(config["paths"]["data_cleaned_dir"])

    logger.info("Nepali preprocessing pipeline initialized.")
    logger.info("Input: %s -> Output: %s", raw_dir, cleaned_dir)

    min_chars = config.get("preprocessing", {}).get("min_character_length", 50)

    raw_files = list(raw_dir.glob("*.jsonl")) + list(raw_dir.glob("*.jsonl.gz"))
    if not raw_files:
        logger.warning("No raw files found in %s", raw_dir)
        return

    for filepath in raw_files:
        if "repaired" in filepath.name:
            continue

        logger.info("Processing %s...", filepath.name)

        out_name = filepath.name
        if out_name.endswith(".gz"):
            out_name = out_name[:-3] # Remove .gz for the cleaned output

        out_path = cleaned_dir / out_name

        in_docs, in_words, in_chars = 0, 0, 0
        out_docs, out_words, out_chars = 0, 0, 0
        removed_empty = 0
        removed_short = 0

        try:
            if filepath.suffix == '.gz':
                fin = gzip.open(filepath, 'rt', encoding='utf-8')
            else:
                fin = open(filepath, 'r', encoding='utf-8')

            with open(out_path, 'w', encoding='utf-8') as fout:
                for line in fin:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")

                        in_docs += 1
                        in_words += len(text.split())
                        in_chars += len(text)

                        cleaned = clean_text(text)

                        if not cleaned:
                            removed_empty += 1
                            continue

                        if len(cleaned) < min_chars:
                            removed_short += 1
                            continue

                        record["text"] = cleaned
                        fout.write(json.dumps(record, ensure_ascii=False) + '\n')

                        out_docs += 1
                        out_words += len(cleaned.split())
                        out_chars += len(cleaned)

                        if args.sample and in_docs >= args.sample:
                            break

                    except Exception as e:
                        continue
            fin.close()
        except Exception as e:
            logger.error("Failed to process %s: %s", filepath.name, e)

        logger.info("  Input: %d docs, %d words, %d chars", in_docs, in_words, in_chars)
        logger.info("  Output: %d docs, %d words, %d chars", out_docs, out_words, out_chars)
        logger.info("  Removed %d empty, %d too short (<%d chars)", removed_empty, removed_short, min_chars)
        logger.info("  Saved to %s", out_path.name)

    logger.info("Stage 3 Preprocessing Complete.")

if __name__ == "__main__":
    main()
