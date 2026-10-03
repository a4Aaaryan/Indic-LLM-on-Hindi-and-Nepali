"""Stage 7: Compute and export Nepali corpus and tokenizer statistics.

Calculates:
- Split token counts (train/val/test)
- Downloaded vs manual token breakdown (verifying >= 20% manual requirement)
- Tokenizer fertility, average characters per token, unknown token rates
- Outputs machine-readable JSON/CSV summaries to report/tables/
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_yaml_config, resolve_path, ensure_dir
from common.utils import setup_logger

try:
    import sentencepiece as spm
except ImportError:
    print("Error: sentencepiece is not installed.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Nepali corpus and tokenizer statistics.")
    parser.add_argument(
        "--config",
        type=str,
        default="nepali/configs/phase1_config.yaml",
        help="Path to Phase 1 configuration YAML",
    )
    return parser.parse_args()

def is_manual_source(source_name: str) -> bool:
    return source_name != "indic_hplt_v2" and "hplt" not in source_name.lower()

def safe_div(n, d):
    return n / d if d > 0 else 0.0

def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    logger = setup_logger("nepali_stats")

    stats_dir = ensure_dir(resolve_path(config["paths"]["statistics_dir"]))
    logger.info("Nepali statistics generation initialized. Output dir: %s", stats_dir)

    splits_dir = resolve_path(config["paths"]["data_splits_dir"])
    model_dir = resolve_path(config["paths"]["tokenizer_models_dir"])
    model_path = model_dir / "nepali_bpe_10k_final.model"

    if not splits_dir.exists():
        logger.error("Splits directory not found: %s", splits_dir)
        sys.exit(1)

    if not model_path.exists():
        logger.error("Tokenizer model not found: %s", model_path)
        sys.exit(1)

    logger.info("Loading tokenizer from %s", model_path)
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    unk_id = sp.unk_id()

    stats = {}
    split_stats = {"train": {"docs": 0, "words": 0, "tokens": 0}, "val": {"docs": 0, "words": 0, "tokens": 0}, "test": {"docs": 0, "words": 0, "tokens": 0}}
    total_docs = 0
    total_words = 0
    total_tokens = 0
    total_chars = 0
    total_unks = 0

    logger.info("Streaming corpus from splits in %s...", splits_dir)
    start_time = time.time()

    for split_name in ["train", "val", "test"]:
        split_dir = splits_dir / split_name
        if not split_dir.exists(): continue

        for fpath in split_dir.glob("*.jsonl"):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    src = record.get("source_name", fpath.stem)
                    text = record.get("text", "")
                    if not text: continue

                    if src not in stats:
                        stats[src] = {
                            "docs": 0, "words": 0, "tokens": 0, "chars": 0, "unks": 0
                        }

                    words = len(text.split())
                    chars = len(text)
                    encoded = sp.encode_as_ids(text)
                    tokens = len(encoded)
                    unks = encoded.count(unk_id)

                    stats[src]["docs"] += 1
                    stats[src]["words"] += words
                    stats[src]["chars"] += chars
                    stats[src]["tokens"] += tokens
                    stats[src]["unks"] += unks

                    split_stats[split_name]["docs"] += 1
                    split_stats[split_name]["words"] += words
                    split_stats[split_name]["tokens"] += tokens

                    total_docs += 1
                    total_words += words
                    total_chars += chars
                    total_tokens += tokens
                    total_unks += unks

                    if total_docs % 100000 == 0:
                        logger.info("Processed %d documents...", total_docs)

    runtime = time.time() - start_time
    logger.info("Finished streaming corpus in %.1f seconds. Generating report...", runtime)

    overall_fertility = safe_div(total_tokens, total_words)
    overall_chars_per_tok = safe_div(total_chars, total_tokens)
    overall_unk_pct = safe_div(total_unks, total_tokens) * 100.0

    manual_docs = sum(c["docs"] for s, c in stats.items() if is_manual_source(s))
    manual_words = sum(c["words"] for s, c in stats.items() if is_manual_source(s))
    manual_tokens = sum(c["tokens"] for s, c in stats.items() if is_manual_source(s))

    hplt_docs = sum(c["docs"] for s, c in stats.items() if not is_manual_source(s))
    hplt_words = sum(c["words"] for s, c in stats.items() if not is_manual_source(s))
    hplt_tokens = sum(c["tokens"] for s, c in stats.items() if not is_manual_source(s))

    manual_pct = safe_div(manual_tokens, total_tokens) * 100.0
    hplt_pct = safe_div(hplt_tokens, total_tokens) * 100.0
    passed_requirement = manual_pct >= 20.0

    target_tokens = 300000000 # 300M tokens
    difference = total_tokens - target_tokens

    final_report = {
        "metadata": {
            "generation_timestamp": datetime.utcnow().isoformat() + "Z",
            "corpus_input_dir": str(splits_dir),
            "tokenizer_model_path": str(model_path),
            "tokenizer_model_type": "BPE",
            "vocabulary_size": 10000,
            "runtime_seconds": round(runtime, 2)
        },
        "overall": {
            "documents": total_docs,
            "words": total_words,
            "characters": total_chars,
            "sp_tokens": total_tokens,
            "target_tokens": target_tokens,
            "distance_from_target": difference,
            "fertility": round(overall_fertility, 3),
            "chars_per_token": round(overall_chars_per_tok, 3),
            "unk_count": total_unks,
            "unk_pct": round(overall_unk_pct, 3)
        },
        "splits": split_stats,
        "manual_vs_hplt_breakdown": {
            "manual": {
                "documents": manual_docs,
                "words": manual_words,
                "sp_tokens": manual_tokens,
                "token_percentage": round(manual_pct, 3)
            },
            "hplt": {
                "documents": hplt_docs,
                "words": hplt_words,
                "sp_tokens": hplt_tokens,
                "token_percentage": round(hplt_pct, 3)
            },
            "requirement_passed": passed_requirement,
            "minimum_required_pct": 20.0
        },
        "per_source": {}
    }

    csv_rows = [["Source", "Is Manual", "Documents", "Words", "Characters", "SP Tokens", "Fertility", "Chars per Token", "UNK Count", "UNK Pct"]]

    for src, c in stats.items():
        fertility = safe_div(c["tokens"], c["words"])
        chars_per_tok = safe_div(c["chars"], c["tokens"])
        unk_pct = safe_div(c["unks"], c["tokens"]) * 100.0

        final_report["per_source"][src] = {
            "documents": c["docs"],
            "words": c["words"],
            "characters": c["chars"],
            "sp_tokens": c["tokens"],
            "fertility": round(fertility, 3),
            "chars_per_token": round(chars_per_tok, 3),
            "unk_count": c["unks"],
            "unk_pct": round(unk_pct, 3),
            "is_manual": is_manual_source(src)
        }

        csv_rows.append([
            src, str(is_manual_source(src)), str(c["docs"]), str(c["words"]), str(c["chars"]),
            str(c["tokens"]), f"{fertility:.3f}", f"{chars_per_tok:.3f}", str(c["unks"]), f"{unk_pct:.3f}%"
        ])

    json_out = stats_dir / "nepali_final_stats.json"
    csv_out = stats_dir / "nepali_final_stats.csv"

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    logger.info("Saved JSON report to %s", json_out)
    logger.info("Saved CSV report to %s", csv_out)

    if passed_requirement:
        logger.info("SUCCESS: Manual token percentage (%.2f%%) passes the >= 20%% requirement.", manual_pct)
    else:
        logger.warning("FAILED: Manual token percentage (%.2f%%) is below the 20%% requirement.", manual_pct)

    logger.info("Total BPE Tokens: %d (Target: %d, Diff: %d)", total_tokens, target_tokens, difference)
    logger.info("Stage 7 statistics generation complete.")

if __name__ == "__main__":
    main()
