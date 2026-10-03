"""Stage 7b: Compute statistics for the Assignment-Compliant corpus.

Calculates final tokens using both the actual BPE tokenizer output
and the instructor's 1 word = 3 tokens heuristic.
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

def is_manual_source(source_name: str) -> bool:
    return source_name != "indic_hplt_v2" and "hplt" not in source_name.lower()

def safe_div(n, d):
    return n / d if d > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Generate Assignment Statistics")
    parser.add_argument("--config", default="nepali/configs/phase1_config.yaml")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    logger = setup_logger("nepali_stats_assignment")

    stats_dir = ensure_dir(resolve_path(config["paths"]["statistics_dir"]))
    final_dir = resolve_path("nepali/data/final")
    model_path = resolve_path(config["paths"]["tokenizer_models_dir"]) / "nepali_bpe_10k_assignment.model"

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    unk_id = sp.unk_id()

    stats = {}
    split_stats = {"train": {"docs": 0, "words": 0}, "val": {"docs": 0, "words": 0}, "test": {"docs": 0, "words": 0}}
    total_docs = 0
    total_words = 0
    total_tokens = 0
    total_chars = 0
    total_unks = 0

    start_time = time.time()

    for split_name in ["train", "val", "test"]:
        split_dir = final_dir / split_name
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
                        stats[src] = {"docs": 0, "words": 0, "tokens": 0, "chars": 0, "unks": 0}

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

                    total_docs += 1
                    total_words += words
                    total_chars += chars
                    total_tokens += tokens
                    total_unks += unks

    runtime = time.time() - start_time

    manual_words = sum(c["words"] for s, c in stats.items() if is_manual_source(s))
    hplt_words = sum(c["words"] for s, c in stats.items() if not is_manual_source(s))

    # Calculate using instructor heuristic (1 word = 3 tokens)
    heuristic_manual_tokens = manual_words * 3
    heuristic_hplt_tokens = hplt_words * 3
    heuristic_total = heuristic_manual_tokens + heuristic_hplt_tokens
    heuristic_manual_pct = safe_div(heuristic_manual_tokens, heuristic_total) * 100.0
    passed_heuristic = heuristic_manual_pct >= 20.0

    # Calculate actual
    manual_tokens_actual = sum(c["tokens"] for s, c in stats.items() if is_manual_source(s))
    actual_manual_pct = safe_div(manual_tokens_actual, total_tokens) * 100.0

    final_report = {
        "metadata": {
            "generation_timestamp": datetime.utcnow().isoformat() + "Z",
            "corpus_input_dir": str(final_dir),
            "tokenizer_model_path": str(model_path),
            "tokenizer_model_type": "BPE",
            "vocabulary_size": 10000,
            "runtime_seconds": round(runtime, 2)
        },
        "overall": {
            "documents": total_docs,
            "words": total_words,
            "characters": total_chars,
            "sp_tokens_actual": total_tokens,
            "heuristic_tokens_total": heuristic_total,
            "fertility_actual": round(safe_div(total_tokens, total_words), 3),
            "unk_pct": round(safe_div(total_unks, total_tokens) * 100.0, 3)
        },
        "instructor_heuristic_evaluation_3_tokens_per_word": {
            "heuristic_manual_tokens": heuristic_manual_tokens,
            "heuristic_hplt_tokens": heuristic_hplt_tokens,
            "heuristic_total_tokens": heuristic_total,
            "heuristic_manual_pct": round(heuristic_manual_pct, 3),
            "requirement_passed": passed_heuristic
        },
        "actual_tokenizer_breakdown": {
            "manual_tokens": manual_tokens_actual,
            "hplt_tokens": total_tokens - manual_tokens_actual,
            "manual_pct": round(actual_manual_pct, 3)
        },
        "per_source": stats
    }

    json_out = stats_dir / "nepali_assignment_final_stats.json"
    csv_out = stats_dir / "nepali_assignment_final_stats.csv"

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    csv_rows = [["Source", "Is Manual", "Documents", "Words", "Heuristic Tokens", "Actual SP Tokens", "Actual Fertility"]]
    for src, c in stats.items():
        csv_rows.append([
            src, str(is_manual_source(src)), str(c["docs"]), str(c["words"]),
            str(c["words"] * 3), str(c["tokens"]), f'{safe_div(c["tokens"], c["words"]):.3f}'
        ])

    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    logger.info("Saved JSON report to %s", json_out)
    logger.info("Saved CSV report to %s", csv_out)
    logger.info("Heuristic Manual Pct: %.2f%% (Passed: %s)", heuristic_manual_pct, passed_heuristic)

if __name__ == "__main__":
    main()
