"""Stage 6: Train SentencePiece Tokenizers for Nepali.

Phase A: Train and evaluate candidate tokenizers (BPE at 8k/10k).
"""

import argparse
import sys
import os
import json
import random
import time
from pathlib import Path
from collections import defaultdict

try:
    import sentencepiece as spm
except ImportError:
    print("Error: sentencepiece is not installed.")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_yaml_config, resolve_path, ensure_dir
from common.utils import setup_logger, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train SentencePiece Tokenizers")
    parser.add_argument("--config", default="nepali/configs/phase1_config.yaml")
    parser.add_argument("--phase", choices=["candidates", "final"], default="candidates")
    return parser.parse_args()

def extract_sample_text(train_dir: Path, out_path: Path, max_sentences_per_source: int, logger):
    """Extract a proportional/deterministic sample using reservoir sampling per source."""
    files = list(train_dir.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No training data found in {train_dir}")

    reservoirs = defaultdict(list)
    counts = defaultdict(int)
    rngs = {}

    for f in files:
        logger.info("  Scanning %s...", f.name)
        with open(f, 'r', encoding='utf-8') as fin:
            for line in fin:
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    text = record.get("text", "")
                    if not text: continue
                    source_name = record.get("source_name", f.stem)

                    if source_name not in rngs:
                        rngs[source_name] = random.Random(f"42_{source_name}")

                    lines = text.split('\n')
                    for l in lines:
                        lt = l.strip()
                        if lt:
                            i = counts[source_name]
                            counts[source_name] += 1
                            if i < max_sentences_per_source:
                                reservoirs[source_name].append(lt)
                            else:
                                j = rngs[source_name].randint(0, i)
                                if j < max_sentences_per_source:
                                    reservoirs[source_name][j] = lt
                except Exception:
                    pass

    total_extracted = 0
    source_contributions = {}

    with open(out_path, 'w', encoding='utf-8') as fout:
        for source_name, sentences in reservoirs.items():
            logger.info("  Writing %d sentences from %s (out of %d seen)", len(sentences), source_name, counts[source_name])
            source_contributions[source_name] = len(sentences)
            for s in sentences:
                fout.write(s + "\n")
                total_extracted += 1

    return total_extracted, source_contributions

def evaluate_tokenizer(model_path: str, val_dir: Path, logger):
    """Evaluate a trained tokenizer on the validation set."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    docs = 0
    words = 0
    tokens = 0
    chars = 0
    unks = 0

    unk_id = sp.unk_id()

    files = list(val_dir.glob("*.jsonl"))
    for f in files:
        with open(f, 'r', encoding='utf-8') as fin:
            for line in fin:
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    text = record.get("text", "")
                    if not text: continue

                    docs += 1
                    ws = len(text.split())
                    words += ws
                    chars += len(text)

                    # Encode
                    encoded = sp.encode_as_ids(text)
                    tokens += len(encoded)
                    unks += encoded.count(unk_id)
                except Exception:
                    pass

    fertility = tokens / words if words > 0 else 0
    chars_per_token = chars / tokens if tokens > 0 else 0
    unk_pct = (unks / tokens * 100) if tokens > 0 else 0

    return {
        "docs": docs,
        "words": words,
        "tokens": tokens,
        "fertility": fertility,
        "chars_per_token": chars_per_token,
        "unks": unks,
        "unk_pct": unk_pct
    }

def run_candidates(config, logger):
    logger.info("--- RUNNING PHASE A: CANDIDATE TRAINING ---")
    train_dir = resolve_path(config["paths"]["data_splits_dir"]) / "train"
    val_dir = resolve_path(config["paths"]["data_splits_dir"]) / "val"
    models_dir = ensure_dir(config["paths"]["tokenizer_models_dir"])
    vocab_dir = ensure_dir(config["paths"]["tokenizer_vocab_dir"])
    stats_dir = ensure_dir(config["paths"]["statistics_dir"])

    scratch_dir = resolve_path("scratch")
    ensure_dir(scratch_dir)

    train_txt = scratch_dir / "candidates_train_sample.txt"

    logger.info("1. Constructing representative training sample...")
    # Max 1,250,000 sentences per source
    total_sentences, source_contributions = extract_sample_text(train_dir, train_txt, 1250000, logger)
    logger.info("Extracted %d total sentences for tokenizer training.", total_sentences)

    matrix = [
        ("bpe", 8000),
        ("bpe", 10000)
    ]

    results = []

    logger.info("2. Training Candidates...")
    for algo, vs in matrix:
        name = f"nepali_{algo}_{vs}"
        model_prefix = models_dir / name

        logger.info("  -> Training %s...", name)
        start_time = time.time()
        error_msg = None
        metrics = {}

        spm_params = {
            "model_type": algo,
            "vocab_size": vs,
            "character_coverage": 0.9995,
            "normalization_rule_name": "identity",
            "max_sentence_length": 20000,
            "input_sentence_size": 5000000,
            "shuffle_input_sentence": True,
            "unk_piece": "<unk>",
            "bos_piece": "<s>",
            "eos_piece": "</s>",
            "pad_piece": "<pad>"
        }

        try:
            spm.SentencePieceTrainer.train(
                input=str(train_txt),
                model_prefix=str(model_prefix),
                **spm_params
            )

            # Move vocab file to vocab_dir
            if Path(str(model_prefix) + ".vocab").exists():
                import shutil
                shutil.move(str(model_prefix) + ".vocab", vocab_dir / f"{name}.vocab")

            logger.info("  -> Evaluating %s on Validation split...", name)
            metrics = evaluate_tokenizer(str(model_prefix) + ".model", val_dir, logger)
        except Exception as e:
            logger.error("  -> FAILED %s: %s", name, str(e))
            error_msg = str(e)

        train_time = time.time() - start_time

        res = {
            "candidate_name": name,
            "algorithm": algo,
            "vocabulary_size": vs,
            "training_sample_size": total_sentences,
            "source_contributions": source_contributions,
            "sentencepiece_parameters": spm_params,
            "training_runtime_sec": train_time,
            "validation_metrics": metrics,
            "error": error_msg
        }
        results.append(res)

    logger.info("3. Cleaning up temporary training file...")
    if train_txt.exists():
        train_txt.unlink()

    logger.info("--- CANDIDATE COMPARISON TABLE ---")
    header = f"{'Model':<20} | {'Time(s)':<8} | {'Fertility':<10} | {'Chars/Tok':<10} | {'Unk %':<10} | {'Tokens'}"
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        mname = r["candidate_name"]
        if r.get('error'):
            logger.info(f"{mname:<20} | {r['training_runtime_sec']:<8.1f} | FAILED: {r['error']}")
        else:
            vm = r["validation_metrics"]
            logger.info(f"{mname:<20} | {r['training_runtime_sec']:<8.1f} | {vm['fertility']:<10.3f} | {vm['chars_per_token']:<10.3f} | {vm['unk_pct']:<10.3f}% | {vm['tokens']}")

    # Write to a JSON file for the final report
    report_path = stats_dir / "nepali_tokenizer_candidates.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Phase A complete. Report saved to %s", report_path)

def run_final(config, logger):
    logger.info("--- RUNNING PHASE B: FINAL TOKENIZER TRAINING ---")
    train_dir = resolve_path(config["paths"]["data_splits_dir"]) / "train"
    val_dir = resolve_path(config["paths"]["data_splits_dir"]) / "val"
    models_dir = ensure_dir(config["paths"]["tokenizer_models_dir"])
    vocab_dir = ensure_dir(config["paths"]["tokenizer_vocab_dir"])
    stats_dir = ensure_dir(config["paths"]["statistics_dir"])

    scratch_dir = resolve_path("scratch")
    ensure_dir(scratch_dir)

    train_txt = scratch_dir / "final_train_sample.txt"
    name = "nepali_bpe_10k_final"
    model_prefix = models_dir / name

    logger.info("1. Estimating resources & extracting sample...")
    # Cap at 5,000,000 sentences per source to prevent HPLT domination but gather all manual data
    max_per_source = 5000000
    total_sentences, source_contributions = extract_sample_text(train_dir, train_txt, max_per_source, logger)
    logger.info("Extracted %d total sentences for final tokenizer training.", total_sentences)

    file_size_mb = train_txt.stat().st_size / (1024 * 1024)
    logger.info("Expected temporary text-file size: %.2f MB", file_size_mb)
    logger.info("Expected RAM usage: ~%d GB (SentencePiece internal loading)", int(total_sentences / 1000000 * 1.2))

    logger.info("2. Training Final Tokenizer...")
    start_time = time.time()

    spm_params = {
        "model_type": "bpe",
        "vocab_size": 10000,
        "character_coverage": 0.9995,
        "normalization_rule_name": "identity",
        "max_sentence_length": 20000,
        "input_sentence_size": 20000000, # Large enough to ingest the entire sample
        "shuffle_input_sentence": True,
        "unk_piece": "<unk>",
        "bos_piece": "<s>",
        "eos_piece": "</s>",
        "pad_piece": "<pad>"
    }

    try:
        spm.SentencePieceTrainer.train(
            input=str(train_txt),
            model_prefix=str(model_prefix),
            **spm_params
        )
    except Exception as e:
        logger.error("Failed to train final tokenizer: %s", str(e))
        return

    train_time = time.time() - start_time
    logger.info("Training complete in %.1f seconds.", train_time)

    if Path(str(model_prefix) + ".vocab").exists():
        import shutil
        shutil.move(str(model_prefix) + ".vocab", vocab_dir / f"{name}.vocab")

    model_size_mb = Path(str(model_prefix) + ".model").stat().st_size / (1024 * 1024)
    vocab_size_kb = Path(vocab_dir / f"{name}.vocab").stat().st_size / 1024
    logger.info("Model size: %.2f MB, Vocab size: %.2f KB", model_size_mb, vocab_size_kb)

    logger.info("3. Evaluating Final Tokenizer on Validation split...")
    metrics = evaluate_tokenizer(str(model_prefix) + ".model", val_dir, logger)

    logger.info("--- FINAL VALIDATION METRICS ---")
    logger.info("Tokens: %d", metrics["tokens"])
    logger.info("Fertility: %.3f", metrics["fertility"])
    logger.info("Chars/Tok: %.3f", metrics["chars_per_token"])
    logger.info("UNK count: %d", metrics["unks"])
    logger.info("UNK pct:   %.3f%%", metrics["unk_pct"])

    logger.info("4. Cleaning up temporary training file...")
    if train_txt.exists():
        train_txt.unlink()

    res = {
        "candidate_name": name,
        "algorithm": "bpe",
        "vocabulary_size": 10000,
        "training_sample_size": total_sentences,
        "source_contributions": source_contributions,
        "sentencepiece_parameters": spm_params,
        "training_runtime_sec": train_time,
        "validation_metrics": metrics
    }

    # Save the phase B final report separately
    report_path = stats_dir / "nepali_tokenizer_final.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    logger.info("Phase B final tokenizer training complete. Report saved to %s", report_path)


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    logger = setup_logger("nepali_tokenizer")

    if args.phase == "candidates":
        run_candidates(config, logger)
    elif args.phase == "final":
        run_final(config, logger)
    else:
        logger.info("Phase %s not implemented for Nepali yet.", args.phase)

if __name__ == "__main__":
    main()
