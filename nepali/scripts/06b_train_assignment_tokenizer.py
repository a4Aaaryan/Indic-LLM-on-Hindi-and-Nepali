"""Stage 6b: Train Assignment-Compliant Tokenizer.

Trains the BPE 10K tokenizer on the downsampled assignment-compliant
corpus located in `nepali/data/final/train`.
"""

import argparse
import sys
import json
import time
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError:
    print("Error: sentencepiece is not installed.")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_yaml_config, resolve_path, ensure_dir
from common.utils import setup_logger
from collections import defaultdict
import random

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


def main():
    parser = argparse.ArgumentParser(description="Train Assignment SentencePiece")
    parser.add_argument("--config", default="nepali/configs/phase1_config.yaml")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    logger = setup_logger("nepali_tokenizer_assignment")

    train_dir = resolve_path("nepali/data/final/train")
    val_dir = resolve_path("nepali/data/final/val")
    models_dir = ensure_dir(config["paths"]["tokenizer_models_dir"])
    vocab_dir = ensure_dir(config["paths"]["tokenizer_vocab_dir"])

    scratch_dir = resolve_path("scratch")
    ensure_dir(scratch_dir)

    train_txt = scratch_dir / "assignment_final_train_sample.txt"
    name = "nepali_bpe_10k_assignment"
    model_prefix = models_dir / name

    logger.info("1. Extracting training text from %s...", train_dir)
    max_per_source = 5000000
    total_sentences, source_contributions = extract_sample_text(train_dir, train_txt, max_per_source, logger)
    logger.info("Extracted %d total sentences.", total_sentences)

    logger.info("2. Training Assignment Tokenizer...")
    start_time = time.time()

    spm_params = {
        "model_type": "bpe",
        "vocab_size": 10000,
        "character_coverage": 0.9995,
        "normalization_rule_name": "identity",
        "max_sentence_length": 20000,
        "input_sentence_size": 20000000,
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
        logger.error("Failed to train assignment tokenizer: %s", str(e))
        return

    train_time = time.time() - start_time
    logger.info("Training complete in %.1f seconds.", train_time)

    if Path(str(model_prefix) + ".vocab").exists():
        import shutil
        shutil.move(str(model_prefix) + ".vocab", vocab_dir / f"{name}.vocab")

    logger.info("3. Evaluating on Validation split...")
    metrics = evaluate_tokenizer(str(model_prefix) + ".model", val_dir, logger)

    logger.info("Tokens: %d | Fertility: %.3f | Unk Pct: %.3f%%",
                metrics["tokens"], metrics["fertility"], metrics["unk_pct"])

    if train_txt.exists():
        train_txt.unlink()

    logger.info("Assignment tokenizer saved to %s", model_prefix)

if __name__ == "__main__":
    main()
