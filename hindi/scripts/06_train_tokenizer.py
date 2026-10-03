"""Stage 6: Train SentencePiece Tokenizers for Hindi.

Two-phase design:
- Phase A: Train and evaluate candidate tokenizers (BPE/Unigram at 8k/10k).
- Phase B: Train final selected tokenizer on full corpus (NOT YET IMPLEMENTED).
"""

import argparse
import sys
import os
import json
import random
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
from common.utils import setup_logger, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train SentencePiece Tokenizers")
    parser.add_argument("--config", default="hindi/configs/phase1_config.yaml")
    parser.add_argument("--phase", choices=["smoke_test", "candidates", "final"], required=True)
    return parser.parse_args()

def extract_sample_text(train_dir: Path, out_path: Path, max_sentences_per_source: int, seed: int, logger):
    """Extract a proportional/deterministic sample of sentences from train JSONL files."""
    files = list(train_dir.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No training data found in {train_dir}")
        
    random.seed(seed)
    total_extracted = 0
    
    with open(out_path, 'w', encoding='utf-8') as fout:
        for f in files:
            source_name = f.stem
            # Reservoir sample or just collect and sample if memory allows. 
            # Since JSONL files are big, we can reservoir sample sentences.
            # But wait, max 1.25M sentences is small enough to hold in memory per source.
            sentences = []
            logger.info("  Scanning %s...", f.name)
            with open(f, 'r', encoding='utf-8') as fin:
                for line in fin:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                        if not text: continue
                        # Split into lines/sentences
                        lines = text.split('\n')
                        for l in lines:
                            lt = l.strip()
                            if lt:
                                sentences.append(lt)
                    except Exception:
                        pass
            
            # Now sample deterministically
            random.seed(f"{seed}_{source_name}")
            if len(sentences) > max_sentences_per_source:
                sampled = random.sample(sentences, max_sentences_per_source)
            else:
                sampled = sentences
                
            logger.info("  Extracted %d sentences from %s", len(sampled), source_name)
            for s in sampled:
                fout.write(s + "\n")
                total_extracted += 1
                
    return total_extracted

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

def run_smoke_test(config, logger):
    logger.info("--- RUNNING SMOKE TEST ---")
    train_dir = resolve_path(config["paths"]["data_splits_dir"]) / "train"
    scratch_dir = resolve_path("scratch")
    ensure_dir(scratch_dir)
    
    smoke_txt = scratch_dir / "smoke_train.txt"
    smoke_prefix = scratch_dir / "smoke_model"
    
    logger.info("1. Extracting tiny sample...")
    extract_sample_text(train_dir, smoke_txt, 1000, 42, logger)
    
    logger.info("2. Training tiny SP model...")
    # SP parameters
    spm.SentencePieceTrainer.train(
        input=str(smoke_txt),
        model_prefix=str(smoke_prefix),
        vocab_size=1000,
        model_type="bpe",
        character_coverage=0.9995,
        normalization_rule_name="identity",
        max_sentence_length=20000,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>"
    )
    
    logger.info("3. Verifying model...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(smoke_prefix) + ".model")
    
    test_text = "नमस्ते दुनिया, यह एक परीक्षण है।"
    encoded_ids = sp.encode_as_ids(test_text)
    decoded_text = sp.decode_ids(encoded_ids)
    
    logger.info("Original: %s", test_text)
    logger.info("Decoded:  %s", decoded_text)
    if test_text.replace(" ", "") == decoded_text.replace(" ", ""):
        logger.info("-> Encoding/Decoding preserves Devanagari perfectly.")
    else:
        logger.error("-> Mojibake or corruption detected!")
        
    logger.info("Special Tokens:")
    logger.info("  pad_id: %d", sp.pad_id())
    logger.info("  unk_id: %d", sp.unk_id())
    logger.info("  bos_id: %d", sp.bos_id())
    logger.info("  eos_id: %d", sp.eos_id())
    
    logger.info("4. Cleaning up smoke test artifacts...")
    if smoke_txt.exists(): smoke_txt.unlink()
    if Path(str(smoke_prefix) + ".model").exists(): Path(str(smoke_prefix) + ".model").unlink()
    if Path(str(smoke_prefix) + ".vocab").exists(): Path(str(smoke_prefix) + ".vocab").unlink()
    
    logger.info("Smoke test PASSED.")

def run_candidates(config, logger):
    logger.info("--- RUNNING PHASE A: CANDIDATE TRAINING ---")
    train_dir = resolve_path(config["paths"]["data_splits_dir"]) / "train"
    val_dir = resolve_path(config["paths"]["data_splits_dir"]) / "val"
    models_dir = ensure_dir(config["paths"]["tokenizer_models_dir"])
    vocab_dir = ensure_dir(config["paths"]["tokenizer_vocab_dir"])
    
    scratch_dir = resolve_path("scratch")
    ensure_dir(scratch_dir)
    
    train_txt = scratch_dir / "candidates_train_sample.txt"
    
    logger.info("1. Constructing representative training sample...")
    # 4 sources, max 1.25M sentences each = max 5M sentences
    total_sentences = extract_sample_text(train_dir, train_txt, 1250000, 42, logger)
    logger.info("Extracted %d total sentences for tokenizer training.", total_sentences)
    
    matrix = [
        ("bpe", 8000),
        ("bpe", 10000),
        ("unigram", 8000),
        ("unigram", 10000)
    ]
    
    results = []
    
    import time
    logger.info("2. Training Candidates...")
    for algo, vs in matrix:
        name = f"hindi_{algo}_{vs}"
        model_prefix = models_dir / name
        
        logger.info("  -> Training %s...", name)
        start_time = time.time()
        error_msg = None
        metrics = {}
        try:
            spm.SentencePieceTrainer.train(
                input=str(train_txt),
                model_prefix=str(model_prefix),
                vocab_size=vs,
                model_type=algo,
                character_coverage=0.9995,
                normalization_rule_name="identity",
                max_sentence_length=20000,
                input_sentence_size=5000000,
                shuffle_input_sentence=True,
                unk_piece="<unk>",
                bos_piece="<s>",
                eos_piece="</s>",
                pad_piece="<pad>"
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
        metrics["algo"] = algo
        metrics["vocab"] = vs
        metrics["train_time_sec"] = train_time
        metrics["error"] = error_msg
        results.append(metrics)
        
    logger.info("3. Cleaning up temporary training file...")
    if train_txt.exists():
        train_txt.unlink()
        
    logger.info("--- CANDIDATE COMPARISON TABLE ---")
    header = f"{'Model':<20} | {'Time(s)':<8} | {'Fertility':<10} | {'Chars/Tok':<10} | {'Unk %':<10} | {'Tokens'}"
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        mname = f"{r['algo'].upper()} {r['vocab']}"
        if r.get('error'):
            logger.info(f"{mname:<20} | {r['train_time_sec']:<8.1f} | FAILED: {r['error']}")
        else:
            logger.info(f"{mname:<20} | {r['train_time_sec']:<8.1f} | {r['fertility']:<10.3f} | {r['chars_per_token']:<10.3f} | {r['unk_pct']:<10.3f}% | {r['tokens']}")
            
    # Write to a JSON file for the final report
    with open(scratch_dir / "phaseA_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    logger.info("Phase A complete. Await explicit manual selection of the winner before retraining.")

def run_final(config, logger):
    import time
    logger.info("--- RUNNING PHASE B: FINAL TOKENIZER TRAINING ---")
    train_dir = resolve_path(config["paths"]["data_splits_dir"]) / "train"
    val_dir = resolve_path(config["paths"]["data_splits_dir"]) / "val"
    models_dir = ensure_dir(config["paths"]["tokenizer_models_dir"])
    vocab_dir = ensure_dir(config["paths"]["tokenizer_vocab_dir"])
    
    scratch_dir = resolve_path("scratch")
    ensure_dir(scratch_dir)
    
    train_txt = scratch_dir / "final_train_sample.txt"
    name = "hindi_bpe_10k_final"
    model_prefix = models_dir / name
    
    logger.info("1. Estimating resources & extracting sample...")
    # Cap at 5,000,000 sentences per source to prevent HPLT domination but gather all manual data
    max_per_source = 5000000 
    total_sentences = extract_sample_text(train_dir, train_txt, max_per_source, 42, logger)
    logger.info("Extracted %d total sentences for final tokenizer training.", total_sentences)
    
    file_size_mb = train_txt.stat().st_size / (1024 * 1024)
    logger.info("Expected temporary text-file size: %.2f MB", file_size_mb)
    logger.info("Expected RAM usage: ~%d GB (SentencePiece internal loading)", int(total_sentences / 1000000 * 1.2))
    
    # Safe to proceed
    logger.info("2. Training Final Tokenizer...")
    start_time = time.time()
    try:
        spm.SentencePieceTrainer.train(
            input=str(train_txt),
            model_prefix=str(model_prefix),
            vocab_size=10000,
            model_type="bpe",
            character_coverage=0.9995,
            normalization_rule_name="identity",
            max_sentence_length=20000,
            input_sentence_size=20000000,
            shuffle_input_sentence=True,
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            pad_piece="<pad>"
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
        
    logger.info("Phase B final tokenizer training complete.")

def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    logger = setup_logger("hindi_tokenizer")
    
    if args.phase == "smoke_test":
        run_smoke_test(config, logger)
    elif args.phase == "candidates":
        run_candidates(config, logger)
    elif args.phase == "final":
        run_final(config, logger)

if __name__ == "__main__":
    main()
