"""
Phase 2 — Stage 5: Data Preparation Pipeline

Converts JSONL corpora into memory-mapped uint16 .bin files for
efficient language model training.

Supports both Hindi and Nepali through configuration arguments.
All paths are supplied externally for portability across
local / Kaggle / Colab environments.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sentencepiece as spm


# ---------------------------------------------------------------------------
# Tokenization / conversion
# ---------------------------------------------------------------------------

def stream_jsonl_texts(input_dir: Path, text_field: str = "text"):
    """Yield text strings from all .jsonl files in input_dir (sorted for determinism)."""
    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {input_dir}")
    for jsonl_path in jsonl_files:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                text = doc.get(text_field)
                if text and isinstance(text, str) and text.strip():
                    yield text.strip()
                else:
                    yield None  # signal a skip


def convert_split(
    input_dir: Path,
    output_path: Path,
    tokenizer_path: Path,
    text_field: str = "text",
):
    """
    Stream JSONL documents, tokenize with SentencePiece, and write
    token IDs as a flat uint16 binary file.

    Returns a dict of statistics.
    """
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_path))
    vocab_size = sp.GetPieceSize()

    if vocab_size > 65535:
        raise ValueError(
            f"Vocabulary size {vocab_size} exceeds uint16 max (65535)"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_docs = 0
    n_skipped = 0
    n_tokens = 0
    token_min = 65535
    token_max = 0

    t0 = time.time()

    with open(output_path, "wb") as out_f:
        for text in stream_jsonl_texts(input_dir, text_field):
            if text is None:
                n_skipped += 1
                n_docs += 1
                continue

            ids = sp.Encode(text, out_type=int)
            if not ids:
                n_skipped += 1
                n_docs += 1
                continue

            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(out_f)

            n_tokens += len(ids)
            token_min = min(token_min, int(arr.min()))
            token_max = max(token_max, int(arr.max()))
            n_docs += 1

    elapsed = time.time() - t0
    file_bytes = output_path.stat().st_size if output_path.exists() else 0

    stats = {
        "input_dir": input_dir.name,
        "output_path": output_path.name,
        "tokenizer": tokenizer_path.name,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "documents_processed": n_docs,
        "documents_skipped": n_skipped,
        "token_count": n_tokens,
        "token_min": token_min if n_tokens > 0 else None,
        "token_max": token_max if n_tokens > 0 else None,
        "bytes_written": file_bytes,
        "elapsed_seconds": round(elapsed, 3),
        "tokens_per_second": round(n_tokens / elapsed, 1) if elapsed > 0 else 0,
    }

    return stats


def save_metadata(output_dir: Path, language: str, split_stats: dict,
                  tokenizer_name: str):
    """Save metadata.json alongside the .bin files."""
    meta = {
        "language": language,
        "tokenizer_model": tokenizer_name,
        "vocab_size": 10000,
        "dtype": "uint16",
        "context_length": 512,
        "conversion_timestamp": datetime.now(timezone.utc).isoformat(),
        "splits": split_stats,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta_path


# ---------------------------------------------------------------------------
# Dataset reader (memory-mapped)
# ---------------------------------------------------------------------------

class TokenDataset:
    """
    Memory-mapped dataset over a flat uint16 .bin file.

    Each sample returns:
        x: tokens[i : i + context_length]       (input)
        y: tokens[i+1 : i + context_length + 1]  (target)
    """

    def __init__(self, bin_path: Path, context_length: int = 512):
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary file not found: {self.bin_path}")

        self.data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        self.context_length = context_length
        self.n_tokens = len(self.data)

        # We need at least context_length + 1 tokens for one sample
        if self.n_tokens < context_length + 1:
            raise ValueError(
                f"Token file has {self.n_tokens} tokens but needs at least "
                f"{context_length + 1} for context_length={context_length}"
            )

    def __len__(self):
        """Number of valid starting positions."""
        return self.n_tokens - self.context_length

    def __getitem__(self, idx):
        """Return (x, y) token arrays at position idx."""
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset with {len(self)} positions"
            )
        chunk = self.data[idx : idx + self.context_length + 1].astype(np.int64)
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    def random_sample(self, rng=None):
        """Return a random (x, y) pair."""
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(0, len(self))
        return self[idx]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL corpus to tokenized .bin files"
    )
    parser.add_argument("--language", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory containing .jsonl files for this split")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Root output directory (e.g. data_bin/)")
    parser.add_argument("--tokenizer", required=True, type=Path,
                        help="Path to SentencePiece .model file")
    parser.add_argument("--text-field", default="text",
                        help="JSON field containing document text")
    args = parser.parse_args()

    lang_dir = args.output_dir / args.language
    lang_dir.mkdir(parents=True, exist_ok=True)
    output_path = lang_dir / f"{args.split}.bin"

    print(f"Converting {args.language}/{args.split}")
    print(f"  Input:     {args.input_dir}")
    print(f"  Output:    {output_path}")
    print(f"  Tokenizer: {args.tokenizer}")

    stats = convert_split(
        input_dir=args.input_dir,
        output_path=output_path,
        tokenizer_path=args.tokenizer,
        text_field=args.text_field,
    )

    # Save metadata
    tokenizer_name = args.tokenizer.name
    meta_path = save_metadata(
        lang_dir, args.language,
        {args.split: stats},
        tokenizer_name,
    )

    print(f"\nResults:")
    print(f"  Documents: {stats['documents_processed']}")
    print(f"  Skipped:   {stats['documents_skipped']}")
    print(f"  Tokens:    {stats['token_count']}")
    print(f"  Token range: [{stats['token_min']}, {stats['token_max']}]")
    print(f"  Bytes:     {stats['bytes_written']}")
    print(f"  Time:      {stats['elapsed_seconds']}s")
    print(f"  Speed:     {stats['tokens_per_second']} tok/s")
    print(f"  Metadata:  {meta_path}")


if __name__ == "__main__":
    main()
