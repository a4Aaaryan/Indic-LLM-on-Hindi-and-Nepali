"""Streaming downloader and processor for Indic HPLT v2 corpus with resumable checkpointing,
pre-tokenization heuristic budget stopping, and gzip compression support to minimize local storage usage.

NOTE ON METHODOLOGY:
The token count calculated here (`approx_pretokenized_tokens`) is a pre-tokenization ingestion heuristic
(derived from whitespace word count * 1.35) used strictly to prevent unbounded downloads and protect local disk space.
It is NOT the official tokenizer token count. Official token counts and the >=20% manual token ratio
are measured in Stage 5 and Stage 9 using the student-trained SentencePiece tokenizers.
"""

import gzip
import json
from pathlib import Path
from typing import Iterator, Optional
from datasets import load_dataset

from common.config import ensure_dir, resolve_path
from common.schema import DocumentRecord
from common.utils import setup_logger

# Pre-tokenization heuristic multiplier for Devanagari script (words -> approximate subwords)
# Used solely for ingestion throttling during Stage 1 data collection.
HEURISTIC_TOKENS_PER_WORD = 1.35


class HPLTStreamDownloader:
    """Streams and filters Indic HPLT v2 subset with checkpointing, budget controls, and compression."""

    def __init__(
        self,
        language: str,
        output_file: str | Path,
        checkpoint_file: Optional[str | Path] = None,
        compress: bool = False,
    ):
        self.language = language
        self.compress = compress
        raw_output_path = resolve_path(output_file)

        # Append .gz if compression is requested and not already in path
        if self.compress and not str(raw_output_path).endswith(".gz"):
            self.output_file = Path(str(raw_output_path) + ".gz")
        else:
            self.output_file = raw_output_path
            if str(self.output_file).endswith(".gz"):
                self.compress = True

        ensure_dir(self.output_file.parent)
        self.checkpoint_file = (
            resolve_path(checkpoint_file)
            if checkpoint_file
            else self.output_file.parent / f"hplt_{language}_checkpoint.json"
        )
        self.logger = setup_logger(f"hplt_{language}")

    def _load_checkpoint(self) -> dict:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "saved_count": 0,
            "accumulated_words": 0,
            "approx_pretokenized_tokens": 0,
            "last_doc_id": None,
        }

    def _save_checkpoint(self, saved_count: int, accumulated_words: int, approx_tokens: int, last_doc_id: Optional[str]):
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "saved_count": saved_count,
                    "accumulated_words": accumulated_words,
                    "approx_pretokenized_tokens": approx_tokens,
                    "last_doc_id": last_doc_id,
                },
                f,
                indent=2,
            )

    def _open_writer(self, mode: str = "a"):
        """Open output file with transparent gzip support."""
        if self.compress or str(self.output_file).endswith(".gz"):
            return gzip.open(self.output_file, mode + "t", encoding="utf-8")
        return open(self.output_file, mode, encoding="utf-8")

    def stream_records(
        self,
        max_samples: Optional[int] = None,
        target_approx_tokens: Optional[int] = None,
        target_tokens: Optional[int] = None,
    ) -> Iterator[DocumentRecord]:
        """Stream filtered language records stopping when sample limit or heuristic token budget is reached."""
        effective_target_tokens = target_approx_tokens if target_approx_tokens is not None else target_tokens

        self.logger.info("Connecting to Hugging Face streaming dataset: ashtok897/indic-hplt-v2")
        checkpoint = self._load_checkpoint()
        already_saved = checkpoint.get("saved_count", 0)
        accumulated_words = checkpoint.get("accumulated_words", 0)
        accumulated_approx_tokens = checkpoint.get("approx_pretokenized_tokens", 0)

        if effective_target_tokens and accumulated_approx_tokens >= effective_target_tokens:
            self.logger.info("Pre-tokenization token budget (%d) already met according to checkpoint.", effective_target_tokens)
            return

        dataset = load_dataset("ashtok897/indic-hplt-v2", split="train", streaming=True)

        yielded_count = 0
        for item in dataset:
            if item.get("lang") != self.language:
                continue

            text = item.get("text", "")
            if not text or len(text.strip()) < 50:
                continue

            doc_id_raw = item.get("doc_id", f"doc_{already_saved + yielded_count}")
            doc_id = f"hplt_{self.language}_{doc_id_raw}"

            # Word and heuristic token estimation
            words_in_doc = len(text.split())
            heuristic_tokens_in_doc = int(words_in_doc * HEURISTIC_TOKENS_PER_WORD)

            record = DocumentRecord(
                doc_id=doc_id,
                language=self.language,
                source_type="downloaded",
                source_name="indic_hplt_v2",
                text=text,
                url=item.get("url"),
                title=None,
                raw_location="hf://ashtok897/indic-hplt-v2",
                metadata={
                    "prob": item.get("prob"),
                    "score": item.get("score"),
                    "web_register": item.get("web-register"),
                    "char_count": len(text),
                    "word_count": words_in_doc,
                    "approx_pretokenized_tokens_heuristic": heuristic_tokens_in_doc,
                },
            )

            yield record
            yielded_count += 1
            accumulated_words += words_in_doc
            accumulated_approx_tokens += heuristic_tokens_in_doc

            # Budget checks
            if max_samples is not None and yielded_count >= max_samples:
                self.logger.info("Reached sample limit: %d documents.", max_samples)
                break

            if effective_target_tokens is not None and accumulated_approx_tokens >= effective_target_tokens:
                self.logger.info(
                    "Reached pre-tokenization token budget: %d heuristic tokens (~%d words, %d docs).",
                    accumulated_approx_tokens,
                    accumulated_words,
                    already_saved + yielded_count,
                )
                break

    def download_to_file(
        self,
        max_samples: Optional[int] = None,
        target_approx_tokens: Optional[int] = None,
        target_tokens: Optional[int] = None,
    ) -> int:
        """Download and append records directly to output file (optionally gzipped)."""
        effective_target = target_approx_tokens if target_approx_tokens is not None else target_tokens

        checkpoint = self._load_checkpoint()
        saved_count = checkpoint.get("saved_count", 0)
        accumulated_words = checkpoint.get("accumulated_words", 0)
        accumulated_approx_tokens = checkpoint.get("approx_pretokenized_tokens", 0)

        mode = "a" if self.output_file.exists() else "w"
        with self._open_writer(mode=mode) as f:
            for record in self.stream_records(max_samples=max_samples, target_approx_tokens=effective_target):
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                saved_count += 1
                words = record.metadata.get("word_count", len(record.text.split()))
                tokens = record.metadata.get(
                    "approx_pretokenized_tokens_heuristic",
                    int(words * HEURISTIC_TOKENS_PER_WORD),
                )
                accumulated_words += words
                accumulated_approx_tokens += tokens

                if saved_count % 5000 == 0:
                    self._save_checkpoint(saved_count, accumulated_words, accumulated_approx_tokens, record.doc_id)
                    file_mb = self.output_file.stat().st_size / (1024 * 1024) if self.output_file.exists() else 0
                    self.logger.info(
                        "Saved %d docs | ~%.2fM approx pre-tok tokens | File size: %.2f MB (%s)",
                        saved_count,
                        accumulated_approx_tokens / 1e6,
                        file_mb,
                        "compressed" if self.compress else "uncompressed",
                    )

        self._save_checkpoint(saved_count, accumulated_words, accumulated_approx_tokens, None)
        final_mb = self.output_file.stat().st_size / (1024 * 1024) if self.output_file.exists() else 0
        self.logger.info(
            "Finished HPLT download. Total saved: %d docs | ~%.2fM approx pre-tok tokens | Final size: %.2f MB",
            saved_count,
            accumulated_approx_tokens / 1e6,
            final_mb,
        )
        return saved_count
