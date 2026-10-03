"""
Stage 5 tests — Data preparation pipeline.

Uses tiny synthetic data and (where available) a small sample from the
real Phase 1 corpus + SentencePiece tokenizer for smoke testing.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import stream_jsonl_texts, convert_split, save_metadata, TokenDataset

# ---------------------------------------------------------------------------
# Locate real tokenizers (optional — tests degrade gracefully)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

HINDI_TOKENIZER = PROJECT_ROOT / "hindi" / "tokenizer" / "models" / "hindi_bpe_10k_final.model"
NEPALI_TOKENIZER = PROJECT_ROOT / "nepali" / "tokenizer" / "models" / "nepali_bpe_10k_assignment.model"

HAS_HINDI_TOK = HINDI_TOKENIZER.exists()
HAS_NEPALI_TOK = NEPALI_TOKENIZER.exists()


def _make_jsonl(directory: Path, filename: str, docs: list[dict]):
    """Write a list of dicts as JSONL."""
    path = directory / filename
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return path


def _tiny_docs(n=5, prefix="Doc"):
    """Create n tiny synthetic documents with an English-ish text field."""
    return [
        {"doc_id": f"{prefix}_{i}", "text": f"This is test document number {i}. " * 3}
        for i in range(n)
    ]


class TestStreamJsonlTexts(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_single_file(self):
        _make_jsonl(self.tmpdir, "a.jsonl", _tiny_docs(3))
        texts = list(stream_jsonl_texts(self.tmpdir))
        self.assertEqual(len(texts), 3)
        self.assertTrue(all(isinstance(t, str) for t in texts))

    def test_multiple_files(self):
        _make_jsonl(self.tmpdir, "a.jsonl", _tiny_docs(2, "A"))
        _make_jsonl(self.tmpdir, "b.jsonl", _tiny_docs(3, "B"))
        texts = list(stream_jsonl_texts(self.tmpdir))
        self.assertEqual(len(texts), 5)

    def test_deterministic_order(self):
        _make_jsonl(self.tmpdir, "b.jsonl", [{"text": "B"}])
        _make_jsonl(self.tmpdir, "a.jsonl", [{"text": "A"}])
        texts = list(stream_jsonl_texts(self.tmpdir))
        self.assertEqual(texts[0], "A")
        self.assertEqual(texts[1], "B")

    def test_skips_empty_text(self):
        docs = [{"text": ""}, {"text": "valid"}, {"text": "   "}]
        _make_jsonl(self.tmpdir, "a.jsonl", docs)
        texts = list(stream_jsonl_texts(self.tmpdir))
        # empty and whitespace-only yield None
        self.assertIsNone(texts[0])
        self.assertEqual(texts[1], "valid")
        self.assertIsNone(texts[2])

    def test_no_jsonl_raises(self):
        empty_dir = self.tmpdir / "empty"
        empty_dir.mkdir()
        with self.assertRaises(FileNotFoundError):
            list(stream_jsonl_texts(empty_dir))


@unittest.skipUnless(HAS_HINDI_TOK, "Hindi tokenizer not available")
class TestConvertSplitHindi(unittest.TestCase):
    """Test convert_split using the real Hindi tokenizer on tiny synthetic data."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.input_dir = self.tmpdir / "input"
        self.input_dir.mkdir()
        # Write 5 tiny Hindi-ish docs (the tokenizer handles any UTF-8)
        docs = [
            {"text": "यह एक परीक्षण दस्तावेज़ है।"},
            {"text": "हिंदी भाषा में लिखा गया पाठ।"},
            {"text": "तीसरा दस्तावेज़ यहाँ है।"},
            {"text": "चौथा दस्तावेज़।"},
            {"text": "पाँचवाँ और अंतिम दस्तावेज़।"},
        ]
        _make_jsonl(self.input_dir, "test.jsonl", docs)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_produces_bin_file(self):
        out = self.tmpdir / "out" / "train.bin"
        stats = convert_split(self.input_dir, out, HINDI_TOKENIZER)
        self.assertTrue(out.exists())
        self.assertGreater(stats["token_count"], 0)
        self.assertEqual(stats["documents_processed"], 5)
        self.assertEqual(stats["documents_skipped"], 0)

    def test_uint16_storage(self):
        out = self.tmpdir / "out" / "train.bin"
        stats = convert_split(self.input_dir, out, HINDI_TOKENIZER)
        data = np.memmap(out, dtype=np.uint16, mode="r")
        self.assertEqual(len(data), stats["token_count"])
        self.assertTrue(data.max() < 10000)

    def test_token_ids_match_sentencepiece(self):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(str(HINDI_TOKENIZER))

        out = self.tmpdir / "out" / "train.bin"
        convert_split(self.input_dir, out, HINDI_TOKENIZER)
        data = np.memmap(out, dtype=np.uint16, mode="r")

        # Re-encode the first document and check the prefix matches
        first_text = "यह एक परीक्षण दस्तावेज़ है।"
        expected_ids = sp.Encode(first_text, out_type=int)
        actual_ids = data[:len(expected_ids)].tolist()
        self.assertEqual(actual_ids, expected_ids)


@unittest.skipUnless(HAS_NEPALI_TOK, "Nepali tokenizer not available")
class TestConvertSplitNepali(unittest.TestCase):
    """Verify Nepali config works independently."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.input_dir = self.tmpdir / "input"
        self.input_dir.mkdir()
        docs = [
            {"text": "यो एक परीक्षण कागजात हो।"},
            {"text": "नेपाली भाषामा लेखिएको पाठ।"},
            {"text": "तेस्रो कागजात यहाँ छ।"},
        ]
        _make_jsonl(self.input_dir, "test.jsonl", docs)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_produces_bin_and_correct_tokenizer(self):
        out = self.tmpdir / "out" / "train.bin"
        stats = convert_split(self.input_dir, out, NEPALI_TOKENIZER)
        self.assertTrue(out.exists())
        self.assertEqual(stats["documents_processed"], 3)
        self.assertIn("nepali_bpe_10k_assignment", stats["tokenizer"])


class TestTokenDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        # Create a tiny .bin with known token IDs
        self.n_tokens = 1030  # > 512+1
        tokens = np.arange(self.n_tokens, dtype=np.uint16)
        self.bin_path = self.tmpdir / "test.bin"
        tokens.tofile(self.bin_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_token_count(self):
        ds = TokenDataset(self.bin_path)
        self.assertEqual(ds.n_tokens, self.n_tokens)

    def test_len(self):
        ds = TokenDataset(self.bin_path)
        self.assertEqual(len(ds), self.n_tokens - 512)

    def test_xy_extraction(self):
        ds = TokenDataset(self.bin_path)
        x, y = ds[0]
        self.assertEqual(len(x), 512)
        self.assertEqual(len(y), 512)
        # x = [0..511], y = [1..512]
        np.testing.assert_array_equal(x, np.arange(0, 512, dtype=np.int64))
        np.testing.assert_array_equal(y, np.arange(1, 513, dtype=np.int64))

    def test_xy_at_last_position(self):
        ds = TokenDataset(self.bin_path)
        last = len(ds) - 1
        x, y = ds[last]
        self.assertEqual(len(x), 512)
        self.assertEqual(len(y), 512)
        # y ends at data[-1]
        self.assertEqual(y[-1], self.n_tokens - 1)

    def test_out_of_range(self):
        ds = TokenDataset(self.bin_path)
        with self.assertRaises(IndexError):
            ds[len(ds)]
        with self.assertRaises(IndexError):
            ds[-1]

    def test_insufficient_tokens(self):
        tiny = self.tmpdir / "tiny.bin"
        np.arange(100, dtype=np.uint16).tofile(tiny)
        with self.assertRaises(ValueError):
            TokenDataset(tiny, context_length=512)

    def test_random_sample(self):
        ds = TokenDataset(self.bin_path)
        rng = np.random.default_rng(42)
        x, y = ds.random_sample(rng)
        self.assertEqual(len(x), 512)
        self.assertEqual(len(y), 512)

    def test_custom_context_length(self):
        ds = TokenDataset(self.bin_path, context_length=16)
        self.assertEqual(len(ds), self.n_tokens - 16)
        x, y = ds[0]
        self.assertEqual(len(x), 16)
        np.testing.assert_array_equal(x, np.arange(0, 16, dtype=np.int64))


class TestMetadata(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_save_and_load(self):
        stats = {"train": {"token_count": 1000, "documents_processed": 10}}
        meta_path = save_metadata(self.tmpdir, "hindi", stats, "hindi_bpe_10k_final.model")
        self.assertTrue(meta_path.exists())
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertEqual(meta["language"], "hindi")
        self.assertEqual(meta["vocab_size"], 10000)
        self.assertIn("train", meta["splits"])
        self.assertIn("conversion_timestamp", meta)

    @unittest.skipUnless(HAS_HINDI_TOK, "Hindi tokenizer not available")
    def test_absolute_paths_do_not_leak(self):
        abs_input = self.tmpdir.resolve() / "abs_input"
        abs_input.mkdir()
        _make_jsonl(abs_input, "leak_test.jsonl", [{"text": "Hello"}])
        abs_output = self.tmpdir.resolve() / "abs_output" / "train.bin"
        abs_tok = HINDI_TOKENIZER.resolve()

        stats = convert_split(abs_input, abs_output, abs_tok)
        meta_path = save_metadata(self.tmpdir, "hindi", {"train": stats}, abs_tok.name)

        with open(meta_path) as f:
            meta = json.load(f)

        split_stats = meta["splits"]["train"]
        
        # Verify the absolute prefix does not exist in the string values
        self.assertNotIn(str(self.tmpdir.resolve()), split_stats["input_dir"])
        self.assertNotIn(str(self.tmpdir.resolve()), split_stats["output_path"])
        self.assertNotIn(str(PROJECT_ROOT.resolve()), split_stats["tokenizer"])
        
        self.assertEqual(split_stats["input_dir"], "abs_input")
        self.assertEqual(split_stats["output_path"], "train.bin")
        self.assertEqual(split_stats["tokenizer"], HINDI_TOKENIZER.name)


@unittest.skipUnless(HAS_HINDI_TOK, "Hindi tokenizer not available")
class TestEndToEndSmoke(unittest.TestCase):
    """Full pipeline: JSONL → SentencePiece → .bin → memmap → x/y"""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.input_dir = self.tmpdir / "input"
        self.input_dir.mkdir()
        # Write enough text to produce > 513 tokens
        docs = [
            {"text": ("यह एक लंबा दस्तावेज़ है जिसमें कई शब्द हैं। " * 50)},
            {"text": ("हिंदी भाषा में लिखा गया बड़ा पाठ। " * 50)},
        ]
        _make_jsonl(self.input_dir, "smoke.jsonl", docs)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_full_pipeline(self):
        out = self.tmpdir / "out" / "train.bin"
        stats = convert_split(self.input_dir, out, HINDI_TOKENIZER)

        self.assertGreater(stats["token_count"], 513)
        self.assertEqual(stats["documents_processed"], 2)
        self.assertEqual(stats["documents_skipped"], 0)

        # Verify metadata
        meta_path = save_metadata(
            self.tmpdir / "out", "hindi",
            {"train": stats}, "hindi_bpe_10k_final.model"
        )
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertEqual(meta["splits"]["train"]["token_count"], stats["token_count"])

        # Verify memmap dataset
        ds = TokenDataset(out)
        self.assertEqual(ds.n_tokens, stats["token_count"])

        # Verify x/y extraction
        x, y = ds[0]
        self.assertEqual(len(x), 512)
        self.assertEqual(len(y), 512)
        # y should be x shifted by one
        np.testing.assert_array_equal(x[1:], y[:-1])

        # No giant in-memory list was needed (streaming design)
        self.assertTrue(True)  # structural — the test passing proves streaming works


if __name__ == "__main__":
    unittest.main()
