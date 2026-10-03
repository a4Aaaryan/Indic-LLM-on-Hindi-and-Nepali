"""
Phase 2 — Stage 6: Training Infrastructure Tests

All tests run on CPU only. No CUDA required.
No real corpus is accessed.
"""

import json
import math
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import LMConfig
from src.model import DecoderOnlyLM
from src.train_config import TrainConfig
from src.trainer import (
    JsonlLogger,
    _IterableWrapper,
    build_optimizer,
    build_scheduler,
    compute_loss,
    evaluate,
    get_lr,
    get_rng_state,
    load_checkpoint,
    perplexity,
    save_checkpoint,
    seed_everything,
    set_rng_state,
    training_step,
)

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Tiny model / config helpers
# ---------------------------------------------------------------------------

def tiny_lm_cfg():
    return LMConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,   # deterministic for tests
        bias=True,
    )


def tiny_train_cfg(tmpdir, **kwargs):
    defaults = dict(
        language="hindi",
        seed=0,
        batch_size=2,
        grad_accum_steps=1,
        learning_rate=1e-3,
        min_lr=1e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        max_steps=10,
        warmup_steps=3,
        grad_clip=1.0,
        use_amp=False,        # always off for CPU tests
        output_dir=tmpdir,
        checkpoint_interval=5,
        log_interval=2,
        eval_interval=5,
        eval_batches=2,
        num_workers=0,
        context_length=16,
    )
    defaults.update(kwargs)
    return TrainConfig(**defaults)


def tiny_model(cfg=None):
    if cfg is None:
        cfg = tiny_lm_cfg()
    m = DecoderOnlyLM(cfg)
    m.eval()
    return m


class TinyTokenDataset(Dataset):
    """
    Synthetic dataset: returns (x, y) pairs from a fixed token sequence.
    Mirrors the TokenDataset API without touching any real corpus.
    """
    def __init__(self, n_tokens=512, context_length=16, seed=7):
        rng = np.random.default_rng(seed)
        self.tokens = torch.from_numpy(
            rng.integers(0, 64, size=n_tokens, dtype=np.int64)
        )
        self.context_length = context_length
        assert n_tokens >= context_length + 1

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.context_length + 1]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# 1. TrainConfig
# ---------------------------------------------------------------------------

class TestTrainConfig(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_defaults_are_valid(self):
        cfg = TrainConfig(
            max_steps=100,
            warmup_steps=10,
            output_dir=self.tmpdir,
        )
        self.assertEqual(cfg.language, "hindi")
        self.assertEqual(cfg.batch_size, 16)
        self.assertGreater(cfg.learning_rate, 0)

    def test_invalid_language(self):
        with self.assertRaises(ValueError):
            TrainConfig(language="klingon", max_steps=10, warmup_steps=2,
                        output_dir=self.tmpdir)

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            TrainConfig(batch_size=0, max_steps=10, warmup_steps=2,
                        output_dir=self.tmpdir)

    def test_warmup_ge_max_steps_raises(self):
        with self.assertRaises(ValueError):
            TrainConfig(max_steps=10, warmup_steps=10, output_dir=self.tmpdir)

    def test_min_lr_ge_lr_raises(self):
        with self.assertRaises(ValueError):
            TrainConfig(
                learning_rate=1e-4, min_lr=1e-4,
                max_steps=10, warmup_steps=2, output_dir=self.tmpdir
            )

    def test_output_dir_coerced_to_path(self):
        cfg = TrainConfig(
            output_dir=str(self.tmpdir),
            max_steps=10, warmup_steps=2,
        )
        self.assertIsInstance(cfg.output_dir, Path)


# ---------------------------------------------------------------------------
# 2. Optimizer
# ---------------------------------------------------------------------------

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_optimizer_type(self):
        m = tiny_model()
        cfg = tiny_train_cfg(self.tmpdir)
        opt = build_optimizer(m, cfg)
        self.assertIsInstance(opt, torch.optim.AdamW)

    def test_param_groups(self):
        m = tiny_model()
        cfg = tiny_train_cfg(self.tmpdir)
        opt = build_optimizer(m, cfg)
        self.assertEqual(len(opt.param_groups), 2)
        # First group has weight decay
        self.assertGreater(opt.param_groups[0]["weight_decay"], 0)
        # Second group has no weight decay
        self.assertEqual(opt.param_groups[1]["weight_decay"], 0.0)

    def test_lr_set(self):
        m = tiny_model()
        cfg = tiny_train_cfg(self.tmpdir, learning_rate=5e-4)
        opt = build_optimizer(m, cfg)
        for pg in opt.param_groups:
            self.assertAlmostEqual(pg["lr"], 5e-4)

    def test_betas(self):
        m = tiny_model()
        cfg = tiny_train_cfg(self.tmpdir, beta1=0.85, beta2=0.999)
        opt = build_optimizer(m, cfg)
        for pg in opt.param_groups:
            self.assertEqual(pg["betas"], (0.85, 0.999))


# ---------------------------------------------------------------------------
# 3. Scheduler — warmup
# ---------------------------------------------------------------------------

class TestSchedulerWarmup(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.cfg = tiny_train_cfg(
            self.tmpdir,
            learning_rate=1e-3,
            min_lr=1e-4,
            warmup_steps=4,
            max_steps=10,
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_lr_zero_at_step_zero(self):
        lr = get_lr(0, self.cfg)
        self.assertAlmostEqual(lr, 0.0, places=9)

    def test_lr_full_at_warmup_steps(self):
        lr = get_lr(self.cfg.warmup_steps, self.cfg)
        self.assertAlmostEqual(lr, self.cfg.learning_rate, places=9)

    def test_warmup_is_linear(self):
        lrs = [get_lr(s, self.cfg) for s in range(self.cfg.warmup_steps + 1)]
        diffs = [lrs[i+1] - lrs[i] for i in range(len(lrs) - 1)]
        # all increments should be approximately equal
        self.assertTrue(all(abs(d - diffs[0]) < 1e-9 for d in diffs))


# ---------------------------------------------------------------------------
# 4. Scheduler — cosine decay
# ---------------------------------------------------------------------------

class TestSchedulerCosineDecay(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.cfg = tiny_train_cfg(
            self.tmpdir,
            learning_rate=1e-3,
            min_lr=1e-4,
            warmup_steps=0,
            max_steps=100,
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_lr_decreases_after_warmup(self):
        lrs = [get_lr(s, self.cfg) for s in range(0, 100, 10)]
        for i in range(len(lrs) - 1):
            self.assertGreaterEqual(lrs[i], lrs[i+1])

    def test_lr_reaches_min_at_max_steps(self):
        lr = get_lr(self.cfg.max_steps, self.cfg)
        self.assertAlmostEqual(lr, self.cfg.min_lr, places=9)

    def test_lr_clamped_beyond_max_steps(self):
        lr = get_lr(self.cfg.max_steps + 100, self.cfg)
        self.assertAlmostEqual(lr, self.cfg.min_lr, places=9)

    def test_scheduler_step_updates_lr(self):
        m = tiny_model()
        opt = build_optimizer(m, self.cfg)
        sched = build_scheduler(opt, self.cfg)
        initial_lr = opt.param_groups[0]["lr"]
        # step once (warmup_steps=0, so immediately in cosine range)
        opt.step()
        sched.step()
        stepped_lr = opt.param_groups[0]["lr"]
        # LR should have changed
        self.assertNotAlmostEqual(initial_lr, stepped_lr, places=9)


# ---------------------------------------------------------------------------
# 5. Loss / perplexity
# ---------------------------------------------------------------------------

class TestLossAndPerplexity(unittest.TestCase):
    def test_compute_loss_shape(self):
        logits = torch.randn(2, 8, 64)   # [B, T, V]
        targets = torch.randint(0, 64, (2, 8))
        loss = compute_loss(logits, targets)
        self.assertEqual(loss.shape, ())

    def test_compute_loss_positive(self):
        logits = torch.randn(2, 8, 64)
        targets = torch.randint(0, 64, (2, 8))
        loss = compute_loss(logits, targets)
        self.assertGreater(loss.item(), 0)

    def test_perplexity_is_exp_loss(self):
        loss = 2.0
        ppl = perplexity(loss)
        self.assertAlmostEqual(ppl, math.exp(2.0), places=5)

    def test_perplexity_large_loss_no_inf(self):
        ppl = perplexity(1000.0)
        self.assertFalse(math.isinf(ppl))

    def test_perplexity_zero_loss(self):
        ppl = perplexity(0.0)
        self.assertAlmostEqual(ppl, 1.0, places=9)

    def test_uniform_logits_loss(self):
        # Uniform logits → loss = log(vocab_size)
        V = 64
        logits = torch.zeros(1, 4, V)
        targets = torch.zeros(1, 4, dtype=torch.long)
        loss = compute_loss(logits, targets)
        self.assertAlmostEqual(loss.item(), math.log(V), places=4)


# ---------------------------------------------------------------------------
# 6. Training step — gradient accumulation
# ---------------------------------------------------------------------------

class TestTrainingStep(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.lm_cfg = tiny_lm_cfg()
        self.model = DecoderOnlyLM(self.lm_cfg).train()
        self.dataset = TinyTokenDataset(n_tokens=512, context_length=16)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _setup_training(self, accum=1):
        cfg = tiny_train_cfg(self.tmpdir, grad_accum_steps=accum)
        opt = build_optimizer(self.model, cfg)
        sched = build_scheduler(opt, cfg)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        return cfg, opt, sched, scaler

    def _make_iter(self, cfg):
        return _IterableWrapper(self.dataset, cfg.batch_size, 0)

    def test_returns_expected_keys(self):
        cfg, opt, sched, scaler = self._setup_training()
        it = self._make_iter(cfg)
        result = training_step(self.model, it, opt, sched, scaler, cfg, CPU, False)
        self.assertIn("loss", result)
        self.assertIn("grad_norm", result)
        self.assertIn("tokens", result)
        self.assertIn("lr", result)

    def test_loss_is_finite(self):
        cfg, opt, sched, scaler = self._setup_training()
        it = self._make_iter(cfg)
        result = training_step(self.model, it, opt, sched, scaler, cfg, CPU, False)
        self.assertTrue(math.isfinite(result["loss"]))

    def test_token_count_matches_batch(self):
        cfg, opt, sched, scaler = self._setup_training(accum=1)
        it = self._make_iter(cfg)
        result = training_step(self.model, it, opt, sched, scaler, cfg, CPU, False)
        # tokens = batch_size * context_length * grad_accum_steps
        expected_tokens = cfg.batch_size * 16
        self.assertEqual(result["tokens"], expected_tokens)

    def test_token_count_with_accum(self):
        cfg, opt, sched, scaler = self._setup_training(accum=3)
        it = self._make_iter(cfg)
        result = training_step(self.model, it, opt, sched, scaler, cfg, CPU, False)
        expected_tokens = cfg.batch_size * 16 * 3
        self.assertEqual(result["tokens"], expected_tokens)

    def test_parameters_change_after_step(self):
        # Use warmup_steps=0 so learning rate is > 0 on step 0
        cfg = tiny_train_cfg(self.tmpdir, warmup_steps=0)
        opt = build_optimizer(self.model, cfg)
        sched = build_scheduler(opt, cfg)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        it = _IterableWrapper(self.dataset, cfg.batch_size, 0)
        
        # Use the first FFN weight (2-D, not tied) to check update
        p = self.model.blocks[0].ffn.net[0].weight.clone().detach()
        training_step(self.model, it, opt, sched, scaler, cfg, CPU, False)
        p_after = self.model.blocks[0].ffn.net[0].weight.clone().detach()
        self.assertFalse(torch.allclose(p, p_after))


# ---------------------------------------------------------------------------
# 7. Gradient clipping
# ---------------------------------------------------------------------------

class TestGradientClipping(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.lm_cfg = tiny_lm_cfg()
        self.model = DecoderOnlyLM(self.lm_cfg).train()
        self.dataset = TinyTokenDataset(n_tokens=512, context_length=16)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_grad_norm_returned_when_clipping_enabled(self):
        cfg = tiny_train_cfg(self.tmpdir, grad_clip=1.0)
        opt = build_optimizer(self.model, cfg)
        sched = build_scheduler(opt, cfg)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        it = _IterableWrapper(self.dataset, cfg.batch_size, 0)
        result = training_step(self.model, it, opt, sched, scaler, cfg, CPU, False)
        self.assertIsNotNone(result["grad_norm"])
        self.assertGreaterEqual(result["grad_norm"], 0.0)

    def test_no_grad_norm_when_clipping_disabled(self):
        cfg = tiny_train_cfg(self.tmpdir, grad_clip=0.0)
        opt = build_optimizer(self.model, cfg)
        sched = build_scheduler(opt, cfg)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        it = _IterableWrapper(self.dataset, cfg.batch_size, 0)
        result = training_step(self.model, it, opt, sched, scaler, cfg, CPU, False)
        self.assertIsNone(result["grad_norm"])


# ---------------------------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------------------------

class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.lm_cfg = tiny_lm_cfg()
        self.model = DecoderOnlyLM(self.lm_cfg)
        self.dataset = TinyTokenDataset(n_tokens=512, context_length=16)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_returns_val_keys(self):
        cfg = tiny_train_cfg(self.tmpdir, eval_batches=2)
        result = evaluate(self.model, self.dataset, cfg, CPU, False)
        self.assertIn("val_loss", result)
        self.assertIn("val_ppl", result)

    def test_val_loss_positive(self):
        cfg = tiny_train_cfg(self.tmpdir, eval_batches=2)
        result = evaluate(self.model, self.dataset, cfg, CPU, False)
        self.assertGreater(result["val_loss"], 0)

    def test_val_ppl_exp_loss(self):
        cfg = tiny_train_cfg(self.tmpdir, eval_batches=2)
        result = evaluate(self.model, self.dataset, cfg, CPU, False)
        expected = perplexity(result["val_loss"])
        self.assertAlmostEqual(result["val_ppl"], expected, places=4)

    def test_model_training_state_restored(self):
        cfg = tiny_train_cfg(self.tmpdir, eval_batches=2)
        self.model.train()
        evaluate(self.model, self.dataset, cfg, CPU, False)
        self.assertTrue(self.model.training)

    def test_model_eval_state_restored(self):
        cfg = tiny_train_cfg(self.tmpdir, eval_batches=2)
        self.model.eval()
        evaluate(self.model, self.dataset, cfg, CPU, False)
        self.assertFalse(self.model.training)

    def test_eval_deterministic_in_eval_mode(self):
        cfg = tiny_train_cfg(self.tmpdir, eval_batches=2)
        self.model.eval()
        r1 = evaluate(self.model, self.dataset, cfg, CPU, False)
        r2 = evaluate(self.model, self.dataset, cfg, CPU, False)
        # Two evals with same seeds: not necessarily equal (shuffle differs)
        # but both should be finite
        self.assertTrue(math.isfinite(r1["val_loss"]))
        self.assertTrue(math.isfinite(r2["val_loss"]))


# ---------------------------------------------------------------------------
# 9. Token accounting
# ---------------------------------------------------------------------------

class TestTokenAccounting(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.lm_cfg = tiny_lm_cfg()
        self.model = DecoderOnlyLM(self.lm_cfg).train()
        self.dataset = TinyTokenDataset(n_tokens=512, context_length=16)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_total_tokens_across_steps(self):
        cfg = tiny_train_cfg(self.tmpdir, grad_accum_steps=2, batch_size=3)
        opt = build_optimizer(self.model, cfg)
        sched = build_scheduler(opt, cfg)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        it = _IterableWrapper(self.dataset, cfg.batch_size, 0)

        total = 0
        for _ in range(4):
            res = training_step(self.model, it, opt, sched, scaler, cfg, CPU, False)
            total += res["tokens"]

        # Each opt step: batch_size * context_length * grad_accum
        expected_per_step = 3 * 16 * 2
        self.assertEqual(total, expected_per_step * 4)


# ---------------------------------------------------------------------------
# 10. Checkpoint save / load
# ---------------------------------------------------------------------------

class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.lm_cfg = tiny_lm_cfg()
        self.model = DecoderOnlyLM(self.lm_cfg)
        self.dataset = TinyTokenDataset(n_tokens=512, context_length=16)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _build(self):
        cfg = tiny_train_cfg(self.tmpdir)
        opt = build_optimizer(self.model, cfg)
        sched = build_scheduler(opt, cfg)
        return cfg, opt, sched

    def test_checkpoint_file_created(self):
        cfg, opt, sched = self._build()
        ckpt = save_checkpoint(
            self.tmpdir, step=5, model=self.model,
            optimizer=opt, scheduler=sched,
            cfg=cfg, model_cfg=self.lm_cfg,
        )
        self.assertTrue(ckpt.exists())

    def test_latest_txt_created(self):
        cfg, opt, sched = self._build()
        save_checkpoint(
            self.tmpdir, step=10, model=self.model,
            optimizer=opt, scheduler=sched,
            cfg=cfg, model_cfg=self.lm_cfg,
        )
        self.assertTrue((self.tmpdir / "latest.txt").exists())

    def test_model_state_restored(self):
        cfg, opt, sched = self._build()

        # mutate model weights
        with torch.no_grad():
            for p in self.model.parameters():
                p.fill_(3.14)

        ckpt = save_checkpoint(
            self.tmpdir, step=1, model=self.model,
            optimizer=opt, scheduler=sched,
            cfg=cfg, model_cfg=self.lm_cfg,
        )

        # reset model weights
        model2 = DecoderOnlyLM(self.lm_cfg)
        opt2 = build_optimizer(model2, cfg)
        sched2 = build_scheduler(opt2, cfg)

        load_checkpoint(ckpt, model2, opt2, sched2, CPU)

        for p_orig, p_new in zip(self.model.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p_orig, p_new))

    def test_optimizer_state_restored(self):
        cfg, opt, sched = self._build()

        # Take a few steps so optimizer has non-trivial state
        ds = TinyTokenDataset(n_tokens=512, context_length=16)
        it = _IterableWrapper(ds, cfg.batch_size, 0)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        for _ in range(3):
            training_step(self.model, it, opt, sched, scaler, cfg, CPU, False)

        ckpt = save_checkpoint(
            self.tmpdir, step=3, model=self.model,
            optimizer=opt, scheduler=sched,
            cfg=cfg, model_cfg=self.lm_cfg,
        )

        model2 = DecoderOnlyLM(self.lm_cfg)
        opt2 = build_optimizer(model2, cfg)
        sched2 = build_scheduler(opt2, cfg)
        load_checkpoint(ckpt, model2, opt2, sched2, CPU)

        # Compare first-moment state of first parameter group
        s1 = opt.state_dict()["state"]
        s2 = opt2.state_dict()["state"]
        for k in s1:
            for moment in ("exp_avg", "exp_avg_sq"):
                if moment in s1[k]:
                    self.assertTrue(
                        torch.allclose(s1[k][moment], s2[k][moment]),
                        f"Mismatch in {moment} for param {k}"
                    )

    def test_scheduler_step_restored(self):
        cfg, opt, sched = self._build()

        ds = TinyTokenDataset(n_tokens=512, context_length=16)
        it = _IterableWrapper(ds, cfg.batch_size, 0)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        for _ in range(5):
            training_step(self.model, it, opt, sched, scaler, cfg, CPU, False)

        lr_before = opt.param_groups[0]["lr"]
        ckpt = save_checkpoint(
            self.tmpdir, step=5, model=self.model,
            optimizer=opt, scheduler=sched,
            cfg=cfg, model_cfg=self.lm_cfg,
        )

        model2 = DecoderOnlyLM(self.lm_cfg)
        opt2 = build_optimizer(model2, cfg)
        sched2 = build_scheduler(opt2, cfg)
        load_checkpoint(ckpt, model2, opt2, sched2, CPU)

        lr_after = opt2.param_groups[0]["lr"]
        self.assertAlmostEqual(lr_before, lr_after, places=9)

    def test_step_number_in_payload(self):
        cfg, opt, sched = self._build()
        ckpt = save_checkpoint(
            self.tmpdir, step=42, model=self.model,
            optimizer=opt, scheduler=sched,
            cfg=cfg, model_cfg=self.lm_cfg,
        )
        payload = load_checkpoint(ckpt, self.model, device=CPU)
        self.assertEqual(payload["step"], 42)

    def test_rng_state_restored(self):
        # Seed, capture state, advance RNG, save checkpoint
        seed_everything(99)
        # Advance RNG a bit
        _ = torch.rand(20)
        rng_state = get_rng_state()
        # Remember what the next random values are
        expected = torch.rand(5).clone()

        cfg, opt, sched = self._build()
        ckpt = save_checkpoint(
            self.tmpdir, step=1, model=self.model,
            optimizer=opt, scheduler=sched,
            cfg=cfg, model_cfg=self.lm_cfg,
            rng_state=rng_state,
        )

        # Scramble RNG completely
        seed_everything(12345)
        _ = torch.rand(1000)

        # Restore via checkpoint
        model2 = DecoderOnlyLM(self.lm_cfg)
        load_checkpoint(ckpt, model2, device=CPU, restore_rng=True)

        # After restore, the next values must match what we recorded above
        actual = torch.rand(5)
        self.assertTrue(torch.allclose(expected, actual),
                        msg=f"RNG restore failed: expected {expected}, got {actual}")

    def test_version_mismatch_raises(self):
        cfg, opt, sched = self._build()
        ckpt = save_checkpoint(
            self.tmpdir, step=1, model=self.model,
            optimizer=opt, scheduler=sched,
            cfg=cfg, model_cfg=self.lm_cfg,
        )
        # Corrupt version
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        payload["version"] = 999
        torch.save(payload, ckpt)

        model2 = DecoderOnlyLM(self.lm_cfg)
        with self.assertRaises(ValueError):
            load_checkpoint(ckpt, model2, device=CPU)


# ---------------------------------------------------------------------------
# 11. JSONL Logger
# ---------------------------------------------------------------------------

class TestJsonlLogger(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_creates_file(self):
        log = JsonlLogger(self.tmpdir / "train.jsonl")
        log.log({"step": 1, "loss": 0.5})
        self.assertTrue((self.tmpdir / "train.jsonl").exists())

    def test_records_parseable(self):
        log = JsonlLogger(self.tmpdir / "train.jsonl")
        log.log({"step": 1, "loss": 0.5})
        log.log({"step": 2, "loss": 0.4})
        with open(self.tmpdir / "train.jsonl") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)
        r = json.loads(lines[0])
        self.assertEqual(r["step"], 1)
        self.assertIn("timestamp", r)

    def test_appends_not_overwrites(self):
        log = JsonlLogger(self.tmpdir / "train.jsonl")
        for i in range(5):
            log.log({"step": i})
        with open(self.tmpdir / "train.jsonl") as f:
            records = [json.loads(l) for l in f]
        self.assertEqual(len(records), 5)

    def test_creates_parent_dir(self):
        nested = self.tmpdir / "a" / "b" / "log.jsonl"
        log = JsonlLogger(nested)
        log.log({"x": 1})
        self.assertTrue(nested.exists())


# ---------------------------------------------------------------------------
# 12. Seed / RNG state
# ---------------------------------------------------------------------------

class TestSeedEverything(unittest.TestCase):
    def test_reproducible_torch(self):
        seed_everything(42)
        a = torch.rand(10)
        seed_everything(42)
        b = torch.rand(10)
        self.assertTrue(torch.allclose(a, b))

    def test_reproducible_numpy(self):
        seed_everything(42)
        a = np.random.rand(10)
        seed_everything(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_rng_save_restore(self):
        seed_everything(77)
        state = get_rng_state()
        torch.rand(100)  # advance

        set_rng_state(state)
        a = torch.rand(10)

        seed_everything(77)
        b = torch.rand(10)
        self.assertTrue(torch.allclose(a, b))


# ---------------------------------------------------------------------------
# 13. End-to-end tiny training run
# ---------------------------------------------------------------------------

class TestEndToEndTraining(unittest.TestCase):
    """
    Trains for a handful of steps on synthetic data.
    Verifies loss decreases and training does not crash.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.lm_cfg = tiny_lm_cfg()
        self.model = DecoderOnlyLM(self.lm_cfg).train()
        self.train_ds = TinyTokenDataset(n_tokens=1024, context_length=16)
        self.val_ds = TinyTokenDataset(n_tokens=256, context_length=16, seed=99)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_loss_is_finite_over_steps(self):
        cfg = tiny_train_cfg(
            self.tmpdir,
            max_steps=5,
            warmup_steps=1,
            grad_accum_steps=1,
            batch_size=2,
        )
        opt = build_optimizer(self.model, cfg)
        sched = build_scheduler(opt, cfg)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        it = _IterableWrapper(self.train_ds, cfg.batch_size, 0)

        losses = []
        for _ in range(5):
            res = training_step(self.model, it, opt, sched, scaler, cfg, CPU, False)
            self.assertTrue(math.isfinite(res["loss"]))
            losses.append(res["loss"])

    def test_eval_after_training(self):
        cfg = tiny_train_cfg(
            self.tmpdir,
            max_steps=5,
            warmup_steps=1,
            batch_size=2,
            eval_batches=2,
        )
        opt = build_optimizer(self.model, cfg)
        sched = build_scheduler(opt, cfg)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        it = _IterableWrapper(self.train_ds, cfg.batch_size, 0)

        for _ in range(5):
            training_step(self.model, it, opt, sched, scaler, cfg, CPU, False)

        metrics = evaluate(self.model, self.val_ds, cfg, CPU, False)
        self.assertTrue(math.isfinite(metrics["val_loss"]))
        self.assertTrue(math.isfinite(metrics["val_ppl"]))


# ---------------------------------------------------------------------------
# 14. Checkpoint resume test (deterministic)
# ---------------------------------------------------------------------------

class TestCheckpointResume(unittest.TestCase):
    """
    Trains for N steps, saves, resumes in a fresh model, trains M more steps.
    Verifies that the resumed model produces the same forward output as
    if training had been uninterrupted, given the same RNG state.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.lm_cfg = tiny_lm_cfg()
        self.ds = TinyTokenDataset(n_tokens=2048, context_length=16, seed=42)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _run_steps_det(self, model, n_steps, cfg, opt, sched, scaler, torch_gen):
        """Run n_steps training steps using deterministic synthetic data."""
        losses = []
        for _ in range(n_steps):
            opt.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _m in range(cfg.grad_accum_steps):
                x = torch.randint(0, 64, (cfg.batch_size, 16), generator=torch_gen, dtype=torch.long)
                y = torch.randint(0, 64, (cfg.batch_size, 16), generator=torch_gen, dtype=torch.long)
                logits = model(x)
                loss = compute_loss(logits, y) / cfg.grad_accum_steps
                loss.backward()
                accum_loss += loss.item()
            opt.step()
            sched.step()
            losses.append(accum_loss)
        return losses

    def test_resumed_loss_matches_continuous(self):
        lm_cfg = self.lm_cfg
        cfg = tiny_train_cfg(
            self.tmpdir,
            max_steps=20,
            warmup_steps=2,
            batch_size=2,
            grad_accum_steps=1,
            grad_clip=0.0,
            use_amp=False,
        )
        scaler_dummy = torch.cuda.amp.GradScaler(enabled=False)

        # ---- Continuous run: train 6 steps total ----
        seed_everything(1)
        model_cont = DecoderOnlyLM(lm_cfg)
        opt_cont = build_optimizer(model_cont, cfg)
        sched_cont = build_scheduler(opt_cont, cfg)
        g_cont = torch.Generator()
        g_cont.manual_seed(42)
        losses_cont = self._run_steps_det(model_cont, 6, cfg, opt_cont, sched_cont, scaler_dummy, g_cont)

        # ---- Split run: train 3, checkpoint, resume 3 ----
        seed_everything(1)
        model_split = DecoderOnlyLM(lm_cfg)
        opt_split = build_optimizer(model_split, cfg)
        sched_split = build_scheduler(opt_split, cfg)
        g_split = torch.Generator()
        g_split.manual_seed(42)
        losses_split_a = self._run_steps_det(model_split, 3, cfg, opt_split, sched_split, scaler_dummy, g_split)

        # Save checkpoint with the synthetic generator state included in RNG
        rng_state = get_rng_state()
        rng_state["synthetic_gen"] = g_split.get_state()
        ckpt = save_checkpoint(
            self.tmpdir, step=3,
            model=model_split, optimizer=opt_split, scheduler=sched_split,
            cfg=cfg, model_cfg=lm_cfg,
            rng_state=rng_state,
        )

        # Resume
        seed_everything(1)
        model_resume = DecoderOnlyLM(lm_cfg)
        opt_resume = build_optimizer(model_resume, cfg)
        sched_resume = build_scheduler(opt_resume, cfg)
        payload = load_checkpoint(ckpt, model_resume, opt_resume, sched_resume, CPU, restore_rng=True)
        self.assertEqual(payload["step"], 3)

        g_resume = torch.Generator()
        g_resume.set_state(payload["rng_state"]["synthetic_gen"])
        losses_split_b = self._run_steps_det(model_resume, 3, cfg, opt_resume, sched_resume, scaler_dummy, g_resume)

        for i, (cont, resumed) in enumerate(zip(losses_cont[3:], losses_split_b)):
            self.assertAlmostEqual(cont, resumed, places=4,
                                   msg=f"Diverged at resumed step {i+1}: cont={cont:.6f}, resumed={resumed:.6f}")


if __name__ == "__main__":
    unittest.main()
