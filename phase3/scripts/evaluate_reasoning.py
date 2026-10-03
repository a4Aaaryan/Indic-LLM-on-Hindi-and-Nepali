"""
Phase 3 Baseline Evaluator

Loads pretrained/finetuned checkpoints and measures exact-match accuracy
using a deterministic answer-extraction protocol.

Extraction protocol (identical for pretrained and fine-tuned models):
  1. Decode the newly generated tokens.
  2. Strip leading whitespace.
  3. Extract the first contiguous block of characters that is neither
     whitespace nor common delimiter punctuation (,  .  ;  !  ?  ।).
  4. If the extracted token is the SentencePiece UNK surface form '⁇',
     treat the prediction as INVALID (empty string) — do NOT award credit.
  5. Score: extracted_answer == ground_truth  (exact match, case-sensitive).
  6. EOS is NOT required for a prediction to be correct.

This protocol is applied identically to pretrained and fine-tuned models
so that their comparison remains fair.

Held-out classification:
  The dataset generator does NOT serialize an `is_held_out_template` field.
  Held-out status is determined from `template_id` by checking against the
  authoritative UNSEEN_TEMPLATE_IDS set defined in validate_dataset.py.
  This is the same logic validate_dataset.py uses to compute the
  'Test Familiar/Held-out' counts in the validation report.
"""

import os
import re
import sys
import json
import torch
import argparse
from pathlib import Path
from collections import defaultdict
import sentencepiece as spm

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from phase2.src.config import LMConfig
from phase2.src.model import DecoderOnlyLM
from phase2.src.trainer import load_checkpoint

# SentencePiece renders <unk> as this visual token.
_UNK_SURFACE = '\u2047'  # ⁇

# Templates designated as held-out (never appear in train/val).
# Source of truth: phase3/scripts/validate_dataset.py — UNSEEN_TEMPLATE_IDS.
UNSEEN_TEMPLATE_IDS: frozenset = frozenset({
    'H3', 'H7', 'H9', 'H22', 'H31', 'H40',   # Hindi held-out
    'N3', 'N7', 'N9', 'N22', 'N31', 'N40',   # Nepali held-out
})


def is_held_out(example: dict) -> bool:
    """Return True iff the example uses at least one held-out template.

    The template_id field may encode a compound template (e.g. 'H1+H2');
    we split on '+' so that any constituent matching UNSEEN_TEMPLATE_IDS
    correctly triggers held-out status.
    """
    return any(
        t in UNSEEN_TEMPLATE_IDS
        for t in example.get('template_id', '').split('+')
    )


def extract_answer(raw_continuation: str) -> str:
    """
    Deterministic first-word extraction.

    Strips leading whitespace, then takes the first contiguous run of
    characters that is NOT whitespace or ASCII/Hindi delimiter punctuation.
    Returns '' if nothing can be extracted or if the extracted token is
    the UNK surface form '⁇'.

    This function does NOT inspect the ground-truth answer.
    """
    text = raw_continuation.lstrip()
    match = re.match(r'^([^\s.,;!?\u0964]+)', text)
    if not match:
        return ''
    extracted = match.group(1)
    if extracted == _UNK_SURFACE:
        return ''   # invalid — UNK at the boundary position
    return extracted


def generate_with_details(model, sp, prompt_text, max_new_tokens=20, device='cpu'):
    """
    Autoregressive generation, identical to the original generate_answer.
    Returns a dict with all per-example diagnostic fields.
    """
    model.eval()
    prompt_ids = sp.Encode(prompt_text, out_type=int)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos_id = sp.eos_id()

    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > model.config.max_seq_len:
                input_ids = input_ids[:, -model.config.max_seq_len:]
            next_token = input_ids.new_tensor([[model(input_ids)[0, -1, :].argmax().item()]])
            if next_token.item() == eos_id:
                break
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

    raw_continuation = sp.Decode(generated) if generated else ''
    eos_reached = len(generated) < max_new_tokens
    extracted = extract_answer(raw_continuation)

    return {
        'raw_continuation': raw_continuation,
        'eos_reached': eos_reached,
        'generated_tokens': len(generated),
        'extracted_answer': extracted,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning exact-match baseline")
    parser.add_argument('--model_ckpt', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--tokenizer',  type=str, required=True, help="Path to tokenizer model")
    parser.add_argument('--test_file',  type=str, required=True, help="Path to test JSONL")
    parser.add_argument('--lang',       type=str, required=True, choices=["hindi", "nepali"])
    parser.add_argument('--dry-run',    type=int, default=0,     help="Evaluate on N examples with random init")
    parser.add_argument('--max-new-tokens', type=int, default=20, help="Max tokens to generate per example")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)

    if args.dry_run > 0:
        print("\n" + "="*50)
        print("DRY RUN / RANDOM INITIALIZATION — NOT A PRETRAINED BASELINE")
        print("="*50 + "\n")

    model_cfg = LMConfig()
    model = DecoderOnlyLM(model_cfg)
    model.to(device)

    if args.dry_run == 0:
        if not os.path.exists(args.model_ckpt):
            print(f"[ERROR] Pretrained Phase 2 checkpoint not found at {args.model_ckpt}")
            print("Cannot evaluate Phase 3 baseline without the pretrained weights.")
            sys.exit(1)
        print(f"Loading pretrained Phase 2 checkpoint from {args.model_ckpt}")
        load_checkpoint(Path(args.model_ckpt), model, device=device, restore_rng=False)
        print("Pretrained checkpoint loaded.")
    else:
        print("Skipping checkpoint load for dry-run. Randomly initialized weights will be used.")

    with open(args.test_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if args.dry_run > 0:
        lines = lines[:args.dry_run]

    correct = 0
    total = 0
    eos_count = 0
    max_len_count = 0
    invalid_count = 0   # extracted == '' (UNK or empty)
    correct_no_eos = 0  # correct answer extracted but EOS not emitted

    # Stratified metrics
    by_task     = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_attr     = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_cat      = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_template = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_heldout  = defaultdict(lambda: {'correct': 0, 'total': 0})

    print(f"Evaluating {len(lines)} examples...")
    print(f"Extraction protocol: first non-whitespace/non-punctuation word; '⁇' → invalid")

    for i, line in enumerate(lines):
        d = json.loads(line)
        # Prompt format is identical for Hindi and Nepali — both use the same
        # marker as produced by the dataset generator.
        prompt_text = f"प्रश्न: {d['premise']} {d['question']} उत्तर: "

        ground_truth = str(d['answer']).strip()

        details = generate_with_details(
            model, sp, prompt_text,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        extracted  = details['extracted_answer']
        eos_reached = details['eos_reached']
        is_correct = (extracted == ground_truth)

        total += 1
        if is_correct:
            correct += 1
        if eos_reached:
            eos_count += 1
        else:
            max_len_count += 1
        if not extracted:
            invalid_count += 1
        if is_correct and not eos_reached:
            correct_no_eos += 1

        task       = d.get('task_type', 'unknown')
        attr       = d.get('attribute', 'unknown')
        cat        = d.get('question_category', 'unknown')
        templ      = d.get('template_id', 'unknown')
        # BUG FIX: `is_held_out_template` is never serialized by the generator.
        # Use template_id-based check against UNSEEN_TEMPLATE_IDS instead.
        heldout_str = "held_out" if is_held_out(d) else "familiar"

        by_task[task]['total']         += 1
        by_attr[attr]['total']         += 1
        by_cat[cat]['total']           += 1
        by_template[templ]['total']    += 1
        by_heldout[heldout_str]['total'] += 1

        if is_correct:
            by_task[task]['correct']         += 1
            by_attr[attr]['correct']         += 1
            by_cat[cat]['correct']           += 1
            by_template[templ]['correct']    += 1
            by_heldout[heldout_str]['correct'] += 1

        if (i + 1) % 50 == 0 or args.dry_run > 0:
            print(f"[{i+1}/{len(lines)}] acc={correct/total*100:.1f}% "
                  f"eos={eos_count/(i+1)*100:.0f}% "
                  f"invalid={invalid_count/(i+1)*100:.0f}%")
            if args.dry_run > 0:
                print(f"  Prompt   : {prompt_text}")
                print(f"  GT       : {ground_truth!r}")
                print(f"  RawCont  : {details['raw_continuation']!r}")
                print(f"  Extracted: {extracted!r}")
                print(f"  Match    : {is_correct}  EOS: {eos_reached}")

    results = {
        'overall_accuracy':           correct / total if total > 0 else 0,
        'correct':                    correct,
        'total':                      total,
        'eos_rate':                   eos_count / total if total > 0 else 0,
        'max_length_rate':            max_len_count / total if total > 0 else 0,
        'invalid_prediction_rate':    invalid_count / total if total > 0 else 0,
        'correct_no_eos':             correct_no_eos,
        'extraction_protocol': (
            "first_word: re.match(r'^([^\\s.,;!?\\u0964]+)', cont.lstrip()); "
            "'⁇' → invalid; exact_match = (extracted == gt); EOS not required"
        ),
        'by_task':       {k: v['correct'] / v['total'] for k, v in by_task.items()},
        'by_attr':       {k: v['correct'] / v['total'] for k, v in by_attr.items()},
        'by_category':   {k: v['correct'] / v['total'] for k, v in by_cat.items()},
        'by_template':   {k: v['correct'] / v['total'] for k, v in by_template.items()},
        'by_held_out':   {k: v['correct'] / v['total'] for k, v in by_heldout.items()},
    }

    out_dir = Path("results/phase3_finetune")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"baseline_{args.lang}.json"
    if args.dry_run == 0:
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nBaseline results saved to {out_file}")
        print(f"  exact_match_accuracy   : {results['overall_accuracy']*100:.2f}%")
        print(f"  eos_rate               : {results['eos_rate']*100:.2f}%")
        print(f"  max_length_rate        : {results['max_length_rate']*100:.2f}%")
        print(f"  invalid_prediction_rate: {results['invalid_prediction_rate']*100:.2f}%")
        print(f"  correct_no_eos         : {results['correct_no_eos']}")
    else:
        print("\n--- DRY RUN RESULTS ---")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print("(Results NOT saved to disk for dry run)")


if __name__ == '__main__':
    main()
