# Phase 3 Fine-tuning Pipeline Plan

## Goal
Implement a robust, resume-capable fine-tuning pipeline for Phase 3 reasoning data. The pipeline will strictly separate Hindi and Nepali, freeze the tokenizer/vocabulary, maintain the Phase 2 architecture, and introduce answer-only loss masking.

## User Review Required
> [!IMPORTANT]
> The pretrained Phase 2 checkpoints (`step_00020000.pt`) were originally saved on Kaggle and are not present in the local `C:\Users\arped\OneDrive\Desktop\LMA` directory. For the purpose of testing the local pipeline, I will initialize the evaluator and fine-tuner to gracefully accept a randomly-initialized mock checkpoint if the real checkpoint is missing, so we can verify the data-loading and tensor shapes during the "dry run". When you run this on a GPU instance, you will just point the config to the real downloaded checkpoints.

## Clarification on Validation Report Statistic
> [!NOTE]
> Regarding your correction: You are absolutely right! The statistic `10,452 unique reasoning instances` printed at the end of the Hindi Test block was accumulating unique instances across **all three splits** (Train+Val+Test, total 12,000), not just the 1,200 test examples. The script maintained a global `seen_reasoning` dictionary to check for leakage, and its length was printed at the very end of the run (which happens to be the test block). Out of 12,000 generated Hindi examples, 10,452 have a structurally unique underlying reasoning graph.

## Proposed Changes

### Configuration
#### [NEW] `phase3/finetune_config.yaml`
A central configuration file defining all hyperparameters.
- **Paths**: `pretrained_checkpoint`, `tokenizer_path`, `train_data`, etc.
- **Hyperparameters**: `batch_size: 16`, `grad_accum_steps: 1`, `learning_rate: 5e-5` (lower than Phase 2's `3e-4`), `epochs: 3`, `context_length: 512`.
- **Outputs**: `results/phase3_finetune/hindi/` and `results/phase3_finetune/nepali/`.

### Data Loading & Loss Masking
#### [NEW] `phase3/src/dataset.py` (or integrated into a script)
- We will parse JSONL directly since fine-tuning sets are small (9.6k examples = ~300k tokens).
- **Loss Masking**: PyTorch `CrossEntropyLoss` uses `ignore_index=-100`. 
  - `text = "प्रश्न: ... उत्तर: <answer>"`
  - `prompt = "प्रश्न: ... उत्तर: "`
  - We tokenize `text`, determine the length of `prompt` tokens $L$.
  - We set `labels = [-100] * L + text_tokens[L:] + [EOS]`.
  - This perfectly retains the causal LM objective but only accumulates gradients for the answer tokens and the final EOS token.

### Evaluation
#### [NEW] `phase3/scripts/evaluate_reasoning.py`
A script that loads a checkpoint, iterates over a JSONL test set, and calculates exact-match accuracy.
- Takes arguments for `--model_ckpt`, `--tokenizer`, `--test_file`.
- Uses a greedy autoregressive generation loop, stopping at the exact tokenizer EOS ID.
- Strips the prompt from the generated text and compares the remaining string exactly with the ground-truth answer.
- Aggregates metrics globally and sliced by task, attribute, question category, and template family (familiar vs held-out).
- Produces `baseline_hindi.json` and a Markdown summary.
- **Dry-run**: Supports a `--dry-run 20` flag to process only 20 examples, allowing us to verify the entire stack locally without a GPU.

### Fine-Tuning Plumbing
#### [NEW] `phase3/scripts/finetune.py`
The main training loop (conceptually similar to `phase2/src/trainer.py` but adapted for Phase 3).
- Initializes model from Phase 2 architecture `DecoderOnlyLM`.
- Loads pretrained checkpoint state exactly.
- Uses `AdamW` optimizer and cosine scheduler.
- **Resuming**: Supports `--resume_from` to seamlessly reload model/optimizer/scheduler state and global step.

## Verification Plan

### Automated Tests
1. **Loss Masking Check**: Manually print the tokenized input and label array for one example to visually confirm `-100` aligns perfectly with the prompt.
2. **Evaluator Dry-Run**: Run `python phase3/scripts/evaluate_reasoning.py --lang hindi --dry-run 10`. Verify the script loads data, infers shapes, extracts the answer, computes exact-match, and outputs the JSON/MD summaries.
3. **Finetune Dry-Run**: Run `python phase3/scripts/finetune.py --config phase3/finetune_config.yaml --dry-run 2` to verify model loading, backward pass, and loss masking without crashing.

### Manual Verification
Review the exact-match output logs to confirm generation terminates correctly on EOS and text extraction is clean.
