--------------------------------------------------
PROJECT
--------------------------------------------------

Phase 2 of a university NLP assignment.

Two independent decoder-only Transformer language models:

- Model H: Hindi
- Model L: Nepali

Each model should have approximately 25M trainable parameters.

The two models must remain completely independent:
- no transfer learning
- no parameter sharing
- independent tokenizers
- independent corpora
- independent model weights

Architecture must be implemented using basic PyTorch components. HuggingFace Transformer model classes are not allowed.

Required architecture:
- token embeddings
- positional representation
- multi-head causal self-attention
- causal masking
- Transformer blocks
- residual connections
- layer normalization
- feed-forward network
- LM head
- exact parameter count

Training:
- AdamW
- learning-rate scheduler
- checkpointing
- resume support
- checkpoint must contain model weights, optimizer state, scheduler state, training step, and config
- training logs
- loss curves
- mixed precision where appropriate
- gradient accumulation where useful

Evaluation:
- validation loss
- perplexity
- BPB
- greedy generation
- temperature 0.5
- temperature 1.0
- temperature 1.5
- BLEU
- chrF / chrF++
- ROUGE-L
- repetition diagnostics
- Distinct-1
- Distinct-2
- qualitative examples

Attention analysis:
- early and late layers
- multiple heads
- attention heatmaps
- attention entropy
- mean attention distance
- Hindi vs Nepali interpretation

Planned implementation order:

Stage 0: branch + repository/environment audit
Stage 1: configuration + embeddings/positional representation
Stage 2: multi-head causal self-attention + mask tests
Stage 3: Transformer block + tests
Stage 4: full LM + LM head + weight tying + exact parameter count
Stage 5: tokenize corpus into uint16 mmap .bin files
Stage 6: training loop + AdamW + scheduler + checkpoint/resume
Stage 7: tiny training + deliberate kill/resume smoke test
Stage 8: full Hindi and Nepali pretraining
Stage 9: intrinsic evaluation
Stage 10: generation + automatic metrics
Stage 11: attention analysis
Stage 12: Hindi/Nepali comparison
Stage 13: report + figures
Stage 14: final audit + phase-2 push

Do not skip the smoke-test stage before full pretraining.

--------------------------------------------------
PHASE 1 FINAL ARTIFACTS
--------------------------------------------------

Hindi final tokenizer:

hindi/tokenizer/models/hindi_bpe_10k_final.model

Vocabulary:

hindi/tokenizer/vocab/hindi_bpe_10k_final.vocab

Nepali assignment-compliant final tokenizer:

nepali/tokenizer/models/nepali_bpe_10k_assignment.model

Vocabulary:

nepali/tokenizer/vocab/nepali_bpe_10k_assignment.vocab

There is also:

nepali/tokenizer/models/nepali_bpe_10k_final.model

This is the experimental full-corpus Nepali tokenizer and MUST NOT be used for the final Nepali Phase 2 model.

Final Nepali Phase 2 Model L must use:

nepali_bpe_10k_assignment.model

--------------------------------------------------
FINAL CORPORA
--------------------------------------------------

Hindi:
- train: hindi/data/splits/train/
- val: hindi/data/splits/val/
- test: hindi/data/splits/test/

Nepali assignment corpus:
- train: nepali/data/final/train/
- val: nepali/data/final/val/
- test: nepali/data/final/test/

These directories contain multiple JSONL files. Future data preparation must process them correctly without assuming one monolithic file.

--------------------------------------------------
IMPORTANT NEPALI CORPUS FACT
--------------------------------------------------

The final assignment-compliant Nepali corpus contains:

- manual words: 31,592,362
- HPLT words: 118,000,000
- total words: 149,592,666
- instructor-approved heuristic tokens: 448,777,998
- manual heuristic share: 21.119%
- actual BPE tokens with assignment tokenizer: 234,800,139
- manual actual BPE tokens: 50,248,944
- manual actual-token share: 21.401%

The instructor explicitly permits the heuristic 1 word = 3 tokens for the token-percentage requirement.

--------------------------------------------------
COMPUTE
--------------------------------------------------

Development machine:

- RTX 3060 Laptop GPU
- 6 GB VRAM
- Ryzen 7 5800H
- Python 3.12.0
- SentencePiece installed
- PyTorch currently NOT installed
- approximately 43 GB free disk at Stage 0 audit

Cloud training may use Kaggle and/or Google Colab.

Training code must therefore avoid hardcoded machine-specific paths and must support portable checkpoints/resuming.

--------------------------------------------------
IMPORTANT STORAGE DESIGN
--------------------------------------------------

The eventual tokenized training data should use uint16 token IDs stored in memory-mapped .bin files because the vocabularies are around 10K.

Avoid unnecessary duplicate copies of the large corpora.

Large generated artifacts such as:
- tokenized .bin files
- checkpoints
- optimizer states
- large logs

should generally not be committed to Git.

Source code, configs, tests, report, and useful figures should be committed.

Do not modify .gitignore yet.

--------------------------------------------------
CURRENT STATUS
--------------------------------------------------

Stage 0 audit completed.
Architecture design phase completed. The model architecture is now formally locked at approximately 25.86M parameters (details in `ARCHITECTURE.md`).

Next actual development stage will be Stage 1. 

Do not implement that architecture yet.
