# Hindi & Nepali Monolingual Transformer Language Models

**Course**: Language Models and Agents (Monsoon 2026) — Individual Project  
**Repository**: [individual-project-a4Aaaryan](https://github.com/CL3-410/individual-project-a4Aaaryan.git)  
**Branch**: `phase-1`

---

## 📌 Project Overview

This repository implements two completely independent decoder-only Transformer language models (~25M parameters each) trained from scratch on two Indian languages:
- **Model H (Higher-Resource)**: Hindi
- **Model L (Lower-Resource)**: Nepali

### Core Constraints & Principles
- **Strict Independence**: Hindi and Nepali pipelines maintain separate corpora, tokenizers, vocabularies, and weights. No cross-lingual sharing or transfer learning.
- **Data Scale**: Target ~500M training tokens per language after tokenization.
- **Manual Data Requirement**: >= 20% of final training tokens must come from verified manual collection (Wikipedia dumps processed in-house + scraped news articles).
- **Tokenizers from Scratch**: Independent BPE/Unigram tokenizers trained from scratch using SentencePiece (no pretrained tokenizers).
- **Architecture**: From-scratch PyTorch decoder-only Transformer with custom multi-head attention and causal masking (no `nn.Transformer` or HuggingFace model classes).

---

## 🗂 Repository Structure

```text
.
├── README.md                          # Project documentation and Google Drive links
├── requirements.txt                   # Environment dependencies
├── .gitignore                         # Ignore rules for datasets, models, caches
│
├── common/                            # Shared utilities (no language-specific data)
│   ├── config.py                      # Root path resolution and YAML config loader
│   └── utils.py                       # Logging, seed management, I/O helpers
│
├── hindi/                             # Hindi (Model H - Higher Resource)
│   ├── configs/
│   │   └── phase1_config.yaml         # Data, paths, and tokenizer configurations
│   ├── data/
│   │   ├── raw/                       # Raw downloads and scrapes (excluded from git)
│   │   ├── interim/                   # Intermediate preprocessed files
│   │   ├── cleaned/                   # Cleaned, deduplicated corpus
│   │   └── splits/                    # Document-level train/val/test splits
│   ├── tokenizer/
│   │   ├── train/                     # Tokenizer training inputs
│   │   ├── models/                    # Trained SentencePiece model files
│   │   └── vocab/                     # Vocabulary files
│   ├── scripts/                       # Modular pipeline execution scripts
│   │   ├── 01_collect_sources.py
│   │   ├── 02_inspect_raw.py
│   │   ├── 03_preprocess.py
│   │   ├── 04_deduplicate.py
│   │   ├── 05_split_data.py
│   │   ├── 06_train_tokenizer.py
│   │   └── 07_generate_statistics.py
│   └── notebooks/                     # Exploratory analysis and visualizations
│
├── nepali/                            # Nepali (Model L - Lower Resource)
│   ├── configs/
│   │   └── phase1_config.yaml         # Data, paths, and tokenizer configurations
│   ├── data/
│   │   ├── raw/
│   │   ├── interim/
│   │   ├── cleaned/
│   │   └── splits/
│   ├── tokenizer/
│   │   ├── train/
│   │   ├── models/
│   │   └── vocab/
│   ├── scripts/
│   │   ├── 01_collect_sources.py
│   │   ├── 02_inspect_raw.py
│   │   ├── 03_preprocess.py
│   │   ├── 04_deduplicate.py
│   │   ├── 05_split_data.py
│   │   ├── 06_train_tokenizer.py
│   │   └── 07_generate_statistics.py
│   └── notebooks/
│
├── report/                            # Phase reports, figures, and summary tables
│   ├── figures/                       # Plots and visualizations
│   ├── tables/                        # Machine-readable summary statistics (JSON/CSV)
│   └── phase1_report.md               # Phase 1 submission report
│
└── scripts/
    └── verify_stage0.py               # Sanity check for environment and directory layout
```

---

## 🚀 Environment Setup

1. **Clone and checkout `phase-1` branch**:
   ```bash
   git clone https://github.com/CL3-410/individual-project-a4Aaaryan.git
   cd individual-project-a4Aaaryan
   git checkout phase-1
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Stage 0 environment**:
   ```bash
   python scripts/verify_stage0.py
   ```

---

## 🔗 External Artifacts & Google Drive Links

Per assignment instructions, large datasets and checkpoints are hosted on Google Drive:

| Artifact Description | Language | Google Drive Link |
|---|---|---|
| All Phase 1 Artifacts | Hindi & Nepali | [Google Drive Folder](https://drive.google.com/drive/folders/1IfzywcoVXSYJI_VuljQMAH7Qhwka3CTy?usp=drive_link) |
| Download trained model checkpoints (Phase 2) | Hindi & Nepali | [Google Drive Folder](https://drive.google.com/drive/folders/1J2CLalX8e9tWaH7i1XbgtrmQq5HKgNxr?usp=drive_link) |

---

## 🔄 Phase 1 Pipeline Execution Guide

Each stage is executed via modular language-specific scripts:

1. **Source Acquisition (Stage 1)**:
   ```bash
   python hindi/scripts/01_collect_sources.py
   python nepali/scripts/01_collect_sources.py
   ```
2. **Corpus Inspection (Stage 2)**:
   ```bash
   python hindi/scripts/02_inspect_raw.py
   python nepali/scripts/02_inspect_raw.py
   ```
3. **Preprocessing & Normalization (Stage 3)**:
   ```bash
   python hindi/scripts/03_preprocess.py
   python nepali/scripts/03_preprocess.py
   ```
4. **Deduplication (Stage 4)**:
   ```bash
   python hindi/scripts/04_deduplicate.py
   python nepali/scripts/04_deduplicate.py
   ```
5. **Splitting (Stage 6)**:
   ```bash
   python hindi/scripts/05_split_data.py
   python nepali/scripts/05_split_data.py
   ```
6. **Tokenizer Training & Evaluation (Stages 7 & 8)**:
   ```bash
   python hindi/scripts/06_train_tokenizer.py --model-type bpe --vocab-size 32000
   python nepali/scripts/06_train_tokenizer.py --model-type bpe --vocab-size 32000
   ```
7. **Statistics Generation (Stages 5 & 9)**:
   ```bash
   python hindi/scripts/07_generate_statistics.py
   python nepali/scripts/07_generate_statistics.py
   ```

---

## 🚀 Phase 2: Model Implementation & Evaluation

### Purpose & Architecture
Phase 2 implements and trains the two independent decoder-only language models from scratch:
- **Architecture**: 8 layers, 448 `d_model`, 7 heads (64 head dim), 2048 `d_ff`, 512 context length.
- **Parameters**: ~25.8M parameters per model.

### Execution Guide
The Phase 2 pipeline relies entirely on standalone Kaggle notebooks for reproducible cloud execution (a GPU is required).

**Execution Order**:
1. **Hindi Training**: Run `phase2_kaggle.ipynb` to train the Hindi model.
2. **Nepali Training**: Run `phase2_nepali_kaggle.ipynb` to train the Nepali model.
3. **Evaluation Prep**: Download the trained checkpoints (or use the provided [Google Drive Link](https://drive.google.com/drive/folders/1J2CLalX8e9tWaH7i1XbgtrmQq5HKgNxr?usp=drive_link)) and make them available to the evaluation notebook.
4. **Evaluation**: Run `phase2_evaluation.ipynb` to compute intrinsic metrics, run generation pipelines, and perform attention analysis.
5. **Artifact Retrieval**: Download the generated evaluation outputs from the notebook.

### Evaluation Artifacts & Reporting
- **Phase 2 Report**: The final analysis of the evaluation results is located at `report/phase2_report.md`.
- **Git Tracking**: To keep the repository lightweight, raw generation dumps and large evaluation string outputs are intentionally excluded from Git via `.gitignore`. Lightweight JSON metric summaries and the main Phase 2 report are committed.
