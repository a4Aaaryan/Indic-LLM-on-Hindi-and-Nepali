# Phase 1 Report: Data Collection and Tokenizer Construction

**Project**: Building and Analyzing Monolingual Transformer LMs in Two Indian Languages
**Course**: Language Models and Agents — Monsoon 2026
**Branch**: `phase-1`

---

## 1. Introduction

This report details Phase 1 of the project. The task was to construct monolingual corpora and train tokenizers for one higher-resource language (Hindi) and one lower-resource language (Nepali).

The Hindi and Nepali pipelines are completely independent. They do not share datasets, tokenizers, vocabularies, or model files. The only shared code consists of general Python configuration and utility functions.

The assignment target is approximately 500M training tokens per language, with at least 20% of the final tokens coming from manually collected material.

---

## 2. Language Selection

| Language | Tier | Script | Justification |
|---|---|---|---|
| **Hindi** | Higher-resource | Devanagari | Widely spoken with abundant digital resources, large web corpora (e.g., Indic HPLT v2), Wikipedia, and news sites. |
| **Nepali** | Lower-resource | Devanagari | Shares the Devanagari script but has significantly fewer digital resources, a smaller Wikipedia, and less web text, fitting the lower-resource requirement. |

---

## 3. Data Acquisition

For both languages, we used the Indic HPLT v2 dataset as the primary downloaded backbone. To meet the 20% manual data requirement, we extracted Wikipedia and Wikisource dumps ourselves and scraped several news sites.

### 3.1 Hindi Sources

| Source | Category | Collection Method | Raw Documents | Raw Words |
|---|---|---|---|---|
| Indic HPLT v2 (hi) | Downloaded | HuggingFace dataset download | 614,314 | 285,401,242 |
| Hindi Wikipedia | Manual | Wikimedia XML dump extracted via custom script | 168,299 | 51,803,550 |
| Hindi Wikisource | Manual | Wikimedia XML dump extracted via custom script | 167,471 | 52,805,749 |
| Dainik Jagran | Manual | Web scraping (article body text) | 30 | 12,308 |
| Amar Ujala | Manual | Web scraping (article body text) | 85 | 34,901 |
| Live Hindustan | Manual | Web scraping (article body text) | 119 | 73,708 |

### 3.2 Nepali Sources

| Source | Category | Collection Method | Raw Documents | Raw Words |
|---|---|---|---|---|
| Indic HPLT v2 (ne) | Downloaded | HuggingFace dataset download | 966,131 | 285,524,100 |
| Nepali Wikipedia | Manual | Wikimedia XML dump extracted via custom script | 30,285 | 7,061,109 |
| Nepali Wikisource | Manual | Wikimedia XML dump extracted via custom script | 80 | 90,301 |
| Onlinekhabar | Manual | Web scraping (article body text) | 28,987 | 10,489,745 |
| Ekantipur | Manual | Web scraping (article body text) | 29,088 | 12,373,873 |
| Ratopati | Manual | Web scraping (article body text) | 7,245 | 2,177,890 |
| Setopati | Manual | Web scraping (article body text) | 2,581 | 284,433 |
| Kantipur | Manual | Web scraping (article body text) | 90 | 39,202 |
| Nepali Archives | Manual | Web scraping (literary works) | 4 | 6,002 |
| E-Pustakalaya | Manual | Catalog metadata | 4 | 2,134 |

---

## 4. Methodology

### 4.1 Preprocessing

Both pipelines process data sequentially using streaming JSONL readers to manage memory. The preprocessing steps applied to every document are:

1. **HTML Removal**: Regex-based removal of HTML tags.
2. **Unicode Normalization**: NFKC normalization to standardize character representations.
3. **Whitespace Normalization**: Multiple spaces and tabs were collapsed into single spaces, but paragraph boundaries were preserved.
4. **Filtering**: Documents shorter than 50 characters were removed.
5. **Content Preservation**: We retained digits and legitimate multilingual content (e.g., Latin-script names and technical terms). For Nepali, an aggressive Devanagari-ratio filter was considered but ultimately rejected because it inadvertently removed valid news articles containing English loanwords or numbers.

During processing, provenance metadata (`doc_id`, `source_name`, `language`) was preserved for every document.

### 4.2 Deduplication

Deduplication was performed globally across all sources for each language.

1. We stripped leading and trailing whitespace from the normalized document text.
2. We generated a SHA-256 hash of the exact text.
3. We maintained a set of seen hashes. If a document's hash was already in the set, it was discarded.
4. We processed manual sources before downloaded sources. This priority ordering ensured that if a document appeared in both a scraped news site and HPLT, the manual source's version (with correct provenance) was retained.

### 4.3 Splitting

We used a deterministic document-level 90/5/5 split for train, validation, and test sets. We used a fixed random seed (42). The random generator was seeded per source (`f"{seed}_{source_name}"`) so that the split remains stable regardless of the order in which sources are processed.

### 4.4 Tokenizer Training

We trained SentencePiece BPE tokenizers using a deterministic sample drawn from the training split. We generated token statistics and counts by applying the trained tokenizer to the final splits.

---

## 5. Hindi Pipeline and Statistics

### 5.1 Deduplication and Splits

The Hindi corpus contained 950,318 raw documents. After preprocessing and global deduplication, 948,012 documents remained (a reduction of 2,306 documents). A dedicated Hindi deduplication report file was not generated, so these numbers are derived from the raw inspection and final statistics.

The deterministic 90/5/5 split produced:
* **Train**: 853,505 documents (351,201,146 words)
* **Validation**: 47,496 documents (19,464,204 words)
* **Test**: 47,011 documents (19,328,415 words)

### 5.2 Tokenizer Training

We trained a SentencePiece BPE tokenizer for Hindi. While 8K and 10K vocabulary candidates were evaluated during the pipeline, the specific candidate comparison metrics were not retained in the repository. We selected the 10,000 vocabulary size based on those initial evaluations.

The final tokenizer (`hindi_bpe_10k_final`) was trained on a deterministically sampled subset of the training data.

### 5.3 Final Hindi Statistics

The final Hindi corpus easily exceeds the minimum requirements.

* **Algorithm**: SentencePiece BPE
* **Vocabulary Size**: 10,000
* **Total Actual BPE Tokens**: 587,927,125
* **Fertility**: 1.508 tokens per word
* **UNK Percentage**: 2.576%

**Manual vs. Downloaded Breakdown (by Actual Tokens):**
* **Manual**: 176,068,967 tokens (**29.95%**)
* **Downloaded (HPLT)**: 411,858,158 tokens (70.05%)

The manual token proportion of 29.95% satisfies the ≥20% requirement.

---

## 6. Nepali Pipeline, Tokenizer Models, and Final Selection

The Nepali pipeline required downsampling to meet the manual-data composition requirements. We produced two tokenizers, which serve different purposes.

### 6.1 Phase A: Tokenizer Candidate Comparison

We trained two SentencePiece BPE candidates on a reservoir sample of 2,330,280 sentences from the full training split.

| Candidate | Vocab Size | Val Tokens | Fertility | Chars/Token | UNK % |
|---|---|---|---|---|---|
| `nepali_bpe_8000` | 8,000 | 26,096,966 | 1.636 | 3.949 | 2.104% |
| `nepali_bpe_10000` | 10,000 | 25,067,886 | 1.571 | 4.111 | 2.191% |

We selected the 10K candidate. It achieved lower fertility (more efficient encoding) with only a minimal increase in the UNK rate (+0.09%).

### 6.2 Phase B: The Full-Corpus Model (`nepali_bpe_10k_final`)

We trained the selected 10K BPE tokenizer on a larger deterministic sample of the full corpus (up to 5 million sentences per source). We applied this tokenizer (`nepali_bpe_10k_final`) to the complete Nepali corpus (1,061,336 documents, 317,069,245 words).

* **Actual BPE tokens**: 496,534,521
* **Fertility**: 1.566
* **UNK Percentage**: 2.162%

**The Composition Issue**: Under this tokenizer, the manual data accounted for only **10.14%** of the actual BPE tokens, because the downloaded HPLT corpus was much larger.

The instructor Q&A permitted using a `1 word = 3 tokens` heuristic for evaluating corpus composition. Under this heuristic, the full corpus contained approximately 21.1% manual data. However, to strictly satisfy the 20% manual requirement under *both* the heuristic and the actual tokenizer output, we created a second dataset.

We retained `nepali_bpe_10k_final` and the full corpus as a high-data experimental baseline, but it is not our official assignment submission model.

### 6.3 Phase C: The Assignment-Compliant Model (`nepali_bpe_10k_assignment`)

To increase the manual token proportion, we deterministically downsampled the HPLT corpus (from ~285M words to ~118M words) while keeping all manual data unmodified. We wrote these downsampled files to `nepali/data/final/`.

We then trained a new tokenizer (`nepali_bpe_10k_assignment`) on this assignment-compliant corpus.

**Final Assignment-Compliant Statistics:**
* **Total Words**: 149,592,666
* **Algorithm**: SentencePiece BPE
* **Vocabulary Size**: 10,000
* **Fertility**: 1.570
* **UNK Percentage**: 2.181%

**Manual Proportion (Heuristic vs Actual):**
* **Instructor Heuristic (1 word = 3 tokens)**: 448,777,998 total tokens. 94,777,086 manual tokens (**21.12% manual**).
* **Actual BPE Tokenizer Output**: 234,800,139 total actual tokens. 50,248,944 manual actual tokens (**21.40% manual**).

*Note: The assignment target is approximately 500M tokens. Using the permitted heuristic, this corpus contains ~448.78M tokens, meeting the target. The actual BPE token count of ~234.8M is lower because Nepali SentencePiece fertility (1.570) is much lower than the heuristic assumption of 3.*

**This downsampled corpus and the `nepali_bpe_10k_assignment` tokenizer constitute our final submission for Nepali.**

### 6.4 Nepali Deduplication Statistics

Prior to splitting and downsampling, deduplication on the full Nepali corpus showed:
* **Input documents**: 1,064,492
* **Retained documents**: 1,061,336
* **Duplicates removed**: 3,156 (0.30%)

Most exact duplicates were found within Ratopati (3,019 internal duplicates) and Ekantipur/Kantipur (81 cross-source duplicates).

---

## 7. Final Comparison

The following table compares the Hindi corpus against the final assignment-compliant Nepali corpus.

| Metric | Hindi | Nepali (Assignment) |
|---|---|---|
| **Total documents** | 948,012 | 494,642 |
| **Total words** | 389,993,765 | 149,592,666 |
| **Manual words** | 104,664,750 | 31,592,362 |
| **Downloaded words** | 285,329,015 | 118,000,304 |
| **Instructor heuristic token estimate** | — | ~448.78M |
| **Actual BPE tokens** | 587,927,125 | 234,800,139 |
| **Manual token percentage (actual)** | **29.95%** | **21.40%** |
| **Tokenizer algorithm** | SentencePiece BPE | SentencePiece BPE |
| **Vocabulary size** | 10,000 | 10,000 |
| **Fertility** | 1.508 | 1.570 |
| **UNK percentage** | 2.576% | 2.181% |

---

## 8. Reproducibility and Repository Structure

The code is organized by language in the `phase-1` branch:

* `hindi/` and `nepali/` directories contain separate `scripts/`, `configs/`, and `tokenizer/` folders.
* Processing is controlled by `phase1_config.yaml` in each language directory.
* Statistics and reports are saved to `report/tables/`.
* Random operations use fixed seeds (Seed: 42) for reproducibility.

The large dataset artifacts and trained tokenizer models are hosted on Google Drive and are not committed to Git.

**Google Drive Links**:
* [Google Drive Directory for All Phase 1 Artifacts](https://drive.google.com/drive/folders/1IfzywcoVXSYJI_VuljQMAH7Qhwka3CTy?usp=drive_link)
