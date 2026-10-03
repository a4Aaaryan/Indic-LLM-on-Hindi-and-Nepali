# Phase 1 Instructions — Hindi + Nepali Monolingual Transformer LM Project

## 0. Purpose and scope

This document is the complete implementation brief for Phase 1 of the Language Models and Agents — Monsoon 2026 Individual Project.

Repository:
`https://github.com/CL3-410/individual-project-a4Aaaryan.git`

Current scope:
- Complete **Phase 1 only**.
- Do not implement Phase 2 or Phase 3 yet, except where Phase 1 code must be designed so that later phases can reuse it.
- The Phase 1 branch must be `phase-1`.
- The Phase 1 deadline is **19 August 2026, 11:59 P.M.**
- The project ultimately has three phases:
  - Phase 1: Data Collection and Tokenizer Construction — 25 marks
  - Phase 2: Model Implementation, Pretraining, and Evaluation — 40 marks
  - Phase 3: Reasoning Finetuning, Attention Analysis, and Final Report — 35 marks

The professor grades the state of the corresponding branch at its deadline. Commit and push meaningful progress regularly. Do not make a single last-minute dump of work.

The complete assignment requires two independent monolingual decoder-only Transformer LMs:
- Model H = higher-resource language
- Model L = lower-resource language

For this project, the language decision is fixed:
- **Model H: Hindi**
- **Model L: Nepali**

Do not change these languages unless the student explicitly instructs you to do so.

---

# 1. Non-negotiable project constraints

These constraints come directly from the assignment and professor clarifications.

## 1.1 Two independent languages

Hindi and Nepali must remain independent.

Do NOT:
- mix their corpora;
- concatenate the languages into one corpus;
- share tokenizer vocabulary;
- share tokenizer model files;
- share model weights;
- initialize one model from the other;
- create a multilingual model.

We may share generic Python utility code between the two pipelines. We must not share language-specific data or artifacts.

## 1.2 Data target

Target approximately **500M training tokens after tokenization per language**.

For Nepali, if 500M genuinely cannot be obtained from public sources, collect the maximum feasible amount, report the exact token count, and justify the shortfall. Do not artificially inflate the count.

Do not confuse:
- raw characters,
- words,
- documents,
- whitespace-separated tokens,
- tokenizer tokens.

The assignment's target is approximately 500M **training tokens after tokenization**.

## 1.3 Manual collection requirement

At least **20% of the final training tokens must come from manually collected material for BOTH languages**.

This is not only for the low-resource language.

Professor clarification:
- Web scraping is allowed.
- Wikipedia can be manually collected.
- Wikipedia dumps count as manual collection if we retrieve them and preprocess them ourselves with our own scripts.
- Wikipedia API retrieval also falls under the same manually collected/preprocessed approach.
- Do not count an already-preprocessed third-party Wikipedia corpus as our manual collection.
- Do not count HPLT's own web crawl as our manual collection.
- The professor stated that failing to collect 20% of the dataset ourselves costs 20% of the Phase 1 points.

Target approximately **25–30% manual data**, rather than exactly 20%, to provide a safety margin after cleaning, filtering, deduplication, and tokenization.

Every document must retain provenance sufficient to demonstrate whether it is manual or downloaded.

At minimum track:
- document ID;
- language;
- source type (`manual` or `downloaded`);
- source name;
- source URL where applicable;
- collection date;
- raw file location;
- cleaned file location;
- license/usage information where available.

## 1.4 Raw + cleaned data

Professor clarification: upload BOTH raw and cleaned datasets to Google Drive.

Do not commit large datasets to Git.

The GitHub repository should contain:
- collection scripts;
- preprocessing scripts;
- tokenizer training code;
- statistics;
- configuration/metadata;
- reports;
- README;
- Google Drive links to large artifacts.

Google Drive must be accessible to TAs without requesting access.

## 1.5 Tokenizer

Train a separate tokenizer from scratch for each language.

Allowed:
- BPE
- SentencePiece
- Unigram

Professor clarification: tokenizer libraries may be used.

Do not use a pretrained tokenizer.

We should use a tokenizer library rather than implementing BPE/Unigram from scratch.

The tokenizer choice and vocabulary size should be based on measured behavior rather than arbitrarily hard-coded.

Required tokenizer statistics:
- vocabulary size;
- token-frequency statistics;
- average characters per token;
- tokenization examples;
- unknown-token statistics.

Also collect useful additional statistics such as:
- fertility / tokens per input unit;
- tokens per character;
- token frequency distribution;
- sentence/document token-length distributions;
- proportion of rare/singleton tokens where practical.

## 1.6 Language filtering

Professor clarification:
- digits are allowed;
- non-language words should be removed.

Do NOT blindly remove every Latin character because some legitimate content may contain names, abbreviations, URLs, technical terms, etc. Implement a documented and defensible filtering policy.

The policy should be consistent across both languages while allowing language-specific script rules where necessary.

## 1.7 Code format

`.ipynb` files are allowed.

However, prefer modular `.py` scripts for reusable pipelines and use notebooks for exploratory analysis/visualization where useful.

The final repository should be reproducible without relying on manually executed hidden notebook state.

---

# 2. Source strategy

The source strategy below is the current working plan.

Do not add many extra corpora merely because they exist. More data sources increase overlap, deduplication, licensing, provenance, and preprocessing complexity.

Use the primary sources first. Use backups only if the measured clean/tokenized corpus is insufficient or a quality problem is discovered.

---

## 2.1 Hindi — downloaded corpus

### Primary source: Indic HPLT v2

Dataset:
`https://huggingface.co/datasets/ashtok897/indic-hplt-v2`

Use the Hindi subset as the primary downloaded backbone.

The dataset provides large-scale Hindi web text and metadata. The exact final number of documents/tokens retained must be measured after our own preprocessing and tokenization.

Do not assume the dataset's reported token estimate equals our final tokenizer token count.

### Backup source: IndicCorp v2

Dataset:
`https://huggingface.co/datasets/ai4bharat/IndicCorpV2`

Use only if needed.

Do not automatically merge IndicCorp v2 into HPLT v2. First determine whether HPLT alone gives enough clean data and whether additional domain diversity is actually useful.

---

## 2.2 Hindi — manually collected corpus

### Primary manual source: Hindi Wikipedia

Official site:
`https://hi.wikipedia.org/`

Wikipedia dump:
`https://dumps.wikimedia.org/hiwiki/latest/`

Collect the raw dump ourselves and extract/process it with our own scripts.

Preserve:
- raw dump;
- extraction script;
- extracted raw text;
- cleaned text;
- document/page metadata where practical.

Do not download a pre-cleaned third-party Hindi Wikipedia corpus and classify it as manual.

### Hindi web sources

Use these as manually scraped sources:

1. Dainik Jagran
   `https://www.jagran.com/`

2. Amar Ujala
   `https://www.amarujala.com/`

3. Hindustan
   `https://www.livehindustan.com/`

Start with Wikipedia + these web sources.

If enough manual tokens are obtained with a comfortable 25–30% buffer, stop collecting additional sources.

When scraping, retain:
- URL;
- title;
- publication date where available;
- source;
- raw HTML;
- extracted article text.

Extract article content only. Remove:
- navigation;
- advertisements;
- cookie banners;
- menus;
- related-article widgets;
- comments;
- footer text;
- social widgets;
- boilerplate.

Respect the website's access restrictions and do not attempt to bypass technical protections.

---

## 2.3 Nepali — downloaded corpus

### Primary source: Indic HPLT v2

Dataset:
`https://huggingface.co/datasets/ashtok897/indic-hplt-v2`

Use the Nepali subset (`ne`) as the primary downloaded backbone.

The currently available dataset provides substantially more than the target scale, so do not automatically retain everything.

After our own cleaning and tokenizer training, select an appropriate amount to meet the target while preserving a controlled manual fraction.

### Backup source: Nepali Corpus v1

Dataset:
`https://huggingface.co/datasets/paudelnirajan/nepali-corpus-v1`

Use only if needed.

### Backup source: CC100 Nepali

Dataset:
`https://huggingface.co/datasets/himalaya-ai/cc100-nepali`

Use only if needed.

### Do NOT use as the primary source

Do not use the `neBrahma-Nepali-Pretrain-Corpus` as the primary downloaded corpus unless explicitly reconsidered later. Its source composition overlaps with existing resources, making provenance and overlap control less clean for this assignment.

---

## 2.4 Nepali — manually collected corpus

### Primary manual source: Nepali Wikipedia

Official site:
`https://ne.wikipedia.org/`

Wikipedia dump:
`https://dumps.wikimedia.org/newiki/latest/`

Collect and preprocess it ourselves.

### Primary Nepali web source: Onlinekhabar

`https://www.onlinekhabar.com/`

Use article pages and extract actual Nepali article content.

### Secondary Nepali web source: Ratopati

`https://www.ratopati.com/`

Use if additional manual tokens are required.

### Backup Nepali web source: Kantipur

`https://ekantipur.com/`

Use only if necessary.

As with Hindi, retain:
- URL;
- title;
- date where available;
- source;
- raw HTML;
- extracted article text.

Do not scrape navigation, ads, comments, or boilerplate.

---

# 3. Target corpus composition

Do NOT decide the final downloaded/manual token counts before measuring the actual data.

The pipeline should be token-budget driven.

For each language:

1. Collect manual sources.
2. Clean them.
3. Tokenize a representative/final version.
4. Count actual tokenizer tokens.
5. Collect downloaded sources.
6. Clean them.
7. Deduplicate across all sources.
8. Tokenize/count.
9. Construct the final corpus while maintaining a manual fraction safely above 20%.

A reasonable target is approximately:

- total training tokens: ~500M;
- manual: ~125–150M;
- downloaded: whatever additional amount is required.

Do not force exactly 500M if that would require arbitrary truncation or produce a poor corpus. Approximately 500M is the assignment target.

For Nepali in particular, if we cannot reach 500M after genuinely exhausting reasonable public/manual sources, document the exact feasible amount and the reason.

---

# 4. Repository structure

Start with a clean, language-separated structure. Adapt existing repository contents rather than deleting useful work.

Recommended structure:

```text
repo/
├── README.md
├── requirements.txt
├── .gitignore
│
├── hindi/
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
│   ├── configs/
│   └── notebooks/
│
├── nepali/
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
│   ├── configs/
│   └── notebooks/
│
├── common/
│   └── utilities/
│
└── report/
    ├── figures/
    ├── tables/
    └── phase1_report.md
```

The exact names can be adapted to the existing repository.

Do not create duplicate implementations of generic functionality unnecessarily.

Language-specific artifacts must remain clearly separated.

---

# 5. Phase 1 implementation stages

Complete the following stages in order.

---

## Stage 0 — Repository audit and environment setup

Before writing substantial code:

1. Inspect the current repository.
2. Preserve existing useful files.
3. Identify the current branch and create/use `phase-1`.
4. Check whether any starter code exists.
5. Create/update `.gitignore`.
6. Make sure raw datasets, cleaned datasets, tokenizer models, caches, checkpoints, and other large artifacts are not accidentally committed.
7. Set up Python environment/dependencies.
8. Record exact package versions where practical.
9. Create the language-separated directory structure.
10. Update README with the intended Phase 1 workflow.

Do not overwrite or delete existing project work without checking it first.

---

## Stage 1 — Source acquisition and provenance

Build reproducible collection/download scripts.

### Hindi

Downloaded:
- Indic HPLT v2 Hindi subset.

Manual:
- Hindi Wikipedia dump.
- Dainik Jagran.
- Amar Ujala.
- Hindustan.

### Nepali

Downloaded:
- Indic HPLT v2 Nepali subset.

Manual:
- Nepali Wikipedia dump.
- Onlinekhabar.
- Ratopati.
- Kantipur only if needed.

For every source:
- document the source;
- preserve raw data;
- record URLs/metadata;
- record collection method;
- record whether it is manual or downloaded.

Do not call a source "manual" simply because it is a web corpus. It must have been independently retrieved/collected and processed by our pipeline.

At the end of this stage, produce a source inventory table for each language.

---

## Stage 2 — Raw corpus inspection

Before aggressive cleaning, inspect samples from every source.

Measure:
- document counts;
- raw character counts;
- rough word counts;
- script distribution;
- obvious non-language contamination;
- duplicate/near-duplicate patterns;
- HTML/boilerplate contamination;
- empty/very-short documents;
- encoding issues.

Create source-specific cleaning rules only where necessary.

Do not silently throw data away.

Every removal rule must be documented.

---

## Stage 3 — Preprocessing and normalization

Implement a deterministic preprocessing pipeline.

At minimum:

1. Decode text correctly.
2. Unicode normalization.
3. Remove HTML/markup where applicable.
4. Extract article/page body from scraped pages.
5. Remove boilerplate.
6. Remove clearly invalid documents.
7. Remove non-language words/material according to a documented policy.
8. Keep digits.
9. Normalize whitespace.
10. Remove obvious corruption.
11. Deduplicate.
12. Run final quality checks.

Avoid over-cleaning.

Do not remove legitimate punctuation or numerals merely because they are not alphabetic.

Do not apply a single crude "keep only Unicode range X" rule without checking its effect on real text.

---

## Stage 4 — Deduplication

Deduplication must happen before final train/validation/test splitting.

At minimum implement exact document/string deduplication.

If practical, implement a lightweight near-duplicate method as well.

Important:
- deduplicate across sources, not only within one source;
- preserve provenance for the surviving document;
- report how many documents/tokens were removed.

This is particularly important because different web corpora may contain overlapping articles.

Do not claim perfect semantic deduplication unless it is actually implemented.

---

## Stage 5 — Corpus statistics

Generate statistics separately for Hindi and Nepali.

Required:
- number of raw documents;
- number of cleaned documents;
- character count;
- approximate word count where meaningful;
- final corpus size;
- source distribution;
- manual vs downloaded distribution;
- manual token percentage;
- cleaning/removal statistics;
- duplicate-removal statistics.

After tokenizer selection, also report:
- tokenizer token count;
- training token count;
- validation token count;
- test token count.

Produce machine-readable statistics such as JSON/CSV so that the report can be regenerated.

---

## Stage 6 — Train/validation/test split

Split independently for Hindi and Nepali.

Preferred approach:
- document-level split;
- deduplicate before splitting;
- avoid putting nearly identical documents in different splits.

Use a fixed random seed for any randomized component.

Record:
- split proportions;
- exact document counts;
- exact token counts after tokenizer selection;
- seed;
- splitting method.

Do not leak documents across splits.

Do not train the tokenizer on the test set in a way that compromises the intended held-out evaluation.

Keep the split definitions reproducible.

---

## Stage 7 — Tokenizer experiments

Train separate tokenizer candidates for Hindi and Nepali.

Initial candidate family:
- SentencePiece BPE;
- optionally SentencePiece Unigram.

Initial vocabulary-size candidates can include:
- 16k;
- 32k;
- 48k.

These are candidate experiments, NOT fixed final decisions.

The final choice must be based on measurements.

Use training data for tokenizer training and evaluate tokenizer behavior on held-out text.

For every candidate collect:
- vocabulary size;
- unknown-token rate;
- fertility;
- average characters per token;
- token frequency statistics;
- token-length distribution;
- example segmentations;
- tokens per sentence/document.

Do not use pretrained tokenizers.

Do not share the Hindi tokenizer with Nepali.

---

## Stage 8 — Select final tokenizers

Choose Hindi and Nepali tokenizers independently.

The selection should consider:
- low/acceptable unknown-token rate;
- reasonable fertility;
- reasonable characters/token;
- vocabulary efficiency;
- frequency distribution;
- compatibility with the future ~25M parameter Transformer;
- practical sequence lengths for Phase 2.

Do not simply choose the largest vocabulary.

Do not optimize one metric in isolation.

Record the rationale in the Phase 1 report.

The final tokenizer artifacts must be saved and reproducible.

---

## Stage 9 — Final corpus tokenization and target sizing

Once the final tokenizer for each language is selected:

1. Tokenize train/validation/test.
2. Count exact tokens.
3. Verify manual fraction on the final training-token count.
4. Verify target corpus size.
5. If the manual fraction is below 20%, add more manual data or reduce the downloaded portion.
6. Prefer 25–30% manual data for safety.
7. If the target cannot be reached, document why.

Important:
The manual fraction is based on the **final training tokens**, not merely the number of raw documents.

Do not accidentally let a large downloaded source dominate the final corpus after tokenization.

---

## Stage 10 — Phase 1 report

The report must contain, at minimum:

### A. Language selection
- Hindi as Model H.
- Nepali as Model L.
- brief justification of the resource-tier distinction.

### B. Corpus sources
For each language:
- source;
- source type;
- collection/download method;
- approximate contribution;
- provenance;
- license/usage information where available.

### C. Manual vs downloaded data
Explicitly show:
- manual tokens;
- downloaded tokens;
- total training tokens;
- manual percentage.

Show this separately for Hindi and Nepali.

### D. Preprocessing
Explain:
- Unicode normalization;
- language filtering;
- digit policy;
- boilerplate removal;
- invalid-document filtering;
- deduplication;
- splitting.

### E. Dataset statistics
Include tables/plots for:
- document counts;
- token counts;
- source distribution;
- split sizes;
- manual/downloaded fractions;
- cleaning statistics.

### F. Tokenizer
For each language:
- algorithm;
- vocabulary size;
- training procedure;
- token frequency statistics;
- fertility;
- average characters/token;
- unknown-token statistics;
- representative tokenization examples.

### G. Choice justification
Explain why the selected tokenizer configuration was chosen over alternatives.

Do not claim that one configuration is objectively "best" unless the evidence supports it.

---

# 6. Phase 1 quality checks

Before submission, run a dedicated validation script/checklist.

Verify:

### Data
- [ ] Hindi corpus exists.
- [ ] Nepali corpus exists.
- [ ] Raw data preserved on Google Drive.
- [ ] Cleaned data preserved on Google Drive.
- [ ] Source provenance is recorded.
- [ ] Manual sources are genuinely collected by our scripts/process.
- [ ] Manual fraction >=20% for Hindi.
- [ ] Manual fraction >=20% for Nepali.
- [ ] Preferably manual fraction is 25–30%.
- [ ] No accidental language mixing.
- [ ] Digits are retained.
- [ ] Obvious non-language contamination is handled.
- [ ] Duplicates are removed.
- [ ] Train/validation/test are separated.
- [ ] No obvious train/test leakage.

### Tokenizer
- [ ] Hindi tokenizer is trained from Hindi data.
- [ ] Nepali tokenizer is trained from Nepali data.
- [ ] They are separate artifacts.
- [ ] No pretrained tokenizer is used.
- [ ] Candidate experiments and final selection are documented.
- [ ] Vocabulary size is reported.
- [ ] Unknown-token statistics are reported.
- [ ] Fertility is reported.
- [ ] Average characters/token is reported.
- [ ] Token examples are reported.

### Reproducibility
- [ ] Collection scripts work.
- [ ] Preprocessing scripts work.
- [ ] Split generation works.
- [ ] Tokenizer training works.
- [ ] Statistics can be regenerated.
- [ ] Random seeds are recorded.
- [ ] Dependencies are documented.
- [ ] README contains reproduction instructions.
- [ ] Google Drive links are present and accessible.

### Git
- [ ] Work is on `phase-1`.
- [ ] Meaningful commits exist.
- [ ] No huge datasets are committed.
- [ ] No API keys/secrets are committed.
- [ ] Final report is inside the repository.
- [ ] Figures/tables intended for grading are inside the branch.
- [ ] Final push happens before the deadline.

---

# 7. Important future-phase design requirements

Do not implement Phase 2 yet, but Phase 1 should not make Phase 2 unnecessarily difficult.

The final project will require:

- two separate ~25M parameter decoder-only Transformers;
- separate vocabularies;
- separate model weights;
- approximately 500M training tokens per model where feasible;
- checkpoint resume;
- AdamW;
- learning-rate scheduling;
- validation during training;
- PPL;
- BPB;
- greedy/temperature generation;
- BLEU;
- chrF/chrF++;
- ROUGE-L;
- repetition;
- Distinct-1/Distinct-2;
- attention heatmaps;
- attention entropy;
- mean attention distance.

Phase 3 will require:
- independent synthetic reasoning datasets in Hindi and Nepali;
- train/validation/test splits;
- reproducible generation scripts;
- fine-tuning from each language's own pretrained checkpoint;
- pretrained-vs-finetuned attention analysis;
- final cross-resource comparison.

Therefore:
- keep tokenizer APIs stable;
- keep token IDs reproducible;
- save tokenizer configuration;
- save special-token definitions;
- save vocabulary files/model files;
- keep corpus metadata;
- keep deterministic split definitions;
- do not hard-code paths to one machine;
- do not build the pipeline around temporary Colab paths.

Do not make architecture decisions in Phase 1 unless needed for tokenizer sizing compatibility.

---

# 8. Things that must NOT be done

Do NOT:
- use a pretrained language model;
- use a pretrained tokenizer;
- use HuggingFace Transformer model classes;
- share Hindi and Nepali vocabulary;
- share Hindi and Nepali model weights;
- transfer-learn Nepali from Hindi;
- mix Hindi and Nepali training data;
- count HPLT web data as our manually collected data;
- count an already-preprocessed Wikipedia dataset as our manually collected data;
- label downloaded data as manual merely because it was downloaded manually;
- commit large datasets to Git;
- put raw corpus files directly into the repository;
- ignore the 20% manual requirement;
- claim 500M tokens without actually measuring tokenizer token counts;
- silently delete large amounts of text without reporting it;
- scrape comments/ads/navigation as if they were language data;
- use test data for training;
- use test data to justify tokenizer selection;
- make Phase 1 dependent on external dashboards for graded results.

---

# 9. Handling uncertainty

When a decision is not specified by the assignment, do not invent a rigid requirement.

Instead:
1. identify the reasonable options;
2. measure what can be measured;
3. choose the simplest defensible option;
4. document the choice;
5. keep the implementation configurable.

This is especially important for:
- exact corpus size;
- exact manual fraction;
- tokenizer algorithm;
- vocabulary size;
- train/validation/test proportions;
- near-duplicate thresholds;
- language-filter thresholds.

Do not optimize for theoretical perfection at the expense of completing Phase 1 on time.

---

# 10. Working principle for this 72-hour deadline

Prioritize correctness and evidence over sophistication.

Priority order:

1. Satisfy the manual-data requirement for both languages.
2. Obtain enough clean data.
3. Correctly separate/provenance-tag all sources.
4. Build reliable preprocessing and deduplication.
5. Create reproducible splits.
6. Train and evaluate tokenizer candidates.
7. Produce required statistics.
8. Produce a clear Phase 1 report.
9. Make the repository reproducible.
10. Only then add optional refinements.

If time becomes tight, do not sacrifice:
- provenance;
- manual percentage;
- data correctness;
- tokenizer reproducibility;
- report evidence;
- Git/branch correctness.

Cut optional experimentation before cutting those items.

---

# 11. Definition of done for Phase 1

Phase 1 is complete only when the repository can demonstrate:

**Hindi**
- reproducible collection/preprocessing;
- raw + cleaned corpus stored externally;
- separate train/validation/test splits;
- >=20% manually collected final training tokens;
- final tokenizer and vocabulary;
- tokenizer statistics;
- source/provenance statistics.

**Nepali**
- reproducible collection/preprocessing;
- raw + cleaned corpus stored externally;
- separate train/validation/test splits;
- >=20% manually collected final training tokens;
- final tokenizer and vocabulary;
- tokenizer statistics;
- source/provenance statistics.

**Repository**
- `phase-1` branch;
- code organized by language;
- README with reproduction instructions;
- Google Drive links;
- Phase 1 report;
- figures/tables inside the repository;
- meaningful commit history;
- no large binary/data artifacts committed to Git.

Do not declare Phase 1 complete merely because the tokenizer trained successfully. The deliverable is the entire dataset + preprocessing + split + tokenizer pipeline and its evidence.

---

# 12. Final instruction to the implementation agent

Work incrementally.

After every major stage:
1. run the relevant code;
2. inspect outputs;
3. check statistics;
4. look for silent failures;
5. fix problems before moving on;
6. make a meaningful Git commit.

Do not fabricate dataset sizes, token counts, manual percentages, source statistics, or experimental results.

If an external source fails, becomes unavailable, or differs from its expected format, adapt the pipeline and record the actual behavior rather than inventing a result.

The final implementation must be explainable by the student. Prefer straightforward, readable Python over opaque abstractions.

The assignment explicitly states that AI-assisted code may be used, but the student must be able to explain the code and how it was modified/verified. Therefore, every non-trivial implementation choice should be understandable from the code and documentation.

The objective is not to create the most sophisticated Phase 1 possible. The objective is to create a clean, defensible, reproducible Phase 1 that gives Phase 2 and Phase 3 a strong foundation.
