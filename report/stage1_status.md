# Stage 1 Status Report: Source Acquisition & Provenance Tracking

**Project**: Building and Analyzing Monolingual Transformer LMs in Hindi and Nepali  
**Branch**: `phase-1`  
**Date**: 17 August 2026  

---

## 1. Sources Implemented & Supported

| Language | Source Name | Source Type | Acquisition Method | Target Output Path |
|---|---|---|---|---|
| **Hindi** (`hi`) | `indic_hplt_v2` | Downloaded | Streaming filter (`lang == 'hi'`) from `ashtok897/indic-hplt-v2` with resumable JSON checkpointing | `hindi/data/raw/indic_hplt_v2_hi.jsonl` |
| **Hindi** (`hi`) | `hindi_wikipedia` | Manual | Constant-memory XML streaming extractor (`ElementTree.iterparse`) from raw local XML dump | `hindi/data/raw/hindi_wikipedia.jsonl` |
| **Hindi** (`hi`) | `dainik_jagran` | Manual | Polite web scraper (RSS discovery + article parser + raw HTML archive) | `hindi/data/raw/hindi_news_scraped.jsonl` |
| **Hindi** (`hi`) | `amar_ujala` | Manual | Polite web scraper (RSS discovery + article parser + raw HTML archive) | `hindi/data/raw/hindi_news_scraped.jsonl` |
| **Hindi** (`hi`) | `live_hindustan` | Manual | Polite web scraper (Section crawl + article parser + raw HTML archive) | `hindi/data/raw/hindi_news_scraped.jsonl` |
| **Nepali** (`ne`) | `indic_hplt_v2` | Downloaded | Streaming filter (`lang == 'ne'`) from `ashtok897/indic-hplt-v2` with resumable JSON checkpointing | `nepali/data/raw/indic_hplt_v2_ne.jsonl` |
| **Nepali** (`ne`) | `nepali_wikipedia` | Manual | Constant-memory XML streaming extractor (`ElementTree.iterparse`) from raw local XML dump | `nepali/data/raw/nepali_wikipedia.jsonl` |
| **Nepali** (`ne`) | `onlinekhabar` | Manual | Polite web scraper (RSS discovery + article parser + raw HTML archive) | `nepali/data/raw/nepali_news_scraped.jsonl` |
| **Nepali** (`ne`) | `ratopati` | Manual | Polite web scraper (RSS discovery + article parser + raw HTML archive) | `nepali/data/raw/nepali_news_scraped.jsonl` |

---

## 2. Validation & Test Results

All 9 sources were tested using [`scripts/test_stage1_sample.py`](file:///c:/Users/arped/OneDrive/Desktop/LMA/scripts/test_stage1_sample.py):

| Pipeline Category | Target Component | Test Sample Count | Status | Schema & Provenance Compliance |
|---|---|---|---|---|
| **HPLT Streaming** | Hindi (`hi`) | 3 documents | **PASSED** | 100% compliant (`source_type: downloaded`) |
| **HPLT Streaming** | Nepali (`ne`) | 3 documents | **PASSED** | 100% compliant (`source_type: downloaded`) |
| **Wikipedia XML** | Streaming Extractor (Hindi & Nepali) | 2 articles each | **PASSED** | 100% compliant (`source_type: manual`) |
| **News Scraper** | Dainik Jagran | 2 articles | **PASSED** | 100% compliant (`source_type: manual`) |
| **News Scraper** | Amar Ujala | 1 article | **PASSED** | 100% compliant (`source_type: manual`) |
| **News Scraper** | Live Hindustan | 2 articles | **PASSED** | 100% compliant (`source_type: manual`) |
| **News Scraper** | Onlinekhabar | 2 articles | **PASSED** | 100% compliant (`source_type: manual`) |
| **News Scraper** | Ratopati | 2 articles | **PASSED** | 100% compliant (`source_type: manual`) |

---

## 3. Detected Local Wikipedia Dumps

- **Hindi**: [`hindi/data/raw/hiwiki-latest-pages-articles.xml`](file:///c:/Users/arped/OneDrive/Desktop/LMA/hindi/data/raw/hiwiki-latest-pages-articles.xml)
  - Status: File present.
- **Nepali**: [`nepali/data/raw/newiki-latest-pages-articles.xml`](file:///c:/Users/arped/OneDrive/Desktop/LMA/nepali/data/raw/newiki-latest-pages-articles.xml)
  - Status: File present.

> **Note**: The XML streaming extractor ([`common/wiki_extractor.py`](file:///c:/Users/arped/OneDrive/Desktop/LMA/common/wiki_extractor.py)) operates with $O(1)$ memory usage via `xml.etree.ElementTree.iterparse`, safely processing multi-gigabyte Wikipedia dumps without high RAM consumption.

---

## 4. Provenance Record Schema

Every record produced across all pipelines strictly follows the standard schema:
```json
{
  "doc_id": "unique_string_id",
  "language": "hi", 
  "source_type": "manual",
  "source_name": "dainik_jagran",
  "text": "Extracted cleaned article text...",
  "url": "https://www.jagran.com/...",
  "title": "Article Title",
  "collection_date": "2026-08-17T05:13:00.000000+00:00",
  "raw_location": "hindi/data/raw/raw_html/dainik_jagran/...",
  "metadata": {
    "char_count": 2450
  }
}
```

---

## 5. Source-Specific Notes & Handling

1. **HPLT v2 Streaming**: The dataset `ashtok897/indic-hplt-v2` is streamed directly and filtered row-by-row using the `lang` column (`hi` and `ne`). Checkpoints are written to `hplt_<lang>_checkpoint.json` periodically so the process can be safely paused and resumed without re-downloading.
2. **Amar Ujala / Hindustan**: Dynamic advertisements and inline JSON blobs are filtered to retain only authentic Devanagari article body text.
3. **Web Scraper Rate Limiting**: All scrapers enforce polite request delays (`request_delay = 0.5s - 1.0s`) and custom browser User-Agent headers to adhere to polite crawling standards.

---

## 6. Execution Commands for Full Acquisition

When you are ready to execute the full data collection, run the following commands:

### A. Hindi Acquisition
```bash
# 1. Extract full Hindi Wikipedia dump (already local in raw/)
python hindi/scripts/01_collect_sources.py --source wikipedia

# 2. Scrape manual Hindi news sources (Jagran, Amar Ujala, Live Hindustan)
python hindi/scripts/01_collect_sources.py --source news

# 3. Stream and download Indic HPLT v2 Hindi backbone
python hindi/scripts/01_collect_sources.py --source hplt
```

### B. Nepali Acquisition
```bash
# 1. Extract full Nepali Wikipedia dump (already local in raw/)
python nepali/scripts/01_collect_sources.py --source wikipedia

# 2. Scrape manual Nepali news sources (Onlinekhabar, Ratopati)
python nepali/scripts/01_collect_sources.py --source news

# 3. Stream and download Indic HPLT v2 Nepali backbone
python nepali/scripts/01_collect_sources.py --source hplt
```

### C. Run All Sources with Sample Limiting (for sanity checks)
```bash
python hindi/scripts/01_collect_sources.py --source all --sample-size 50
python nepali/scripts/01_collect_sources.py --source all --sample-size 50
```
