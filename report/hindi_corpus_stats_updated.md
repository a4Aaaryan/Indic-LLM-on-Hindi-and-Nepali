# Updated Hindi Stage 1 Corpus Statistics Report

**Date**: 18 August 2026
**Heuristic**: $\text{Approx Tokens} = \text{Whitespace Words} \times 1.35$  (Excluding HPLT, which uses exact 385M heuristic estimate).

---

## 1. Hindi Corpus Data Breakdown

| Source Name | Category | Document Count | Word Count | Approx. Tokens | Container File Size |
|---|---|---|---|---|---|
| **Wikipedia** | Manual | 168,299 | 51,803,550 | 69,934,792 (~69.93M) | 784.02 MB |
| **Wikisource** | Manual | 167,471 | 52,805,749 | 71,287,761 (~71.29M) | 798.40 MB |
| **Dainik Jagran** | Manual | 30 | 12,308 | 16,615 (~0.02M) | (Shared in News) |
| **Amar Ujala** | Manual | 85 | 34,901 | 47,116 (~0.05M) | (Shared in News) |
| **Live Hindustan** | Manual | 119 | 73,708 | 99,505 (~0.10M) | 1.94 MB (News Total) |
| **HPLT (indic_hplt_v2)** | Downloaded | 614,314 | 285,401,242 | 385,000,000 (~385.00M)* | 847.72 MB |

*\* Note: The HPLT token count uses the exact downloaded heuristic estimate of ~385M tokens, rather than multiplying word counts by 1.35 again.*

---

## 2. Totals and Token Percentages

### Total Hindi Manual-Data Corpus (Excluding HPLT)
- **Total Manual Documents**: 336,004
- **Total Manual Words**: 104,730,216
- **Total Manual Approx. Tokens**: **141,385,791 (~141.39M)**

### Total Overall Hindi Corpus (Manual + HPLT)
- **Total Corpus Documents**: 950,318
- **Total Corpus Words**: 390,131,458
- **Total Corpus Approx. Tokens**: **526,385,791 (~526.39M)**

### Manual Token Percentage
- **Manual-Token Percentage**: **26.86%**
*(Requirement of $\ge 20\%$ manual tokens is strongly satisfied).*

---

## 3. Major Data File Sizes on Disk

| File | Description | Size |
|---|---|---|
| `hiwiki-latest-pages-articles.xml` | Raw Hindi Wikipedia XML dump | 1.56 GB |
| `hiwikisource-latest-pages-articles` | Raw Hindi Wikisource XML dump | 0.86 GB |
| `hindi_wikipedia.jsonl` | Processed manual JSONL | 784.02 MB |
| `hindi_wikisource.jsonl` | Processed manual JSONL | 798.40 MB |
| `hindi_news_scraped.jsonl` | Processed manual JSONL | 1.94 MB |
| `indic_hplt_v2_hi.jsonl.gz` | Downloaded HPLT JSONL gzip | 847.72 MB |

---
**Conclusion**: With the addition of Hindi Wikisource and the HPLT downloads, the Hindi corpus targets are fully met, providing over 526M tokens in total, with manual data accounting for 26.86% of the set.
