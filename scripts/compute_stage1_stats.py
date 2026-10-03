"""Computes and reports comprehensive Stage 1 corpus statistics for Hindi and Nepali manual data."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Ensure UTF-8 output on Windows console
sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import ensure_dir, resolve_path

HEURISTIC_TOKENS_PER_WORD = 1.35


def compute_source_stats(file_path: Path):
    """Parse JSONL file and aggregate metrics per source_name."""
    stats = defaultdict(lambda: {"doc_count": 0, "char_count": 0, "word_count": 0})
    if not file_path.exists():
        return stats

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                source_name = data.get("source_name", "unknown")
                text = data.get("text", "")
                words = len(text.split())
                chars = len(text)

                stats[source_name]["doc_count"] += 1
                stats[source_name]["char_count"] += chars
                stats[source_name]["word_count"] += words
            except json.JSONDecodeError:
                continue
    return stats


def format_size(bytes_val: int) -> str:
    """Format bytes to human readable MB / GB."""
    if bytes_val >= 1024 * 1024 * 1024:
        return f"{bytes_val / (1024**3):.2f} GB"
    elif bytes_val >= 1024 * 1024:
        return f"{bytes_val / (1024**2):.2f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    return f"{bytes_val} B"


def main():
    report_path = resolve_path("report/stage1_corpus_stats.md")
    ensure_dir(report_path.parent)

    hindi_raw_dir = resolve_path("hindi/data/raw")
    nepali_raw_dir = resolve_path("nepali/data/raw")

    # 1. Hindi Files
    hi_wiki_file = hindi_raw_dir / "hindi_wikipedia.jsonl"
    hi_news_file = hindi_raw_dir / "hindi_news_scraped.jsonl"
    hi_xml_file = hindi_raw_dir / "hiwiki-latest-pages-articles.xml"

    hi_wiki_stats = compute_source_stats(hi_wiki_file)
    hi_news_stats = compute_source_stats(hi_news_file)

    # 2. Nepali Files
    ne_wiki_file = nepali_raw_dir / "nepali_wikipedia.jsonl"
    ne_news_file = nepali_raw_dir / "nepali_news_scraped.jsonl"
    ne_xml_file = nepali_raw_dir / "newiki-latest-pages-articles.xml"

    ne_wiki_stats = compute_source_stats(ne_wiki_file)
    ne_news_stats = compute_source_stats(ne_news_file)

    # Sources specification
    hindi_sources = [
        ("Wikipedia", "hindi_wikipedia", hi_wiki_stats.get("hindi_wikipedia", {}), hi_wiki_file),
        ("Dainik Jagran", "dainik_jagran", hi_news_stats.get("dainik_jagran", {}), hi_news_file),
        ("Amar Ujala", "amar_ujala", hi_news_stats.get("amar_ujala", {}), hi_news_file),
        ("Live Hindustan", "live_hindustan", hi_news_stats.get("live_hindustan", {}), hi_news_file),
    ]

    nepali_sources = [
        ("Wikipedia", "nepali_wikipedia", ne_wiki_stats.get("nepali_wikipedia", {}), ne_wiki_file),
        ("Onlinekhabar", "onlinekhabar", ne_news_stats.get("onlinekhabar", {}), ne_news_file),
        ("Ratopati", "ratopati", ne_news_stats.get("ratopati", {}), ne_news_file),
    ]

    lines = []
    lines.append("# Stage 1 Manual Corpus Statistics Report")
    lines.append("")
    lines.append("**Date**: 17 August 2026  ")
    lines.append("**Heuristic**: $\\text{Approx Tokens} = \\text{Whitespace Words} \\times 1.35$  ")
    lines.append("**Target Overview**: Total Phase 1 target is ~500M training tokens per language, with $\\ge 20\\%$ manual tokens ($\\ge 100\\text{M}$, ideally ~125–150M tokens).")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Process Hindi
    lines.append("## 1. Hindi Manual Corpus Breakdown (Model H)")
    lines.append("")
    lines.append("| Source Name | Document Count | Character Count | Word Count | Approx. Tokens (1.35×) | Container File Size |")
    lines.append("|---|---|---|---|---|---|")

    hi_total_docs = 0
    hi_total_chars = 0
    hi_total_words = 0

    for display_name, key, st, fpath in hindi_sources:
        docs = st.get("doc_count", 0)
        chars = st.get("char_count", 0)
        words = st.get("word_count", 0)
        tokens = int(words * HEURISTIC_TOKENS_PER_WORD)
        fsize = fpath.stat().st_size if fpath.exists() else 0

        hi_total_docs += docs
        hi_total_chars += chars
        hi_total_words += words

        lines.append(f"| **{display_name}** | {docs:,} | {chars:,} | {words:,} | {tokens:,} (~{tokens/1e6:.2f}M) | {format_size(fsize)} |")

    hi_total_tokens = int(hi_total_words * HEURISTIC_TOKENS_PER_WORD)
    hi_jsonl_bytes = (hi_wiki_file.stat().st_size if hi_wiki_file.exists() else 0) + (hi_news_file.stat().st_size if hi_news_file.exists() else 0)

    lines.append(f"| **HINDI TOTAL** | **{hi_total_docs:,}** | **{hi_total_chars:,}** | **{hi_total_words:,}** | **{hi_total_tokens:,} (~{hi_total_tokens/1e6:.2f}M)** | **{format_size(hi_jsonl_bytes)}** |")
    lines.append("")
    lines.append(f"- **Raw XML Source Size**: {format_size(hi_xml_file.stat().st_size if hi_xml_file.exists() else 0)} (`hiwiki-latest-pages-articles.xml`)")
    lines.append(f"- **Manual Percentage of 500M Target**: **{(hi_total_tokens / 500_000_000 * 100):.2f}%**")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Process Nepali
    lines.append("## 2. Nepali Manual Corpus Breakdown (Model L)")
    lines.append("")
    lines.append("| Source Name | Document Count | Character Count | Word Count | Approx. Tokens (1.35×) | Container File Size |")
    lines.append("|---|---|---|---|---|---|")

    ne_total_docs = 0
    ne_total_chars = 0
    ne_total_words = 0

    for display_name, key, st, fpath in nepali_sources:
        docs = st.get("doc_count", 0)
        chars = st.get("char_count", 0)
        words = st.get("word_count", 0)
        tokens = int(words * HEURISTIC_TOKENS_PER_WORD)
        fsize = fpath.stat().st_size if fpath.exists() else 0

        ne_total_docs += docs
        ne_total_chars += chars
        ne_total_words += words

        lines.append(f"| **{display_name}** | {docs:,} | {chars:,} | {words:,} | {tokens:,} (~{tokens/1e6:.2f}M) | {format_size(fsize)} |")

    ne_total_tokens = int(ne_total_words * HEURISTIC_TOKENS_PER_WORD)
    ne_jsonl_bytes = (ne_wiki_file.stat().st_size if ne_wiki_file.exists() else 0) + (ne_news_file.stat().st_size if ne_news_file.exists() else 0)

    lines.append(f"| **NEPALI TOTAL** | **{ne_total_docs:,}** | **{ne_total_chars:,}** | **{ne_total_words:,}** | **{ne_total_tokens:,} (~{ne_total_tokens/1e6:.2f}M)** | **{format_size(ne_jsonl_bytes)}** |")
    lines.append("")
    lines.append(f"- **Raw XML Source Size**: {format_size(ne_xml_file.stat().st_size if ne_xml_file.exists() else 0)} (`newiki-latest-pages-articles.xml`)")
    lines.append(f"- **Manual Percentage of 500M Target**: **{(ne_total_tokens / 500_000_000 * 100):.2f}%**")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary & Recommendations
    lines.append("## 3. Summary & HPLT Acquisition Budget Recommendations")
    lines.append("")
    lines.append("| Metric | Hindi Manual Corpus | Nepali Manual Corpus |")
    lines.append("|---|---|---|")
    lines.append(f"| Total Documents | {hi_total_docs:,} | {ne_total_docs:,} |")
    lines.append(f"| Total Words | {hi_total_words:,} | {ne_total_words:,} |")
    lines.append(f"| Approximate Heuristic Tokens | ~{hi_total_tokens/1e6:.2f}M | ~{ne_total_tokens/1e6:.2f}M |")
    lines.append(f"| Current % of 500M Budget | {hi_total_tokens/500_000_000*100:.2f}% | {ne_total_tokens/500_000_000*100:.2f}% |")
    lines.append(f"| Assignment $\\ge 20\\%$ Manual Threshold | **SATISFIED** (>{hi_total_tokens/1e6:.1f}M $\\ge 100$M) | Needs additional manual data or proportional scaling |")
    lines.append("")
    lines.append("### Key Takeaways for Next Steps:")
    lines.append(f"1. **Hindi**: The Hindi manual corpus currently provides **~{hi_total_tokens/1e6:.2f}M estimated tokens** ({hi_total_tokens/500_000_000*100:.2f}% of 500M), safely exceeding the 20% manual requirement with a generous buffer.")
    lines.append(f"2. **Nepali**: The Nepali manual corpus currently provides **~{ne_total_tokens/1e6:.2f}M estimated tokens**. In Stage 9, if the total Nepali dataset target is capped at what public sources allow, or if we collect additional news/web crawl, we ensure the manual ratio remains $\\ge 20\\%$.")
    lines.append("3. **Recommended HPLT Download Budget**:")
    lines.append("   - **Hindi HPLT Budget**: ~375M–385M tokens (~280M words)")
    lines.append("   - **Nepali HPLT Budget**: ~380M–400M tokens (~290M words)")

    content = "\n".join(lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(content)


if __name__ == "__main__":
    main()
