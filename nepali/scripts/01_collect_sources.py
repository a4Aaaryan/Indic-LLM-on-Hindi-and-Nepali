"""Stage 1: Source acquisition and provenance tracking for Nepali (Model L).

Supports:
1. Downloaded: Indic HPLT v2 Nepali subset (streaming from Hugging Face with budget control and gzip compression)
2. Manual: Local Nepali Wikipedia XML dump extraction (newiki-latest-pages-articles.xml)
3. Manual News Scrapers: Onlinekhabar, Ratopati

All outputs are saved as standardized JSONL (or JSONL.GZ) files in nepali/data/raw/
with full provenance records according to the project schema.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import ensure_dir, load_yaml_config, resolve_path
from common.hplt_downloader import HPLTStreamDownloader
from common.provenance import ProvenanceTracker
from common.utils import set_seed, setup_logger
from common.web_scraper import (
    EPustakalayaScraper,
    KantipurScraper,
    EKantipurScraper,
    NepaliArchivesScraper,
    OnlinekhabarScraper,
    RatopatiScraper,
    SetopatiScraper,
)
from common.wiki_extractor import WikipediaExtractor
from common.wikisource_extractor import NepaliWikisourceExtractor, WikisourceExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Acquire and track Nepali dataset sources.")
    parser.add_argument(
        "--config",
        type=str,
        default="nepali/configs/phase1_config.yaml",
        help="Path to Phase 1 configuration YAML",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "hplt", "wikipedia", "wikisource", "news", "onlinekhabar", "ratopati", "kantipur", "ekantipur", "setopati", "epustakalaya", "nepaliarchives"],
        help="Source to collect/extract",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit number of articles/records collected per source (useful for testing)",
    )
    parser.add_argument(
        "--target-approx-tokens",
        "--target-tokens",
        dest="target_approx_tokens",
        type=int,
        default=385_000_000,
        help="Heuristic pre-tokenization token budget for HPLT raw ingestion (default ~385M to provide a safe buffer for cleaning/dedup towards ~500M total post-tokenization budget)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=True,
        help="Save raw JSONL files as gzip compressed .jsonl.gz to minimize disk usage (default: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify configuration and connections without full extraction/scraping",
    )
    return parser.parse_args()


def collect_wikipedia(raw_dir: Path, sample_size: int | None, logger) -> int:
    xml_path = raw_dir / "newiki-latest-pages-articles.xml"
    out_file = raw_dir / "nepali_wikipedia.jsonl"
    logger.info("Extracting Nepali Wikipedia from %s -> %s", xml_path, out_file)

    if not xml_path.exists() or xml_path.stat().st_size == 0:
        logger.warning("Nepali Wikipedia dump %s is empty or not found. Skipping extraction.", xml_path)
        return 0

    extractor = WikipediaExtractor(xml_path=xml_path, language="ne")
    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for record in extractor.extract_articles(max_articles=sample_size):
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
            if count % 500 == 0:
                logger.info("Extracted %d Wikipedia articles...", count)

    logger.info("Completed Nepali Wikipedia extraction: %d articles.", count)
    return count


def collect_wikisource(raw_dir: Path, sample_size: int | None, logger) -> int:
    candidate_paths = [
        raw_dir / "sourceswiki-latest-pages-articles.xml",
        raw_dir / "sourceswiki-latest-pages-articles",
        raw_dir / "newikisource-latest-pages-articles.xml",
        raw_dir / "newikisource-latest-pages-articles",
    ]
    xml_path = None
    for p in candidate_paths:
        if p.exists() and p.stat().st_size > 100_000:  # Prefer full dumps over tiny stubs
            xml_path = p
            break
    if xml_path is None:
        for p in candidate_paths:
            if p.exists() and p.stat().st_size > 0:
                xml_path = p
                break

    if xml_path is None:
        xml_path = candidate_paths[0]

    out_file = raw_dir / "nepali_wikisource.jsonl"
    logger.info("Extracting Nepali Wikisource from %s -> %s", xml_path, out_file)

    if not xml_path.exists() or xml_path.stat().st_size == 0:
        logger.warning("Nepali Wikisource dump %s is empty or not found. Skipping extraction.", xml_path)
        return 0

    extractor = NepaliWikisourceExtractor(xml_path=xml_path)
    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for record in extractor.extract_articles(max_articles=sample_size):
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
            if count % 500 == 0:
                logger.info("Extracted %d Wikisource articles...", count)

    logger.info("Completed Nepali Wikisource extraction: %d articles.", count)
    return count


def collect_hplt(raw_dir: Path, sample_size: int | None, target_tokens: int | None, compress: bool, logger) -> int:
    ext = ".jsonl.gz" if compress else ".jsonl"
    out_file = raw_dir / f"indic_hplt_v2_ne{ext}"
    logger.info("Streaming Indic HPLT v2 Nepali subset (budget: %s tokens, compress=%s) -> %s", target_tokens, compress, out_file)
    downloader = HPLTStreamDownloader(language="ne", output_file=out_file, compress=compress)
    return downloader.download_to_file(max_samples=sample_size, target_tokens=target_tokens)


def collect_news_scraper(scraper, out_file: Path, sample_size: int | None, logger) -> int:
    logger.info("Running scraper for %s -> %s", scraper.source_name, out_file)
    count = 0
    mode = "a" if out_file.exists() else "w"
    with open(out_file, mode, encoding="utf-8") as f:
        for record in scraper.scrape_articles(max_articles=sample_size):
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
            logger.info("[%s] Scraped article: %s", scraper.source_name, record.title or record.doc_id)
    return count


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    logger = setup_logger("nepali_collect")
    set_seed(config.get("targets", {}).get("random_seed", 42))

    raw_dir = ensure_dir(config["paths"]["data_raw_dir"])
    logger.info("Nepali Acquisition Pipeline initialized. Raw directory: %s", raw_dir)

    if args.dry_run:
        logger.info("Dry-run mode: Configuration validated. Sources configured: %s", list(config["sources"].keys()))
        return

    results = {}

    # 1. Wikipedia
    if args.source in ("all", "wikipedia"):
        results["nepali_wikipedia"] = collect_wikipedia(raw_dir, args.sample_size, logger)

    if args.source in ("all", "wikisource"):
        results["nepali_wikisource"] = collect_wikisource(raw_dir, args.sample_size, logger)

    # 2. HPLT v2
    if args.source in ("all", "hplt"):
        results["indic_hplt_v2"] = collect_hplt(raw_dir, args.sample_size, args.target_approx_tokens, args.compress, logger)

    # 3. News Sources
    news_out_file = raw_dir / "nepali_news_scraped.jsonl"
    if args.source in ("all", "news", "onlinekhabar"):
        results["onlinekhabar"] = collect_news_scraper(
            OnlinekhabarScraper(output_dir=raw_dir), news_out_file, args.sample_size, logger
        )
    if args.source in ("all", "news", "ratopati"):
        results["ratopati"] = collect_news_scraper(
            RatopatiScraper(output_dir=raw_dir), news_out_file, args.sample_size, logger
        )
    if args.source in ("all", "news", "kantipur"):
        results["kantipur"] = collect_news_scraper(
            KantipurScraper(output_dir=raw_dir), news_out_file, args.sample_size, logger
        )
    if args.source in ("all", "news", "ekantipur"):
        results["ekantipur"] = collect_news_scraper(
            EKantipurScraper(output_dir=raw_dir), news_out_file, args.sample_size, logger
        )
    if args.source in ("all", "news", "setopati"):
        results["setopati"] = collect_news_scraper(
            SetopatiScraper(output_dir=raw_dir), news_out_file, args.sample_size, logger
        )

    # New scrapers go to the same news_out_file (or we can use nepali_archives_scraped.jsonl, but it's fine)
    if args.source in ("all", "epustakalaya"):
        results["epustakalaya"] = collect_news_scraper(
            EPustakalayaScraper(output_dir=raw_dir), news_out_file, args.sample_size, logger
        )
    if args.source in ("all", "nepaliarchives"):
        results["nepaliarchives"] = collect_news_scraper(
            NepaliArchivesScraper(output_dir=raw_dir), news_out_file, args.sample_size, logger
        )

    logger.info("Nepali acquisition stage run complete. Results summary: %s", results)


if __name__ == "__main__":
    main()
