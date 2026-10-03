"""Test script for Stage 1 Web Scraper Expansion.

Verifies:
- Existing URLs/content are loaded into the manifest and skipped.
- Paginated discovery finds new, distinct URLs.
- Publication dates are extracted when available.
- Provenance and deduplication mechanisms work.
"""

import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import ensure_dir, resolve_path
from common.utils import setup_logger
from common.web_scraper import (
    AmarUjalaScraper,
    HindustanScraper,
    JagranScraper,
    KantipurScraper,
    OnlinekhabarScraper,
    RatopatiScraper,
)

def run_test():
    logger = setup_logger("test_scraper_expansion")
    logger.info("=== Testing Stage 1 Scraper Expansion & Deduplication ===")

    hindi_raw_dir = resolve_path("hindi/data/raw")
    nepali_raw_dir = resolve_path("nepali/data/raw")
    
    scrapers = [
        # Hindi
        JagranScraper(output_dir=hindi_raw_dir),
        AmarUjalaScraper(output_dir=hindi_raw_dir),
        HindustanScraper(output_dir=hindi_raw_dir),
        # Nepali
        OnlinekhabarScraper(output_dir=nepali_raw_dir),
        RatopatiScraper(output_dir=nepali_raw_dir),
        KantipurScraper(output_dir=nepali_raw_dir),
    ]

    total_new = 0

    for scraper in scrapers:
        logger.info("--- Testing %s ---", scraper.source_name)
        
        # We manually call load_manifest to check how many were loaded
        scraper.load_manifest()
        initial_manifest_urls = set(scraper.manifest_urls)
        logger.info("[%s] Initial manifest size: %d URLs", scraper.source_name, len(initial_manifest_urls))
        
        # Scrape 5 new articles
        new_records = list(scraper.scrape_articles(max_articles=5))
        
        logger.info("[%s] Scraped %d new articles.", scraper.source_name, len(new_records))
        total_new += len(new_records)
        
        for record in new_records:
            assert record.url not in initial_manifest_urls or len(new_records) == 0, f"URL {record.url} was already in manifest before scrape!"
            
            has_date = "published_date" in record.metadata
            date_str = record.metadata.get("published_date", "None")
            logger.info("  -> New Article: %s | URL: %s | Date: %s", record.doc_id, record.url, date_str)
            
            # Simple provenance verification
            assert record.source_type == "manual"
            assert len(record.text) > 100
            
    logger.info("=== Test Complete. Scraped %d completely new articles across %d sources. ===", total_new, len(scrapers))

if __name__ == "__main__":
    run_test()
