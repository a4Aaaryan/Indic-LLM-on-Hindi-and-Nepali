"""Discovery-only audit to estimate available URLs."""

import sys
from pathlib import Path
from bs4 import BeautifulSoup
import time

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import resolve_path
from common.utils import setup_logger
from common.web_scraper import (
    AmarUjalaScraper,
    HindustanScraper,
    JagranScraper,
    KantipurScraper,
    OnlinekhabarScraper,
    RatopatiScraper,
)

def run_audit():
    logger = setup_logger("discovery_audit")
    logger.info("=== Starting Discovery Audit ===")
    
    hindi_raw_dir = resolve_path("hindi/data/raw")
    nepali_raw_dir = resolve_path("nepali/data/raw")
    
    scrapers = [
        JagranScraper(output_dir=hindi_raw_dir, request_delay=0.1),
        AmarUjalaScraper(output_dir=hindi_raw_dir, request_delay=0.1),
        HindustanScraper(output_dir=hindi_raw_dir, request_delay=0.1),
        OnlinekhabarScraper(output_dir=nepali_raw_dir, request_delay=0.1),
        RatopatiScraper(output_dir=nepali_raw_dir, request_delay=0.1),
        KantipurScraper(output_dir=nepali_raw_dir, request_delay=0.1),
    ]

    for scraper in scrapers:
        logger.info("--- Auditing %s ---", scraper.source_name)
        scraper.load_manifest()
        
        discovered_new = []
        # Override the request delay for discovery to speed it up
        scraper.request_delay = 0.1
        
        try:
            for url in scraper.discover_article_urls():
                if url not in scraper.manifest_urls and url not in discovered_new:
                    discovered_new.append(url)
                if len(discovered_new) >= 2000:
                    break
        except Exception as e:
            logger.error("Error discovering URLs for %s: %s", scraper.source_name, e)
            
        logger.info("[%s] Found %d new URLs.", scraper.source_name, len(discovered_new))
        
        # Check dates for first and last discovered URL
        dates = []
        for url in [discovered_new[0] if discovered_new else None, discovered_new[-1] if discovered_new else None]:
            if url:
                html = scraper.fetch_url(url)
                if html:
                    soup = BeautifulSoup(html, "html.parser")
                    date = scraper.extract_published_date(soup)
                    if date:
                        dates.append(date)
                        
        date_range_str = " - ".join(dates) if dates else "Unknown"
        logger.info("[%s] Approximate Date Range: %s", scraper.source_name, date_range_str)

if __name__ == "__main__":
    run_audit()
