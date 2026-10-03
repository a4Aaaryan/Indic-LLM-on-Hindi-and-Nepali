"""Stage 1 integration and sampling test suite.

Performs small-scale tests of all data acquisition pipelines:
1. HPLT v2 streaming (Hindi & Nepali)
2. Wikipedia XML extraction (checks existing files + validates parsing)
3. Web scrapers (Dainik Jagran, Amar Ujala, Live Hindustan, Onlinekhabar, Ratopati)
4. Strict schema & provenance validation for all generated records.
"""

import io
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import ensure_dir, resolve_path
from common.hplt_downloader import HPLTStreamDownloader
from common.schema import validate_record
from common.utils import setup_logger
from common.web_scraper import (
    AmarUjalaScraper,
    HindustanScraper,
    JagranScraper,
    OnlinekhabarScraper,
    RatopatiScraper,
)
from common.wiki_extractor import WikipediaExtractor, clean_wikitext


def test_hplt_sampling(logger):
    logger.info("--- Testing HPLT v2 Streaming Sampling with Compression & Budget ---")
    results = {}
    for lang in ["hi", "ne"]:
        test_out = resolve_path(f"scratch/test_hplt_{lang}.jsonl.gz")
        downloader = HPLTStreamDownloader(language=lang, output_file=test_out, compress=True)
        # Test download of 3 records directly to gzipped file with token budget
        saved_count = downloader.download_to_file(max_samples=3, target_tokens=10000)
        results[lang] = saved_count
        assert saved_count > 0, f"HPLT {lang} compressed download returned 0 records"
        assert test_out.exists(), f"Gzipped output file {test_out} not created"
        assert test_out.stat().st_size > 0, f"Gzipped file {test_out} is empty"

        # Verify reading back from the gzipped file and validating schema
        import gzip
        with gzip.open(test_out, "rt", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == saved_count
        for rec_dict in lines:
            assert validate_record(rec_dict), f"Invalid schema in HPLT {lang} record: {rec_dict}"
            assert rec_dict["source_type"] == "downloaded"
            assert rec_dict["language"] == lang
            assert "approx_pretokenized_tokens_heuristic" in rec_dict["metadata"]
        logger.info("Successfully fetched %d HPLT %s compressed sample records (Schema verified, size: %d bytes).", len(lines), lang, test_out.stat().st_size)
    return results


def test_wikipedia_extraction(logger):
    logger.info("--- Testing Wikipedia XML Extraction ---")
    results = {}

    sample_wiki_xml = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
  <page>
    <title>भारत</title>
    <ns>0</ns>
    <id>101</id>
    <revision>
      <id>1001</id>
      <timestamp>2026-08-01T10:00:00Z</timestamp>
      <text bytes="500">'''भारत''' (आधिकारिक नाम: ''भारत गणराज्य'') दक्षिण एशिया में स्थित एक देश है। यह भौगोलिक दृष्टि से विश्व का सातवाँ सबसे बड़ा देश है।

== इतिहास ==
भारत का इतिहास अति प्राचीन और समृद्ध है। [[सिंधु घाटी सभ्यता]] यहाँ की प्राचीनतम सभ्यताओं में से एक है।

== भूगोल ==
भारत के उत्तर में [[हिमालय]] पर्वतमाला स्थित है।</text>
    </revision>
  </page>
  <page>
    <title>नेपाल</title>
    <ns>0</ns>
    <id>202</id>
    <revision>
      <id>2002</id>
      <timestamp>2026-08-01T11:00:00Z</timestamp>
      <text bytes="500">'''नेपाल''' (आधिकारिक नाम: ''सङ्घीय लोकतान्त्रिक गणतन्त्र नेपाल'') दक्षिण एसियाको एक भूपरिवेष्ठित देश हो। यसको राजधानी [[काठमाडौँ]] हो।

== भूगोल ==
नेपालको उत्तरमा [[चीन]] र दक्षिण, पूर्व तथा पश्चिममा [[भारत]] पर्दछ। [[सगरमाथा]] नेपालमै पर्दछ।</text>
    </revision>
  </page>
</mediawiki>"""

    test_xml_path = resolve_path("scratch/sample_wiki_test.xml")
    ensure_dir(test_xml_path.parent)
    with open(test_xml_path, "w", encoding="utf-8") as f:
        f.write(sample_wiki_xml)

    for lang in ["hi", "ne"]:
        real_xml = resolve_path(f"{'hindi' if lang == 'hi' else 'nepali'}/data/raw/{lang}wiki-latest-pages-articles.xml")
        exists = real_xml.exists()
        size_bytes = real_xml.stat().st_size if exists else 0
        logger.info("Detected local %s Wikipedia dump at %s (exists=%s, size=%d bytes)", lang, real_xml, exists, size_bytes)

        # Test extractor on the validated XML sample
        extractor = WikipediaExtractor(xml_path=test_xml_path, language=lang)
        records = list(extractor.extract_articles(max_articles=2))
        assert len(records) > 0, f"Wikipedia extractor failed on test XML for {lang}"
        for r in records:
            rec_dict = r.to_dict()
            assert validate_record(rec_dict), f"Invalid schema in Wiki {lang} record: {rec_dict}"
            assert r.source_type == "manual"
        results[lang] = {
            "real_file": str(real_xml),
            "exists": exists,
            "size_bytes": size_bytes,
            "extractor_tested": True,
            "test_sample_count": len(records),
        }
        logger.info("Wikipedia extractor validated for %s.", lang)

    return results


def test_web_scrapers(logger):
    logger.info("--- Testing Web Scrapers ---")
    scrapers = [
        ("hindi", JagranScraper(output_dir="scratch/scrapes/jagran")),
        ("hindi", AmarUjalaScraper(output_dir="scratch/scrapes/amarujala")),
        ("hindi", HindustanScraper(output_dir="scratch/scrapes/hindustan")),
        ("nepali", OnlinekhabarScraper(output_dir="scratch/scrapes/onlinekhabar")),
        ("nepali", RatopatiScraper(output_dir="scratch/scrapes/ratopati")),
    ]

    results = {}
    for lang, scraper in scrapers:
        logger.info("Testing %s (%s)...", scraper.source_name, lang)
        records = list(scraper.scrape_articles(max_articles=2))
        results[scraper.source_name] = len(records)
        assert len(records) > 0, f"Scraper {scraper.source_name} returned 0 articles"
        for r in records:
            rec_dict = r.to_dict()
            assert validate_record(rec_dict), f"Invalid schema for {scraper.source_name}: {rec_dict}"
            assert r.source_type == "manual"
            assert r.language == ("hi" if lang == "hindi" else "ne")
            assert len(r.text) >= 100, f"Scraped text too short for {scraper.source_name}: {len(r.text)}"
        logger.info("Successfully scraped and validated %d articles from %s.", len(records), scraper.source_name)

    return results


def main():
    logger = setup_logger("test_stage1")
    logger.info("=== Running Stage 1 Sampling & Validation Tests ===")

    hplt_results = test_hplt_sampling(logger)
    wiki_results = test_wikipedia_extraction(logger)
    scraper_results = test_web_scrapers(logger)

    logger.info("=== All Stage 1 Pipeline Tests Passed Successfully! ===")
    logger.info("HPLT Results: %s", hplt_results)
    logger.info("Wikipedia Results: %s", wiki_results)
    logger.info("Web Scraper Results: %s", scraper_results)


if __name__ == "__main__":
    main()
