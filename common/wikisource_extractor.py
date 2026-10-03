"""Streaming Wikisource XML dump extractor for Hindi and Nepali with constant memory usage.

Parses MediaWiki XML dumps incrementally using ElementTree.iterparse,
filtering main and page namespaces (ns in ('0', '104', '250')), stripping wikitext boilerplate,
and converting them to standardized DocumentRecords with full provenance tracking.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional
import urllib.parse
import re

from common.schema import DocumentRecord
from common.config import resolve_path
from common.wiki_extractor import clean_wikitext, WikipediaExtractor


CORE_NEPALI_MARKERS = {
    "लाई", "हरू", "गर्छ", "छन्", "भयो", "थियो", "गरेको", "हुने", "भने", "बाट",
    "भनेर", "छैन", "रहेको", "गर्दछ", "गर्ने", "हुन्छ", "भएका", "हुन्", "गर्छन्",
    "भनेपछि", "गर्दैन", "हुँदैन", "भएको", "गर्दा", "हुँदा", "भएपछि", "भएर",
    "हुनुपर्छ", "गर्नुपर्छ", "गरेका"
}


def is_nepali_wikisource_text(text: str, title: str) -> bool:
    """Accurately identify authentic Nepali works in a multilingual Wikisource dump (sourceswiki)."""
    # 1. Explicit Category or Language template header
    if re.search(r"\[\[Category:(नेपाली|Nepali|नेपाली साहित्य|नेपाल|नेपाली कविता|नेपाली कृतिहरू)", text, re.IGNORECASE):
        return True
    if re.search(r"(\|\s*language\s*=\s*(ne|nepali|नेपाली)|\{\{Langinfo\|ne)", text, re.IGNORECASE):
        return True

    # Exclude explicitly marked Hindi content in multilingual dumps
    if "हिंदी नाटक" in text or "हिन्दी नाटक" in text or "भारतेंदु" in text:
        return False

    # 2. Devanagari script ratio check
    dev_chars = len(re.findall(r"[\u0900-\u097F]", text))
    if len(text) < 40 or (dev_chars / max(1, len(text))) < 0.4:
        return False

    # 3. Match distinct Nepali grammatical postpositions/morphemes
    tokens = set(re.findall(r"[\u0900-\u097F]+", text))
    matches = tokens.intersection(CORE_NEPALI_MARKERS)
    return len(matches) >= 4


class WikisourceExtractor(WikipediaExtractor):
    """Streams and extracts articles and proofread pages from a MediaWiki XML dump for Wikisource."""

    def __init__(self, xml_path: str | Path, language: str):
        super().__init__(xml_path, language)
        self.language = language
        self.source_name = f"{'hindi' if language == 'hi' else 'nepali'}_wikisource"
        if language == "hi":
            self.base_url = "https://hi.wikisource.org/wiki/"
        else:
            self.base_url = "https://wikisource.org/wiki/"

    def extract_articles(self, max_articles: Optional[int] = None) -> Iterator[DocumentRecord]:
        """Stream Wikisource articles and proofread pages with O(1) memory usage."""
        if not self.xml_path.exists() or self.xml_path.stat().st_size == 0:
            return

        articles_extracted = 0
        is_multilingual = "sourceswiki" in str(self.xml_path)

        try:
            context = ET.iterparse(str(self.xml_path), events=("end",))
            for event, elem in context:
                tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

                if tag_name == "page":
                    title_elem = elem.find("./title") or elem.find(".//{*}title")
                    ns_elem = elem.find("./ns") or elem.find(".//{*}ns")
                    id_elem = elem.find("./id") or elem.find(".//{*}id")
                    redirect_elem = elem.find("./redirect") or elem.find(".//{*}redirect")

                    revision = elem.find("./revision") or elem.find(".//{*}revision")
                    text_elem = revision.find("./text") or revision.find(".//{*}text") if revision is not None else None
                    timestamp_elem = revision.find("./timestamp") or revision.find(".//{*}timestamp") if revision is not None else None

                    title = title_elem.text if title_elem is not None and title_elem.text else ""
                    ns = ns_elem.text if ns_elem is not None and ns_elem.text else "0"
                    page_id = id_elem.text if id_elem is not None and id_elem.text else str(articles_extracted)
                    raw_text = text_elem.text if text_elem is not None and text_elem.text else ""
                    timestamp = timestamp_elem.text if timestamp_elem is not None else ""

                    # Process main namespace (ns == '0') and Wikisource page namespaces ('104', '250')
                    if ns in ("0", "104", "250") and redirect_elem is None and raw_text:
                        # For multilingual dumps (sourceswiki), filter for authentic Nepali content
                        if self.language == "ne" and is_multilingual:
                            if not is_nepali_wikisource_text(raw_text, title):
                                elem.clear()
                                continue

                        cleaned = clean_wikitext(raw_text)
                        if len(cleaned.strip()) >= 40:  # Skip empty boilerplate
                            encoded_title = urllib.parse.quote(title.replace(" ", "_"))
                            url = f"{self.base_url}{encoded_title}"

                            record = DocumentRecord(
                                doc_id=f"wikisource_{self.language}_{page_id}",
                                language=self.language,
                                source_type="manual",
                                source_name=self.source_name,
                                text=cleaned,
                                url=url,
                                title=title,
                                raw_location=str(self.xml_path),
                                metadata={
                                    "page_id": page_id,
                                    "namespace": ns,
                                    "timestamp": timestamp,
                                    "raw_char_count": len(raw_text),
                                    "cleaned_char_count": len(cleaned),
                                },
                            )
                            yield record
                            articles_extracted += 1
                            if max_articles is not None and articles_extracted >= max_articles:
                                elem.clear()
                                break

                    elem.clear()
        except ET.ParseError:
            pass


class NepaliWikisourceExtractor(WikisourceExtractor):
    """Dedicated extractor for Nepali Wikisource dumps."""

    def __init__(self, xml_path: str | Path):
        super().__init__(xml_path, language="ne")
