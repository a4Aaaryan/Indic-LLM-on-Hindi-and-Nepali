"""Streaming Wikipedia XML dump extractor with constant memory usage.

Parses MediaWiki XML dumps incrementally using ElementTree.iterparse,
filtering main-namespace encyclopedia articles (ns == 0), stripping wikitext boilerplate,
and converting them to standardized DocumentRecords.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional
import urllib.parse

from common.config import resolve_path
from common.schema import DocumentRecord


def clean_wikitext(text: str) -> str:
    """Clean common MediaWiki markup to extract plain article text."""
    if not text:
        return ""

    # Check for redirect
    if text.strip().startswith(("#REDIRECT", "#redirect", "#पुनर्निर्देशन", "#पुनर्निर्देशित")):
        return ""

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove references <ref...>...</ref> and self-closing <ref.../>
    text = re.sub(r"<ref[^>/]*>.*?</ref>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<ref[^>]*/>", "", text, flags=re.IGNORECASE)

    # Remove tables {| ... |}
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)

    # Remove nested templates {{ ... }}
    # Run twice for simple nested templates
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    # Remove category links [[Category:...]] or [[श्रेणी:...]]
    text = re.sub(r"\[\[(Category|श्रेणी|वर्ग):[^\]]*\]\]", "", text, flags=re.IGNORECASE)

    # Remove file/image links [[File:...]] or [[चित्र:...]]
    text = re.sub(r"\[\[(File|Image|चित्र|फाइल):[^\]]*\]\]", "", text, flags=re.IGNORECASE)

    # Convert piped links [[Target|Anchor]] -> Anchor
    text = re.sub(r"\[\[[^\]|]*\|([^\]]+)\]\]", r"\1", text)

    # Convert simple links [[Target]] -> Target
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # Convert external links [http... Anchor] -> Anchor
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\]", "", text)

    # Remove section headers == Header == -> Header
    text = re.sub(r"={2,6}\s*(.*?)\s*={2,6}", r"\1", text)

    # Remove bold/italic markup
    text = re.sub(r"'{2,5}", "", text)

    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    lines = [line.strip() for line in text.splitlines()]
    clean_lines = [line for line in lines if line]
    return "\n\n".join(clean_lines)


class WikipediaExtractor:
    """Streams and extracts articles from a MediaWiki XML dump."""

    def __init__(self, xml_path: str | Path, language: str):
        self.xml_path = resolve_path(xml_path)
        self.language = language
        self.source_name = f"{'hindi' if language == 'hi' else 'nepali'}_wikipedia"
        self.base_url = f"https://{language}.wikipedia.org/wiki/"

    def extract_articles(self, max_articles: Optional[int] = None) -> Iterator[DocumentRecord]:
        """Stream articles from XML dump without loading the full file into RAM."""
        if not self.xml_path.exists() or self.xml_path.stat().st_size == 0:
            return

        articles_extracted = 0

        # Context manager / iterparse for streaming XML
        try:
            context = ET.iterparse(str(self.xml_path), events=("end",))
            for event, elem in context:
                # MediaWiki tag names may have xmlns prefixes like {http://www.mediawiki.org/xml/export-0.10/}page
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

                    # Process only main namespace (ns == 0) and non-redirects
                    if ns == "0" and redirect_elem is None and raw_text:
                        cleaned = clean_wikitext(raw_text)
                        if len(cleaned.strip()) >= 50:  # Skip trivial/empty articles
                            encoded_title = urllib.parse.quote(title.replace(" ", "_"))
                            url = f"{self.base_url}{encoded_title}"

                            record = DocumentRecord(
                                doc_id=f"wiki_{self.language}_{page_id}",
                                language=self.language,
                                source_type="manual",
                                source_name=self.source_name,
                                text=cleaned,
                                url=url,
                                title=title,
                                raw_location=str(self.xml_path),
                                metadata={
                                    "page_id": page_id,
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
            # Handle truncated or invalid XML gracefully
            pass
