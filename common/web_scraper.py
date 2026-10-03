"""Web scrapers for Hindi and Nepali news sources with polite rate-limiting, paginated discovery, and deduplication."""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import requests

from common.config import ensure_dir, resolve_path
from common.schema import DocumentRecord
from common.utils import setup_logger

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "hi,ne,en;q=0.9",
}


class BaseNewsScraper:
    """Base scraper providing common HTTP fetch, rate limiting, deduplication, and article parsing helpers."""

    def __init__(
        self,
        source_name: str,
        language: str,
        output_dir: str | Path,
        request_delay: float = 1.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.source_name = source_name
        self.language = language
        self.output_dir = ensure_dir(output_dir)
        self.raw_html_dir = ensure_dir(self.output_dir / "raw_html" / self.source_name)
        self.request_delay = request_delay
        self.headers = headers or DEFAULT_HEADERS
        self.logger = setup_logger(f"scraper_{source_name}")
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self.manifest_urls = set()
        self.manifest_hashes = set()

    def hash_content(self, text: str) -> str:
        """Create an exact hash of normalized full article text for deduplication."""
        normalized_text = re.sub(r"\s+", "", text)
        return hashlib.md5(normalized_text.encode("utf-8")).hexdigest()

    def load_manifest(self):
        """Load existing scraped news to populate manifest for deduplication."""
        file_prefix = "hindi" if self.language == "hi" else "nepali"
        manifest_file = self.output_dir / f"{file_prefix}_news_scraped.jsonl"
        
        if not manifest_file.exists():
            self.logger.info("[%s] No existing manifest found at %s", self.source_name, manifest_file)
            return

        count = 0
        with open(manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("source_name") == self.source_name:
                        if "url" in data and data["url"]:
                            self.manifest_urls.add(data["url"])
                        if "text" in data and data["text"]:
                            self.manifest_hashes.add(self.hash_content(data["text"]))
                        count += 1
                except json.JSONDecodeError:
                    continue
        self.logger.info("[%s] Loaded %d existing records into deduplication manifest.", self.source_name, count)

    def fetch_url(self, url: str, timeout: int = 15) -> Optional[str]:
        """Fetch URL content with rate limiting and error handling."""
        time.sleep(self.request_delay)
        try:
            response = self.session.get(url, timeout=timeout)
            if response.status_code == 200:
                response.encoding = 'utf-8'
                return response.text
            self.logger.warning("HTTP %d for URL: %s", response.status_code, url)
        except Exception as e:
            self.logger.error("Failed to fetch %s: %s", url, e)
        return None

    def clean_html_to_text(self, html: str) -> str:
        """Strip boilerplate, ads, scripts, and navigation to extract clean article paragraphs."""
        soup = BeautifulSoup(html, "html.parser")

        # Decompose non-content elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "form", "iframe", "noscript", "button"]):
            element.decompose()

        # Find main article container if available
        container = soup.find("article") or soup.find("main") or soup

        paragraphs = []
        for p in container.find_all("p"):
            text = p.get_text().strip()
            # Retain paragraphs with substantial text, skipping JSON blobs and download promos
            if len(text) >= 35 and not text.startswith(("{", "[")) and not any(x in text for x in ["डाउनलोड करें", "विज्ञापन देखें", "app download", "Follow us"]):
                paragraphs.append(text)

        # Fallback if no <p> tags had sufficient text
        if not paragraphs:
            for div in container.find_all("div"):
                if div.find(["div", "p"]):
                    continue
                text = div.get_text().strip()
                if len(text) >= 50 and not text.startswith(("{", "[")):
                    paragraphs.append(text)

        return "\n\n".join(paragraphs)

    def extract_published_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Attempt to extract publication date from common meta tags."""
        date_meta = soup.find("meta", property="article:published_time") or \
                    soup.find("meta", itemprop="datePublished") or \
                    soup.find("meta", attrs={"name": "pubdate"}) or \
                    soup.find("meta", attrs={"name": "publish-date"})
        
        if date_meta and date_meta.get("content"):
            return date_meta["content"]
        
        # Try JSON-LD
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and "datePublished" in data:
                    return data["datePublished"]
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "datePublished" in item:
                            return item["datePublished"]
            except Exception:
                continue
        return None

    def save_raw_html(self, doc_id: str, html: str) -> str:
        """Save raw HTML for reproducibility and provenance."""
        file_path = self.raw_html_dir / f"{doc_id}.html"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)
        return str(file_path)

    def discover_article_urls(self) -> Iterator[str]:
        """Discover article URLs through pagination or category browsing. Yields indefinitely."""
        raise NotImplementedError("Subclasses must implement discover_article_urls")

    def scrape_articles(self, max_articles: Optional[int] = None) -> Iterator[DocumentRecord]:
        """Scrape articles, check manifest, deduplicate, and yield standardized DocumentRecords."""
        self.load_manifest()

        scraped_count = 0
        for url in self.discover_article_urls():
            if max_articles is not None and scraped_count >= max_articles:
                self.logger.info("[%s] Reached maximum requested articles: %d", self.source_name, max_articles)
                break

            if url in self.manifest_urls:
                self.logger.info("[%s] Skipping known URL: %s", self.source_name, url)
                continue

            html = self.fetch_url(url)
            if not html:
                continue

            clean_text = self.clean_html_to_text(html)
            if len(clean_text) < 100:  # Skip empty or failed extractions
                continue

            content_hash = self.hash_content(clean_text)
            if content_hash in self.manifest_hashes:
                self.logger.info("[%s] Skipping duplicate content for URL: %s", self.source_name, url)
                self.manifest_urls.add(url)
                continue

            url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
            doc_id = f"{self.source_name}_{url_hash}"
            raw_path = self.save_raw_html(doc_id, html)

            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.get_text().strip() if soup.title else ""
            pub_date = self.extract_published_date(soup)

            record = DocumentRecord(
                doc_id=doc_id,
                language=self.language,
                source_type="manual",
                source_name=self.source_name,
                text=clean_text,
                url=url,
                title=title,
                raw_location=raw_path,
                metadata={
                    "char_count": len(clean_text),
                    "url": url,
                },
            )
            
            if pub_date:
                record.metadata["published_date"] = pub_date

            self.manifest_urls.add(url)
            self.manifest_hashes.add(content_hash)
            scraped_count += 1
            yield record


class JagranScraper(BaseNewsScraper):
    def __init__(self, output_dir: str | Path, request_delay: float = 0.5):
        super().__init__("dainik_jagran", "hi", output_dir, request_delay)

    def discover_article_urls(self) -> Iterator[str]:
        categories = ["national-news-hindi", "world-news-hindi", "business-hindi"]
        page = 1
        while True:
            found_any = False
            for cat in categories:
                url = f"https://www.jagran.com/news/{cat}.html" if page == 1 else f"https://www.jagran.com/news/{cat}-page{page}.html"
                html = self.fetch_url(url)
                if html:
                    soup = BeautifulSoup(html, "html.parser")
                    links = [a["href"] for a in soup.find_all("a", href=True) if ".html" in a["href"] and len(a["href"]) > 25]
                    for link in set(links):
                        if not link.startswith("http"):
                            link = "https://www.jagran.com" + link
                        yield link
                        found_any = True
            if not found_any or page > 50:
                break
            page += 1


class AmarUjalaScraper(BaseNewsScraper):
    def __init__(self, output_dir: str | Path, request_delay: float = 0.5):
        super().__init__("amar_ujala", "hi", output_dir, request_delay)

    def discover_article_urls(self) -> Iterator[str]:
        categories = ["india-news", "world", "business"]
        page = 1
        while True:
            found_any = False
            for cat in categories:
                url = f"https://www.amarujala.com/{cat}" if page == 1 else f"https://www.amarujala.com/{cat}/page-{page}"
                html = self.fetch_url(url)
                if html:
                    soup = BeautifulSoup(html, "html.parser")
                    links = [a["href"] for a in soup.find_all("a", href=True) if ".html" in a["href"] and len(a["href"]) > 25]
                    for link in set(links):
                        if not link.startswith("http"):
                            link = "https://www.amarujala.com" + link
                        yield link
                        found_any = True
            if not found_any or page > 50:
                break
            page += 1


class HindustanScraper(BaseNewsScraper):
    def __init__(self, output_dir: str | Path, request_delay: float = 0.5):
        super().__init__("live_hindustan", "hi", output_dir, request_delay)

    def discover_article_urls(self) -> Iterator[str]:
        categories = ["national", "international", "business"]
        page = 1
        while True:
            found_any = False
            for cat in categories:
                url = f"https://www.livehindustan.com/{cat}/" if page == 1 else f"https://www.livehindustan.com/{cat}/page-{page}/"
                html = self.fetch_url(url)
                if html:
                    soup = BeautifulSoup(html, "html.parser")
                    links = [a["href"] for a in soup.find_all("a", href=True) if ".html" in a["href"] and len(a["href"]) > 25]
                    for link in set(links):
                        if not link.startswith("http"):
                            link = "https://www.livehindustan.com" + link
                        yield link
                        found_any = True
            if not found_any or page > 50:
                break
            page += 1


class OnlinekhabarScraper(BaseNewsScraper):
    def __init__(self, output_dir: str | Path, request_delay: float = 0.5):
        super().__init__("onlinekhabar", "ne", output_dir, request_delay)

    def discover_article_urls(self) -> Iterator[str]:
        import datetime
        current_year = datetime.datetime.now().year
        current_month = datetime.datetime.now().month
        
        for year in range(current_year, 2005, -1):
            start_month = current_month if year == current_year else 12
            for month in range(start_month, 0, -1):
                page = 1
                while True:
                    url = f"https://www.onlinekhabar.com/{year}/{month:02d}" if page == 1 else f"https://www.onlinekhabar.com/{year}/{month:02d}/page/{page}"
                    html = self.fetch_url(url)
                    if not html:
                        break
                    
                    soup = BeautifulSoup(html, "html.parser")
                    links = [a["href"] for a in soup.find_all("a", href=True) if re.search(r"/\d{4}/\d{2}/\d+", a["href"])]
                    
                    found_new = False
                    for link in set(links):
                        if not link.startswith("http"):
                            link = "https://www.onlinekhabar.com" + link
                        if "english.onlinekhabar.com" not in link:
                            yield link
                            found_new = True
                            
                    if not found_new:
                        break
                    page += 1


class RatopatiScraper(BaseNewsScraper):
    def __init__(self, output_dir: str | Path, request_delay: float = 0.5):
        super().__init__("ratopati", "ne", output_dir, request_delay)

    def discover_article_urls(self) -> Iterator[str]:
        categories = ["news", "business", "sports"]
        page = 1
        while True:
            found_any = False
            for cat in categories:
                url = f"https://www.ratopati.com/category/{cat}" if page == 1 else f"https://www.ratopati.com/category/{cat}?page={page}"
                html = self.fetch_url(url)
                if html:
                    soup = BeautifulSoup(html, "html.parser")
                    links = [a["href"] for a in soup.find_all("a", href=True) if "/story/" in a["href"]]
                    for link in set(links):
                        if not link.startswith("http"):
                            link = "https://www.ratopati.com" + link
                        yield link
                        found_any = True
            if not found_any or page > 50:
                break
            page += 1


class KantipurScraper(BaseNewsScraper):
    def __init__(self, output_dir: str | Path, request_delay: float = 0.5):
        super().__init__("kantipur", "ne", output_dir, request_delay)

    def discover_article_urls(self) -> Iterator[str]:
        categories = ["news", "business", "sports"]
        page = 1
        while True:
            found_any = False
            for cat in categories:
                url = f"https://ekantipur.com/{cat}" if page == 1 else f"https://ekantipur.com/{cat}?page={page}"
                html = self.fetch_url(url)
                if html:
                    soup = BeautifulSoup(html, "html.parser")
                    links = [a["href"] for a in soup.find_all("a", href=True) if re.search(r"/\d{4}/\d{2}/\d+", a["href"])]
                    for link in set(links):
                        if not link.startswith("http"):
                            link = "https://ekantipur.com" + link
                        yield link
                        found_any = True
            if not found_any or page > 50:
                break
            page += 1


class EPustakalayaScraper(BaseNewsScraper):
    """Scraper for E-Pustakalaya (OLE Nepal) digital library."""

    def __init__(self, output_dir: str | Path, request_delay: float = 1.0):
        super().__init__("e_pustakalaya", "ne", output_dir, request_delay)

    def discover_article_urls(self) -> Iterator[str]:
        """Discover documents from E-Pustakalaya search endpoints."""
        types = ["document", "literature"]
        for search_type in types:
            page = 1
            while True:
                url = f"https://pustakalaya.org/en/search/?q=&type={search_type}&page={page}"
                html = self.fetch_url(url)
                found_any = False
                if html:
                    soup = BeautifulSoup(html, "html.parser")
                    links = [
                        urljoin("https://pustakalaya.org", a["href"])
                        for a in soup.find_all("a", href=True)
                        if "/documents/detail/" in a["href"]
                    ]
                    for link in set(links):
                        yield link
                        found_any = True
                if not found_any or page >= 20:  # Paginate up to page 20 per category
                    break
                page += 1

    def clean_html_to_text(self, html: str) -> str:
        """Extract clean Nepali document metadata and summary descriptions."""
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "nav", "header", "footer", "form", "button"]):
            element.decompose()

        title_elem = soup.find("h1") or soup.find("h2") or soup.title
        title = title_elem.get_text().strip() if title_elem else ""
        title = re.sub(r"E-Pustakalaya\s*\|\s*Document\s*\|\s*", "", title).strip()

        paragraphs = []
        if title:
            paragraphs.append(title)

        for p in soup.find_all(["p", "div", "td"]):
            text = p.get_text().strip()
            # Ignore UI boilerplate and non-Devanagari container blocks
            if len(text) >= 30 and not any(
                k in text for k in [
                    "All Document Audio Video", "TITLES", "AUTHOR", "CATEGORIES",
                    "E-PUSTAKALAYA", "CONTACT US", "DEVELOPED", "Creative Commons",
                    "Download App", "Write a Review", "review this item", "Add to favourite"
                ]
            ):
                if re.search(r"[\u0900-\u097F]", text) and text not in paragraphs:
                    paragraphs.append(text)

        return "\n\n".join(paragraphs)


class NepaliArchivesScraper(BaseNewsScraper):
    """Scraper for Nepali Archives (nepaliarchives.org) public domain literary collections."""

    def __init__(self, output_dir: str | Path, request_delay: float = 0.5):
        super().__init__("nepali_archives", "ne", output_dir, request_delay)

    def discover_article_urls(self) -> Iterator[str]:
        """Discover literary works from author and genre indexes."""
        base_url = "https://www.nepaliarchives.org"
        author_slugs = [
            "devkota", "lekhnath_paudyal", "bhimnidhi_tiwari",
            "bhanubhakta_acharya", "motiram_bhatta"
        ]
        genre_slugs = [
            "mahakavya", "khandakavya", "kavita", "nibandha",
            "upanyas", "katha", "balkavita", "git", "gazal", "bhajan"
        ]

        seen_urls = set()

        # 1. Discover via author directories
        for auth in author_slugs:
            auth_url = f"{base_url}/authors/{auth}/"
            html = self.fetch_url(auth_url)
            if html:
                soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if re.match(r"^[a-z0-9_]+/$", href) and href not in [
                        "authors/", "genres/", "collections/", "stats/", "patro/"
                    ]:
                        full_url = urljoin(auth_url, href)
                        if full_url not in seen_urls:
                            seen_urls.add(full_url)
                            yield full_url

        # 2. Discover via genre directories
        for genre in genre_slugs:
            genre_url = f"{base_url}/genres/{genre}/"
            html = self.fetch_url(genre_url)
            if html:
                soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if re.match(r"^[a-z0-9_]+/$", href) and href not in [
                        "authors/", "genres/", "collections/", "stats/", "patro/"
                    ]:
                        full_url = urljoin(genre_url, href)
                        if full_url not in seen_urls:
                            seen_urls.add(full_url)
                            yield full_url

    def clean_html_to_text(self, html: str) -> str:
        """Extract clean Nepali literary text, poems, and essays."""
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button"]):
            element.decompose()

        # Extract title
        title_elem = soup.title or soup.find("h1")
        title = title_elem.get_text().strip() if title_elem else ""
        title = re.sub(r"—\s*नेपाली अभिलेख.*", "", title).strip()

        # Extract text lines
        raw_text = soup.get_text()
        lines = []
        for line in raw_text.splitlines():
            line_str = line.strip()
            # Filter out navigation arrows and footer text
            if len(line_str) >= 2 and not line_str.startswith(("←", "→", "©", "नेपाली अभिलेख", "📖", "⬇")):
                lines.append(line_str)

        return "\n".join(lines)


class EKantipurScraper(BaseNewsScraper):
    def __init__(self, output_dir: str | Path, request_delay: float = 0.5):
        super().__init__("ekantipur", "ne", output_dir, request_delay)

    def discover_article_urls(self) -> Iterator[str]:
        import xml.etree.ElementTree as ET
        ns = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        page = 1
        
        while True:
            sitemap_url = f"https://ekantipur.com/sitemap_index/newsdetail/{page}"
            xml_data = self.fetch_url(sitemap_url)
            if not xml_data:
                break
                
            try:
                root = ET.fromstring(xml_data.encode('utf-8'))
                urls = root.findall('sitemap:url/sitemap:loc', ns)
                
                if not urls:
                    break
                    
                for url in urls:
                    loc = url.text
                    if loc and '/en/' not in loc:
                        yield loc
            except ET.ParseError as e:
                self.logger.error("[%s] XML ParseError on %s: %s", self.source_name, sitemap_url, e)
                break
                
            page += 1
            # Put an upper limit to avoid infinite loops, though eKantipur has thousands of pages
            if page > 5000:
                break

class SetopatiScraper(BaseNewsScraper):
    def __init__(self, output_dir: str | Path, request_delay: float = 0.5):
        super().__init__("setopati", "ne", output_dir, request_delay)

    def discover_article_urls(self) -> Iterator[str]:
        categories = ["politics", "social", "kinmel", "sports", "global", "opinion", "art", "blog", "literature", "ghumphir", "technology"]
        page = 1
        while True:
            found_any = False
            for cat in categories:
                url = f"https://www.setopati.com/{cat}?page={page}"
                html = self.fetch_url(url)
                if html:
                    soup = BeautifulSoup(html, "html.parser")
                    links = [a["href"] for a in soup.find_all("a", href=True) if f"/{cat}/" in a["href"]]
                    for link in set(links):
                        if not link.startswith("http"):
                            link = "https://www.setopati.com" + link
                        if link.count('/') > 3:
                            yield link
                            found_any = True
            if not found_any or page > 500:
                break
            page += 1

    def clean_html_to_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "form", "iframe", "noscript", "button"]):
            element.decompose()
            
        editor_box = soup.find("div", class_="editor-box")
        if editor_box:
            paragraphs = []
            for p in editor_box.find_all(["p", "div"]):
                if p.find(["p", "div"]): continue
                text = p.get_text(separator=" ", strip=True)
                if len(text) >= 35 and not any(x in text for x in ["Share", "Download"]):
                    if text not in paragraphs:
                        paragraphs.append(text)
            
            if paragraphs:
                return "\n\n".join(paragraphs)
            else:
                return editor_box.get_text(separator="\n", strip=True)
                
        return super().clean_html_to_text(html)
