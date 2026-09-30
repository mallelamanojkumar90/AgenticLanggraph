import re
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import httpx
from lxml import html

logger = logging.getLogger(__name__)

# In-memory scrape cache to avoid redundant network calls during research runs
_SCRAPE_CACHE: Dict[str, str] = {}

# Ignored binary/media file extensions
_SKIPPED_EXTENSIONS = (
    ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".exe", ".bin"
)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def clean_html_text(html_content: bytes, max_chars: int = 1500) -> str:
    """Extracts clean human-readable text from raw HTML using lxml."""
    try:
        doc = html.fromstring(html_content)
        # Drop unwanted tags: scripts, styles, navigation, headers, footers
        for tag in doc.xpath(
            "//script | //style | //noscript | //nav | //footer | //header | //aside | //svg | //form | //iframe"
        ):
            tag.drop_tree()
        
        # Priority on main article/content tags if available
        article_elements = doc.xpath("//article | //main | //div[contains(@class, 'content') or contains(@class, 'post')]")
        if article_elements:
            texts = [el.text_content() for el in article_elements]
            raw_text = " ".join(texts)
        else:
            texts = doc.xpath("//p//text() | //h1//text() | //h2//text() | //h3//text() | //li//text()")
            raw_text = " ".join(texts)
            
        cleaned = re.sub(r"\s+", " ", raw_text).strip()
        return cleaned[:max_chars]
    except Exception as e:
        logger.debug(f"HTML parsing fallback failed: {e}")
        return ""

def scrape_webpage(url: str, max_chars: int = 1500, timeout: float = 7.0) -> str:
    """Fetches and cleans the text of a web page.
    
    Uses a 2-tier resilient approach:
    1. Jina Reader API (https://r.jina.ai/{url}) - clean markdown, handles dynamic JS
    2. Direct HTTP GET with lxml HTML parsing as a fallback
    
    Results are cached in memory for the duration of the process.
    """
    if not url or not url.startswith(("http://", "https://")):
        return ""
        
    lower_url = url.lower().split("?")[0]
    if any(lower_url.endswith(ext) for ext in _SKIPPED_EXTENSIONS):
        return ""
        
    if url in _SCRAPE_CACHE:
        return _SCRAPE_CACHE[url]
        
    # Tier 1: Try Jina Reader API (free, fast, clean markdown)
    jina_url = f"https://r.jina.ai/{url}"
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(
                jina_url,
                headers={
                    "User-Agent": DEFAULT_HEADERS["User-Agent"],
                    "Accept": "text/markdown",
                    "X-Timeout": "6",
                }
            )
            if resp.status_code == 200 and len(resp.text.strip()) > 100:
                text = resp.text
                if "Markdown Content:" in text:
                    text = text.split("Markdown Content:", 1)[1]
                # Normalize line breaks and spaces
                cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
                cleaned = cleaned[:max_chars].strip()
                if len(cleaned) > 80:
                    _SCRAPE_CACHE[url] = cleaned
                    logger.info(f"Successfully scraped via Jina Reader: {url} ({len(cleaned)} chars)")
                    return cleaned
    except Exception as e:
        logger.debug(f"Jina Reader failed for {url}: {e}")

    # Tier 2: Direct HTTP GET with lxml cleaner fallback
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=DEFAULT_HEADERS)
            if resp.status_code == 200 and resp.content:
                cleaned = clean_html_text(resp.content, max_chars=max_chars)
                if len(cleaned) > 80:
                    _SCRAPE_CACHE[url] = cleaned
                    logger.info(f"Successfully scraped directly: {url} ({len(cleaned)} chars)")
                    return cleaned
    except Exception as e:
        logger.debug(f"Direct scrape failed for {url}: {e}")

    # If all scraping fails, store empty string so we don't re-attempt
    _SCRAPE_CACHE[url] = ""
    return ""

def scrape_multiple_webpages(urls: List[str], max_chars_per_page: int = 1500, max_workers: int = 3) -> Dict[str, str]:
    """Scrapes multiple URLs concurrently using a thread pool."""
    valid_urls = [u for u in urls if u and u.startswith(("http://", "https://"))]
    if not valid_urls:
        return {}
        
    results: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(len(valid_urls), max_workers)) as executor:
        future_to_url = {
            executor.submit(scrape_webpage, url, max_chars_per_page): url
            for url in valid_urls
        }
        for future in future_to_url:
            url = future_to_url[future]
            try:
                content = future.result()
                if content:
                    results[url] = content
            except Exception as e:
                logger.warning(f"Error scraping {url}: {e}")
                
    return results
