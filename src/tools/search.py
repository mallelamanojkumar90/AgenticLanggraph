import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def search_web(query: str, max_results: int = 4) -> List[Dict[str, str]]:
    """Performs web search using DuckDuckGo and returns a list of results.
    
    Each result has:
    - title: Title of the web page
    - url: Direct URL
    - snippet: Summary / body text
    """
    results = []
    
    # Try importing from ddgs first, then duckduckgo_search
    try:
        from ddgs import DDGS
        ddgs_client = DDGS()
    except ImportError:
        try:
            from duckduckgo_search import DDGS
            ddgs_client = DDGS()
        except Exception as e:
            logger.error(f"Failed to initialize DuckDuckGo search: {e}")
            return []

    try:
        raw_results = list(ddgs_client.text(query, max_results=max_results))
        for item in raw_results:
            title = item.get("title", "").strip()
            url = item.get("href", item.get("link", "")).strip()
            snippet = item.get("body", item.get("snippet", "")).strip()
            
            if title and url:
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })
    except Exception as e:
        logger.warning(f"Search failed for query '{query}': {e}")
        
    return results
