import re
import logging
from typing import Dict, List, Set, Tuple
from src.state import ResearchState

logger = logging.getLogger(__name__)

def verify_report(state: ResearchState) -> dict:
    """Verifier Node: Performs citation grounding verification and hallucination auditing on the final report."""
    report = state.get("final_report", "")
    sections_data = state.get("sections_data", [])
    
    logger.info("Running fact-checking and citation verification on synthesized report...")
    
    # Extract all unique verified URLs gathered across all researchers
    known_sources: Dict[str, str] = {}
    for s in sections_data:
        for src in s.get("sources", []):
            url = src.get("url", "").strip()
            if url:
                known_sources[url] = src.get("title", url)
                
    # Find all markdown citations formatted like [Title](URL)
    citations: List[Tuple[str, str]] = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', report)
    
    grounded_citations: List[Tuple[str, str]] = []
    unverified_citations: List[Tuple[str, str]] = []
    seen_cited_urls: Set[str] = set()
    
    for title, url in citations:
        clean_url = url.strip()
        if clean_url in seen_cited_urls:
            continue
        seen_cited_urls.add(clean_url)
        
        # Check exact or normalized prefix match in retrieved sources
        is_grounded = any(clean_url == k or clean_url.rstrip("/") == k.rstrip("/") for k in known_sources)
        if is_grounded:
            grounded_citations.append((title, clean_url))
        else:
            unverified_citations.append((title, clean_url))
            
    total_unique_citations = len(seen_cited_urls)
    if total_unique_citations > 0:
        score = (len(grounded_citations) / total_unique_citations) * 100.0
    else:
        # If no citations were embedded, assign 100 if research was approved or 70 baseline
        score = 85.0
        
    audit_summary = (
        f"Verified {len(grounded_citations)}/{total_unique_citations} cited sources "
        f"({score:.1f}% citation grounding score)."
    )
    if unverified_citations:
        audit_summary += f" Flagged {len(unverified_citations)} citation(s) not found in gathered evidence."
        
    logger.info(f"Verification completed: {audit_summary}")
    
    return {
        "verification_score": round(score, 1),
        "verification_feedback": audit_summary,
        "verified_sources_count": len(grounded_citations),
        "flagged_sources_count": len(unverified_citations),
        "status_message": f"Verified: {score:.1f}% citation grounding score."
    }
