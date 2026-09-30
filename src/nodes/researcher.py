import logging
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm, MAX_SEARCH_RESULTS
from src.tools.search import search_web
from src.tools.scraper import scrape_multiple_webpages
from src.state import SubTopicTask, SectionData

logger = logging.getLogger(__name__)

RESEARCHER_SYSTEM_PROMPT = """You are a Principal Research Analyst specializing in deep technical and industry investigation.
Your task is to analyze raw web search results for a specific sub-topic and distill them into high-density, factual synthesis notes.

Guidelines:
1. Extract key facts, metrics, statistics, technical comparisons, historical milestones, and expert arguments.
2. Maintain objectivity: present both advantages and limitations/challenges.
3. Every major claim or data point should cite the source URL clearly like [Title](URL).
4. Organize your notes clearly with headers and bullet points. Avoid filler text.
"""

def conduct_research(task: SubTopicTask) -> dict:
    """Worker node: Runs parallel search queries for a sub-topic, scrapes full content from top sources, and synthesizes findings."""
    title = task.get("section_title", "Untitled Section")
    description = task.get("description", "")
    queries = task.get("queries", [])
    
    logger.info(f"Conducting research for section: '{title}' ({len(queries)} queries)")
    
    all_sources = []
    seen_urls = set()
    
    # 1. Run search for each query
    for q in queries:
        results = search_web(q, max_results=2)
        for r in results:
            url = r["url"]
            if url not in seen_urls:
                seen_urls.add(url)
                all_sources.append(r)
                if len(all_sources) >= 5:
                    break
        if len(all_sources) >= 5:
            break
            
    # 2. Deep scrape the top URLs for full-text context
    urls_to_scrape = [s["url"] for s in all_sources[:3]]
    scraped_data = scrape_multiple_webpages(urls_to_scrape, max_chars_per_page=1200)
    logger.info(f"Deep scraped {len(scraped_data)}/{len(urls_to_scrape)} web pages for section '{title}'")
    
    # 3. Assemble evidence blocks combining scraped text and fallback snippets
    collected_snippets = []
    for r in all_sources:
        url = r["url"]
        scraped_text = scraped_data.get(url, "")
        if scraped_text:
            r["scraped"] = True
            r["content_preview"] = scraped_text[:300]
            collected_snippets.append(
                f"- **{r['title']}** ({url}) [DEEP INGESTED]:\n  {scraped_text}"
            )
        else:
            r["scraped"] = False
            clean_snippet = r.get("snippet", "")[:300].strip()
            r["content_preview"] = clean_snippet
            collected_snippets.append(
                f"- **{r['title']}** ({url}) [SEARCH SNIPPET]:\n  {clean_snippet}"
            )
    
    # Synthesize findings with LLM
    if not collected_snippets:
        raw_text = "No live web search results could be retrieved. Provide analysis based on verified domain knowledge."
    else:
        raw_text = "\n".join(collected_snippets)
    
    llm = get_llm(temperature=0.1, max_tokens=1000)
    
    user_prompt = f"""Sub-topic: {title}
Description: {description}
Queries: {', '.join(queries)}

SEARCH FINDINGS:
{raw_text}

Provide concise, high-density synthesis notes for this sub-topic citing URLs where relevant."""

    notes = ""
    for attempt in range(2):
        try:
            resp = llm.invoke([
                SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])
            notes = resp.content.strip()
            break
        except Exception as e:
            logger.warning(f"Researcher LLM attempt {attempt+1} failed: {e}")
            if attempt == 1:
                notes = f"Research summary based on search findings:\n{raw_text}"
    
    section_result: SectionData = {
        "section_title": title,
        "queries_run": queries,
        "notes": notes,
        "sources": all_sources
    }
    
    # Return as list item to trigger the operator.add reducer in ResearchState
    return {
        "sections_data": [section_result]
    }
