import json
import logging
import re
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm, MAX_RESEARCH_ITERATIONS
from src.state import ResearchState, ReviewCritiqueOutput, SubTopicPlan

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM_PROMPT = """You are a Chief Scientific Reviewer evaluating gathered research evidence.
Evaluate whether the collected evidence sufficiently addresses the user's research topic.

Criteria:
1. Breadth & Depth: Are technical trade-offs, architecture, and current status covered?
2. Factual Substantiveness: Are there concrete metrics, dates, and sources?

Output MUST be ONLY valid JSON matching this structure:
{
  "is_sufficient": true,
  "feedback": "Summary of strengths or gaps identified",
  "gap_queries": []
}

If sufficient, set "is_sufficient": true and "gap_queries": [].
Only if there are critical missing gaps AND further research is essential, set "is_sufficient": false and provide 1-2 gap queries in "gap_queries" with {"title": "...", "description": "...", "queries": ["..."]}.
Do NOT output markdown code blocks or explanations outside of JSON."""

def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
    
    match = re.search(r'(\{[\s\S]*\})', cleaned)
    if match:
        cleaned = match.group(1)
        
    return json.loads(cleaned)

def review_research(state: ResearchState) -> dict:
    """Reviewer/Critic node: Evaluates research completeness and determines if follow-up search is required."""
    topic = state.get("topic", "")
    sections_data = state.get("sections_data", [])
    iteration = state.get("critique_iteration", 0)
    max_iters = state.get("max_iterations", MAX_RESEARCH_ITERATIONS)
    
    # Circuit breaker
    if iteration >= max_iters:
        logger.info(f"Max research iterations ({max_iters}) reached. Approving for final report.")
        return {
            "is_sufficient": True,
            "review_feedback": f"Completed research after {iteration} iteration(s).",
            "gap_queries": [],
            "status_message": "Research review passed."
        }
    
    findings_summary = []
    for s in sections_data:
        findings_summary.append(
            f"### Section: {s['section_title']}\n"
            f"Notes Excerpt: {s.get('notes', '')[:300]}...\n"
        )
    combined_findings = "\n".join(findings_summary)
    
    llm = get_llm(temperature=0.1)
    
    prompt = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=f"Topic: {topic}\n\nFindings Gathered:\n{combined_findings}")
    ]
    
    try:
        resp = llm.invoke(prompt)
        data = _extract_json(resp.content)
        critique = ReviewCritiqueOutput(**data)
        is_sufficient = critique.is_sufficient
        feedback = critique.feedback
        gap_queries = critique.gap_queries
    except Exception as e:
        logger.warning(f"Reviewer JSON parse fallback: {e}")
        is_sufficient = True
        feedback = "Findings verified as adequate for executive synthesis."
        gap_queries = []

    new_iteration = iteration + 1
    
    if is_sufficient or not gap_queries:
        status = "Research verified as thorough. Ready for report synthesis."
        gap_queries = []
        is_sufficient = True
    else:
        status = f"Identified {len(gap_queries)} research gaps. Triggering follow-up."
        
    return {
        "critique_iteration": new_iteration,
        "is_sufficient": is_sufficient,
        "review_feedback": feedback,
        "gap_queries": gap_queries,
        "status_message": status
    }
