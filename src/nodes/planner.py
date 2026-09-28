import json
import logging
import re
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm
from src.state import ResearchState, ResearchPlanOutput, SubTopicPlan

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are an elite Research Strategist.
Decompose the given topic into a rigorous research plan.

Instructions:
1. Break the topic down into 3 distinct, non-overlapping sub-topics/thematic sections.
2. For each sub-topic, formulate 2 to 3 targeted search queries designed to find factual data, statistics, and recent benchmarks.
3. Queries must be concise, neutral, and search-engine friendly.
4. Output MUST be ONLY valid JSON matching this structure:
{
  "summary": "Brief 1-2 sentence overview of the research scope",
  "sections": [
    {
      "title": "Subtopic Title",
      "description": "What aspects to investigate",
      "queries": ["query 1", "query 2"]
    }
  ]
}
Do NOT include any markdown code blocks, conversational pleasantries, or explanations. Return raw JSON only."""

def _extract_json(text: str) -> dict:
    """Robustly extracts JSON from an LLM completion."""
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
    
    # Match outermost JSON object if wrapped in other text
    match = re.search(r'(\{[\s\S]*\})', cleaned)
    if match:
        cleaned = match.group(1)
        
    return json.loads(cleaned)

def plan_research(state: ResearchState) -> dict:
    """Planner node: Decomposes topic into structured sub-topics and targeted queries."""
    topic = state.get("topic", "")
    logger.info(f"Generating research plan for topic: '{topic}'")
    
    llm = get_llm(temperature=0.2)
    
    prompt = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Topic to research:\n{topic}")
    ]
    
    data = None
    for attempt in range(2):
        try:
            resp = llm.invoke(prompt)
            data = _extract_json(resp.content)
            plan = ResearchPlanOutput(**data)
            sections = plan.sections
            summary = plan.summary
            break
        except Exception as e:
            logger.warning(f"Planner attempt {attempt+1} failed: {e}")
            if attempt == 0:
                import time; time.sleep(1)
            else:
                sections = [
                    SubTopicPlan(
                        title=f"Core Architecture & Concepts of {topic}",
                        description=f"In-depth investigation of {topic}",
                        queries=[f"{topic} overview", f"{topic} architecture", f"{topic} key components"]
                    ),
                    SubTopicPlan(
                        title=f"Real-World Implementations & Performance",
                        description=f"Practical use cases and performance analysis of {topic}",
                        queries=[f"{topic} benchmarks", f"{topic} production use cases", f"{topic} trade-offs"]
                    )
                ]
                summary = f"Comprehensive investigation of {topic}"

    return {
        "plan_summary": summary,
        "sections": sections,
        "status_message": f"Planned {len(sections)} research areas: {', '.join(s.title for s in sections)}"
    }
