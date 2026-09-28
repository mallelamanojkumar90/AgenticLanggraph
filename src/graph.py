import logging
from typing import List, Union
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from src.state import ResearchState, SubTopicTask
from src.nodes.planner import plan_research
from src.nodes.researcher import conduct_research
from src.nodes.reviewer import review_research
from src.nodes.writer import synthesize_report

logger = logging.getLogger(__name__)

def fan_out_research(state: ResearchState) -> List[Send]:
    """Conditional edge from planner: Fans out each sub-topic to parallel researcher workers."""
    sections = state.get("sections", [])
    logger.info(f"Fanning out research to {len(sections)} parallel researcher nodes.")
    
    return [
        Send("researcher", {
            "section_title": section.title,
            "description": section.description,
            "queries": section.queries
        })
        for section in sections
    ]

def route_after_review(state: ResearchState) -> Union[str, List[Send]]:
    """Conditional edge from reviewer: Either advances to writer or loops back with gap queries."""
    is_sufficient = state.get("is_sufficient", True)
    gap_queries = state.get("gap_queries", [])
    
    if is_sufficient or not gap_queries:
        logger.info("Research approved by reviewer. Advancing to report synthesis.")
        return "writer"
        
    logger.info(f"Reviewer requested further investigation on {len(gap_queries)} gap areas. Fanning out.")
    return [
        Send("researcher", {
            "section_title": gap.title,
            "description": gap.description,
            "queries": gap.queries
        })
        for gap in gap_queries
    ]

def build_research_graph(enable_memory: bool = True):
    """Builds and compiles the complete Deep Research LangGraph workflow."""
    builder = StateGraph(ResearchState)
    
    # 1. Add core nodes
    builder.add_node("planner", plan_research)
    builder.add_node("researcher", conduct_research)
    builder.add_node("reviewer", review_research)
    builder.add_node("writer", synthesize_report)
    
    # 2. Add edges & dynamic map-reduce branching
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", fan_out_research, ["researcher"])
    builder.add_edge("researcher", "reviewer")
    builder.add_conditional_edges("reviewer", route_after_review, ["writer", "researcher"])
    builder.add_edge("writer", END)
    
    checkpointer = MemorySaver() if enable_memory else None
    graph = builder.compile(checkpointer=checkpointer)
    
    return graph
