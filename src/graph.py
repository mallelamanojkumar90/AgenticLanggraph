import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Union
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import OUTPUT_DIR
from src.state import ResearchState, SubTopicTask
from src.nodes.planner import plan_research
from src.nodes.researcher import conduct_research
from src.nodes.reviewer import review_research
from src.nodes.writer import synthesize_report
from src.nodes.verifier import verify_report

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

def build_research_graph(
    enable_memory: bool = True,
    persistent: bool = True,
    db_path: Optional[Union[str, Path]] = None,
    interrupt_before: Optional[List[str]] = None
):
    """Builds and compiles the complete Deep Research LangGraph workflow with verification and persistent checkpoints."""
    builder = StateGraph(ResearchState)
    
    # 1. Add core nodes
    builder.add_node("planner", plan_research)
    builder.add_node("researcher", conduct_research)
    builder.add_node("reviewer", review_research)
    builder.add_node("writer", synthesize_report)
    builder.add_node("verifier", verify_report)
    
    # 2. Add edges & dynamic map-reduce branching
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", fan_out_research, ["researcher"])
    builder.add_edge("researcher", "reviewer")
    builder.add_conditional_edges("reviewer", route_after_review, ["writer", "researcher"])
    builder.add_edge("writer", "verifier")
    builder.add_edge("verifier", END)
    
    checkpointer = None
    if enable_memory:
        if persistent:
            target_db = Path(db_path) if db_path else (OUTPUT_DIR / "checkpoints.db")
            conn = sqlite3.connect(str(target_db), check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            checkpointer.setup()
        else:
            checkpointer = MemorySaver()
            
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)

