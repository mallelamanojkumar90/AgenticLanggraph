import operator
from typing import Annotated, Dict, List, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# Pydantic models for structured generation
class SubTopicPlan(BaseModel):
    title: str = Field(description="Title or theme of this research sub-section")
    description: str = Field(description="What questions or aspects this sub-section needs to answer")
    queries: List[str] = Field(description="2-3 targeted web search queries to gather comprehensive data")

class ResearchPlanOutput(BaseModel):
    summary: str = Field(description="Brief overview of the planned research strategy")
    sections: List[SubTopicPlan] = Field(description="List of 3-5 distinct sub-topics to investigate in parallel")

class ReviewCritiqueOutput(BaseModel):
    is_sufficient: bool = Field(description="True if the collected research is deep, balanced, and complete. False if critical gaps exist.")
    feedback: str = Field(description="Constructive critique of what is covered vs what is still missing")
    gap_queries: List[SubTopicPlan] = Field(
        default_factory=list,
        description="1-2 targeted follow-up sub-sections with queries if is_sufficient is False. Empty if True."
    )

# LangGraph State Definitions
class SectionData(TypedDict):
    section_title: str
    queries_run: List[str]
    notes: str
    sources: List[Dict[str, str]]

class SubTopicTask(TypedDict):
    """Payload passed to an individual parallel researcher node via Send()."""
    section_title: str
    description: str
    queries: List[str]
    private_docs: Optional[List[Dict[str, Any]]]

class ResearchState(TypedDict):
    """The master state of the Deep Research graph."""
    topic: str
    plan_summary: str
    sections: List[SubTopicPlan]
    # Annotated with operator.add so parallel researcher results automatically merge into this list
    sections_data: Annotated[List[SectionData], operator.add]
    critique_iteration: int
    max_iterations: int
    is_sufficient: bool
    review_feedback: str
    gap_queries: List[SubTopicPlan]
    final_report: str
    verification_score: float
    verification_feedback: str
    verified_sources_count: int
    flagged_sources_count: int
    private_docs_context: Optional[List[Dict[str, Any]]]
    audio_script: Optional[str]
    audio_path: Optional[str]
    status_message: str
