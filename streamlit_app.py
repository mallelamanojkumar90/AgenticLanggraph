import os
import sys
import uuid
from pathlib import Path
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from src.config import NVIDIA_API_KEY, DEFAULT_MODEL, OUTPUT_DIR
from src.graph import build_research_graph

st.set_page_config(
    page_title="Deep Research Agent | LangGraph + NVIDIA NIM",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .stDownloadButton button {
        background-color: #059669;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Nvidia_logo.svg", width=130)
    st.title("Settings & Controls")
    
    # API Key Status (loaded securely from .env, never displayed)
    if os.getenv("NVIDIA_API_KEY"):
        st.success("NVIDIA NIM Connected", icon="✅")
    else:
        st.error("NVIDIA_API_KEY missing in .env", icon="⚠️")

    # Model Selection
    available_models = [
        "meta/llama-3.2-11b-vision-instruct",
        "mistralai/mistral-nemotron",
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "meta/muse-glimmer-30b"
    ]
    selected_model = st.selectbox(
        "Active LLM",
        options=available_models,
        index=0,
        help="Model hosted on NVIDIA NIM endpoints"
    )
    os.environ["NVIDIA_MODEL"] = selected_model
    
    st.divider()
    
    # Hyperparameters
    max_iters = st.slider(
        "Max Critique Iterations",
        min_value=1,
        max_value=3,
        value=1,
        help="Number of times the reviewer node can loop back for gap research"
    )
    
    max_search_results = st.slider(
        "Web Sources per Query",
        min_value=2,
        max_value=5,
        value=3,
        help="Number of DuckDuckGo results retrieved per query"
    )
    os.environ["MAX_SEARCH_RESULTS_PER_QUERY"] = str(max_search_results)
    
    st.divider()
    
    # Past Reports Archive
    st.subheader("📚 Saved Reports Archive")
    saved_reports = sorted(list(OUTPUT_DIR.glob("*.md")), key=os.path.getmtime, reverse=True)
    if saved_reports:
        selected_report = st.selectbox(
            "Browse past research:",
            options=[r.name for r in saved_reports],
            index=0
        )
        report_path = OUTPUT_DIR / selected_report
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.download_button(
            label="⬇️ Download Markdown",
            data=content,
            file_name=selected_report,
            mime="text/markdown",
            key="side_dl"
        )
    else:
        st.caption("No reports generated yet.")

# --- MAIN INTERFACE ---
st.markdown('<div class="main-header">🔍 Autonomous Deep Research Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-agent research and report synthesis orchestrated with <b>LangGraph</b> and powered by <b>NVIDIA NIM</b></div>', unsafe_allow_html=True)

# Sample Topics
st.write("**Quick Start Topics:**")
sample_cols = st.columns(4)
SAMPLE_TOPICS = [
    "State of Reasoning LLMs in 2026: Architectures & Test-Time Compute",
    "Agentic AI Frameworks: LangGraph vs AutoGen vs CrewAI",
    "Solid-State Batteries: Commercial Timelines & Physics Bottlenecks",
    "AI Inference Chips: NVIDIA GPUs vs Custom Silicon (TPU, Trainium)"
]

for idx, (col, s_topic) in enumerate(zip(sample_cols, SAMPLE_TOPICS)):
    with col:
        if st.button(f"📌 {s_topic.split(':')[0]}", key=f"sample_{idx}", use_container_width=True):
            st.session_state["topic_input"] = s_topic

# Research Topic Input
topic = st.text_area(
    "**Enter Research Topic or Question:**",
    value=st.session_state.get("topic_input", ""),
    placeholder="e.g. In-depth analysis of Quantum Computing hardware roadmaps for 2026-2028...",
    height=80
)

start_research = st.button("🚀 Start Deep Research", type="primary", use_container_width=True)

if start_research:
    if not os.getenv("NVIDIA_API_KEY"):
        st.error("⚠️ NVIDIA_API_KEY is not configured in your .env file.")
        st.stop()
    if not topic.strip():
        st.warning("⚠️ Please provide a research topic to proceed.")
        st.stop()

    st.write("---")
    
    # Real-time Execution Flow Container
    with st.status("Initializing LangGraph Multi-Agent Workflow...", expanded=True) as status_box:
        app = build_research_graph(enable_memory=True)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "topic": topic,
            "plan_summary": "",
            "sections": [],
            "sections_data": [],
            "critique_iteration": 0,
            "max_iterations": max_iters,
            "is_sufficient": False,
            "review_feedback": "",
            "gap_queries": [],
            "final_report": "",
            "status_message": "Initializing..."
        }
        
        status_box.update(label="🧠 **Stage 1:** Planner Node decomposing research strategy...")
        plan_placeholder = st.empty()
        research_placeholder = st.container()
        review_placeholder = st.empty()
        
        for event in app.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_update in event.items():
                if node_name == "planner":
                    sections = node_update.get("sections", [])
                    status_box.update(label=f"🔍 **Stage 2:** Fanning out {len(sections)} parallel researcher workers...")
                    
                    with plan_placeholder.container():
                        st.success(f"**Research Plan Established:** {node_update.get('plan_summary', '')}")
                        p_cols = st.columns(len(sections))
                        for col, s in zip(p_cols, sections):
                            with col:
                                st.markdown(f"**{s.title}**")
                                st.caption(s.description)
                                for q in s.queries:
                                    st.code(q, language="text")
                                    
                elif node_name == "researcher":
                    data = node_update.get("sections_data", [])
                    if data:
                        latest = data[-1]
                        with research_placeholder:
                            st.info(f"✅ **Completed Section:** {latest.get('section_title')} — Cited {len(latest.get('sources', []))} sources")
                            
                elif node_name == "reviewer":
                    is_suff = node_update.get("is_sufficient", True)
                    critique_iter = node_update.get("critique_iteration", 1)
                    if is_suff:
                        status_box.update(label="📝 **Stage 3:** Research approved! Synthesizing final report...")
                        review_placeholder.success(f"⚖️ **Reviewer Node (Pass {critique_iter}):** {node_update.get('review_feedback')}")
                    else:
                        status_box.update(label=f"🔄 **Stage 2 (Reflection Loop {critique_iter}):** Investigating identified research gaps...")
                        review_placeholder.warning(f"⚖️ **Reviewer Gaps Detected:** {node_update.get('review_feedback')}")
                        
                elif node_name == "writer":
                    status_box.update(label="✅ **Research Completed & Report Generated!**", state="complete")

        # Snapshot final state
        state_snapshot = app.get_state(config)
        st.session_state["latest_research"] = state_snapshot.values

# --- DISPLAY RESULTS TABS ---
if "latest_research" in st.session_state:
    res = st.session_state["latest_research"]
    final_report = res.get("final_report", "")
    sections_data = res.get("sections_data", [])
    
    st.divider()
    
    # Quick metrics header
    total_sources = sum(len(s.get("sources", [])) for s in sections_data)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sections Researched", len(sections_data))
    m2.metric("Total Sources Cited", total_sources)
    m3.metric("Review Iterations", res.get("critique_iteration", 1))
    m4.metric("LLM Provider", "NVIDIA NIM (Llama 3.2)")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📑 Executive Report",
        "🔬 Sub-Section Notes",
        "🌐 Sources & Citations",
        "⚙️ Graph State Inspector"
    ])
    
    with tab1:
        st.download_button(
            label="⬇️ Download Full Markdown Report",
            data=final_report,
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            key="main_dl"
        )
        st.markdown(final_report)
        
    with tab2:
        for idx, s in enumerate(sections_data, 1):
            with st.expander(f"📌 Section {idx}: {s.get('section_title')}", expanded=False):
                st.write("**Search Queries Executed:**")
                st.write(s.get("queries_run", []))
                st.markdown("**Synthesis Notes:**")
                st.markdown(s.get("notes", ""))
                
    with tab3:
        all_unique_sources = {}
        for s in sections_data:
            for src in s.get("sources", []):
                url = src.get("url")
                if url and url not in all_unique_sources:
                    all_unique_sources[url] = src
                    
        st.write(f"Total Unique Sources Found: **{len(all_unique_sources)}**")
        for url, src in all_unique_sources.items():
            st.markdown(f"- **[{src.get('title', url)}]({url})**")
            st.caption(f"Excerpt: {src.get('snippet', '')[:250]}...")
            
    with tab4:
        st.json({
            "topic": res.get("topic"),
            "plan_summary": res.get("plan_summary"),
            "critique_iteration": res.get("critique_iteration"),
            "is_sufficient": res.get("is_sufficient"),
            "review_feedback": res.get("review_feedback"),
            "sections_count": len(sections_data)
        })
