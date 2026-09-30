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
from src.state import SubTopicPlan
from src.tools.exporter import markdown_to_pdf_bytes, markdown_to_docx_bytes

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
    
    enable_hitl = st.checkbox(
        "⏸️ Human-in-the-Loop Plan Review",
        value=False,
        help="Pause after Planner to inspect research subtopics before web research begins"
    )
    
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
            
        b1, b2 = st.columns(2)
        with b1:
            st.download_button(
                label="⬇️ Markdown",
                data=content,
                file_name=selected_report,
                mime="text/markdown",
                key="side_dl_md",
                use_container_width=True
            )
        with b2:
            try:
                pdf_data = markdown_to_pdf_bytes(content)
                st.download_button(
                    label="⬇️ PDF",
                    data=pdf_data,
                    file_name=selected_report.replace(".md", ".pdf"),
                    mime="application/pdf",
                    key="side_dl_pdf",
                    use_container_width=True
                )
            except Exception:
                pass
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

def stream_and_display_graph(app, stream_input, config, status_box, plan_placeholder=None):
    """Streams LangGraph execution events and dynamically updates Streamlit UI components."""
    research_placeholder = st.container()
    review_placeholder = st.empty()
    verifier_placeholder = st.empty()
    
    for event in app.stream(stream_input, config=config, stream_mode="updates"):
        for node_name, node_update in event.items():
            if node_name == "planner":
                sections = node_update.get("sections", [])
                status_box.update(label=f"🔍 **Stage 2:** Fanning out {len(sections)} parallel researcher workers...")
                
                if plan_placeholder:
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
                    num_scraped = sum(1 for src in latest.get('sources', []) if src.get('scraped'))
                    with research_placeholder:
                        st.info(f"✅ **Completed Section:** {latest.get('section_title')} — {len(latest.get('sources', []))} sources ({num_scraped} full web pages scraped)")
                        
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
                status_box.update(label="🛡️ **Stage 4:** Report written! Verifying factual citations & links...")
                
            elif node_name == "verifier":
                score = node_update.get("verification_score", 100.0)
                feedback = node_update.get("verification_feedback", "")
                verifier_placeholder.success(f"🛡️ **Fact-Check & Citation Audit ({score:.1f}%):** {feedback}")
                status_box.update(label="✅ **Research, Synthesis & Fact-Checking Complete!**", state="complete")


start_research = st.button("🚀 Start Deep Research", type="primary", use_container_width=True)

if start_research:
    if not os.getenv("NVIDIA_API_KEY"):
        st.error("⚠️ NVIDIA_API_KEY is not configured in your .env file.")
        st.stop()
    if not topic.strip():
        st.warning("⚠️ Please provide a research topic to proceed.")
        st.stop()

    # Reset any lingering HITL review state
    st.session_state["hitl_paused"] = False
    st.session_state["hitl_thread_id"] = None
    st.session_state["hitl_sections"] = []

    st.write("---")
    
    # Real-time Execution Flow Container
    with st.status("Initializing LangGraph Multi-Agent Workflow...", expanded=True) as status_box:
        app = build_research_graph(
            enable_memory=True,
            persistent=True,
            interrupt_before=["researcher"] if enable_hitl else None
        )
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
            "verification_score": 100.0,
            "verification_feedback": "",
            "verified_sources_count": 0,
            "flagged_sources_count": 0,
            "status_message": "Initializing..."
        }
        
        status_box.update(label="🧠 **Stage 1:** Planner Node decomposing research strategy...")
        plan_placeholder = st.empty()
        
        stream_and_display_graph(app, initial_state, config, status_box, plan_placeholder=plan_placeholder)

        state_snapshot = app.get_state(config)
        
        # Check if execution paused at an interrupt (e.g. Human-in-the-Loop review)
        if state_snapshot.next:
            status_box.update(label="⏸️ **Stage 1 Complete:** Research strategy planned. Awaiting review below...", state="complete")
            st.session_state["hitl_paused"] = True
            st.session_state["hitl_thread_id"] = thread_id
            st.session_state["hitl_topic"] = topic
            st.session_state["hitl_plan_summary"] = state_snapshot.values.get("plan_summary", "")
            raw_sections = state_snapshot.values.get("sections", [])
            st.session_state["hitl_sections"] = [
                {
                    "title": getattr(s, "title", s.get("title", "") if isinstance(s, dict) else ""),
                    "description": getattr(s, "description", s.get("description", "") if isinstance(s, dict) else ""),
                    "queries": list(getattr(s, "queries", s.get("queries", []) if isinstance(s, dict) else []))
                }
                for s in raw_sections
            ]
            st.session_state["latest_research"] = None
            st.rerun()
        else:
            st.session_state["latest_research"] = state_snapshot.values
            st.session_state["chat_messages"] = []

# --- HUMAN-IN-THE-LOOP PLAN REVIEW CONTAINER ---
if st.session_state.get("hitl_paused") and st.session_state.get("hitl_thread_id"):
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #1e293b; border-left: 5px solid #3b82f6; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px;">
        <h3 style="margin: 0 0 8px 0; color: #60a5fa;">⏸️ Human-in-the-Loop: Review & Approve Research Plan</h3>
        <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem;">
            The <b>Planner Agent</b> has structured the research strategy below. 
            You can inspect, refine, add, or remove subtopics and targeted search queries before fanning out autonomous web researcher workers.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info(f"📋 **Strategy Summary:** {st.session_state.get('hitl_plan_summary', '')}")
    
    current_sections = st.session_state.get("hitl_sections", [])
    
    col_hdr, col_add = st.columns([3, 1])
    with col_hdr:
        st.subheader(f"Proposed Research Subtopics ({len(current_sections)})")
    with col_add:
        if st.button("➕ Add Another Subtopic", use_container_width=True):
            current_sections.append({
                "title": f"Subtopic {len(current_sections) + 1}",
                "description": f"Targeted exploration of {st.session_state.get('hitl_topic', 'the topic')}",
                "queries": [f"{st.session_state.get('hitl_topic', '')} analysis"]
            })
            st.session_state["hitl_sections"] = current_sections
            st.rerun()

    for idx, s in enumerate(current_sections):
        with st.expander(f"📌 Subtopic {idx+1}: {s.get('title')}", expanded=True):
            c_title, c_del = st.columns([5, 1])
            with c_title:
                t = st.text_input("Subtopic Title", value=s.get("title", ""), key=f"hitl_title_{idx}")
            with c_del:
                st.write("")
                st.write("")
                if len(current_sections) > 1:
                    if st.button("🗑️ Remove", key=f"hitl_del_{idx}", use_container_width=True):
                        current_sections.pop(idx)
                        st.session_state["hitl_sections"] = current_sections
                        st.rerun()
            d = st.text_input("Description / Scope", value=s.get("description", ""), key=f"hitl_desc_{idx}")
            q_str = st.text_area(
                "Search Queries (one query per line)",
                value="\n".join(s.get("queries", [])),
                key=f"hitl_queries_{idx}",
                help="These queries will be executed concurrently across DuckDuckGo"
            )

    st.write("")
    btn_approve, btn_cancel = st.columns([3, 1])
    with btn_approve:
        approve_action = st.button("🚀 Approve Plan & Fan Out Deep Research", type="primary", use_container_width=True)
    with btn_cancel:
        cancel_action = st.button("❌ Discard Plan", use_container_width=True)

    if cancel_action:
        st.session_state["hitl_paused"] = False
        st.session_state["hitl_thread_id"] = None
        st.session_state["hitl_sections"] = []
        st.rerun()

    if approve_action:
        updated_sections = []
        for idx in range(len(current_sections)):
            t = st.session_state.get(f"hitl_title_{idx}", "").strip()
            d = st.session_state.get(f"hitl_desc_{idx}", "").strip()
            q_raw = st.session_state.get(f"hitl_queries_{idx}", "").strip()
            q_lines = [line.strip() for line in q_raw.splitlines() if line.strip()]
            if t:
                updated_sections.append(SubTopicPlan(title=t, description=d or t, queries=q_lines or [t]))
        
        if not updated_sections:
            st.error("⚠️ At least one valid research subtopic is required to proceed.")
            st.stop()

        with st.status("🚀 Resuming Deep Research & Synthesis...", expanded=True) as resume_box:
            app = build_research_graph(enable_memory=True, persistent=True)
            config = {"configurable": {"thread_id": st.session_state["hitl_thread_id"]}}
            
            # Update the checkpoint state with human-approved sections as_node="planner"
            app.update_state(config, {"sections": updated_sections}, as_node="planner")
            
            # Resume graph execution (input=None tells LangGraph to continue from checkpoint)
            stream_and_display_graph(app, None, config, resume_box)
            
            final_snapshot = app.get_state(config)
            st.session_state["latest_research"] = final_snapshot.values
            st.session_state["hitl_paused"] = False
            st.session_state["hitl_thread_id"] = None
            st.session_state["hitl_sections"] = []
            st.session_state["chat_messages"] = []
            st.rerun()

# --- DISPLAY RESULTS TABS ---
if st.session_state.get("latest_research") and st.session_state["latest_research"].get("final_report"):
    res = st.session_state["latest_research"]
    final_report = res.get("final_report", "")
    sections_data = res.get("sections_data", [])
    v_score = res.get("verification_score", 100.0)
    
    st.divider()
    
    # Quick metrics header
    total_sources = sum(len(s.get("sources", [])) for s in sections_data)
    total_scraped = sum(sum(1 for src in s.get("sources", []) if src.get("scraped")) for s in sections_data)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sections Researched", len(sections_data))
    m2.metric("Total Sources", f"{total_sources} ({total_scraped} scraped)")
    m3.metric("Review Iterations", res.get("critique_iteration", 1))
    m4.metric("Citation Grounding", f"{v_score:.1f}%")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📑 Executive Report",
        "🔬 Sub-Section Notes",
        "🌐 Sources & Citations",
        "⚙️ Graph State Inspector",
        "💬 Chat with Report"
    ])
    
    with tab1:
        d1, d2, d3 = st.columns(3)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        with d1:
            st.download_button(
                label="⬇️ Download Markdown (.md)",
                data=final_report,
                file_name=f"research_{timestamp_str}.md",
                mime="text/markdown",
                key="main_dl_md",
                use_container_width=True
            )
        with d2:
            try:
                pdf_data = markdown_to_pdf_bytes(final_report)
                st.download_button(
                    label="⬇️ Download PDF (.pdf)",
                    data=pdf_data,
                    file_name=f"research_{timestamp_str}.pdf",
                    mime="application/pdf",
                    key="main_dl_pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.caption(f"PDF error: {e}")
        with d3:
            try:
                docx_data = markdown_to_docx_bytes(final_report)
                st.download_button(
                    label="⬇️ Download Word (.docx)",
                    data=docx_data,
                    file_name=f"research_{timestamp_str}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="main_dl_docx",
                    use_container_width=True
                )
            except Exception as e:
                st.caption(f"DOCX error: {e}")
                
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
            badge = "🟢 **[Full Page Scraped]**" if src.get("scraped") else "⚪ **[Search Snippet]**"
            st.markdown(f"- {badge} [{src.get('title', url)}]({url})")
            excerpt = src.get('content_preview', src.get('snippet', ''))[:300]
            st.caption(f"Content Sample: {excerpt}...")
            
    with tab4:
        st.json({
            "topic": res.get("topic"),
            "plan_summary": res.get("plan_summary"),
            "critique_iteration": res.get("critique_iteration"),
            "is_sufficient": res.get("is_sufficient"),
            "review_feedback": res.get("review_feedback"),
            "verification_score": res.get("verification_score"),
            "verification_feedback": res.get("verification_feedback"),
            "sections_count": len(sections_data)
        })
        
    with tab5:
        st.subheader("💬 Ask Follow-up Questions About This Research")
        st.caption("Converse with the synthesized evidence directly without re-running the full research pipeline.")
        
        for msg in st.session_state.get("chat_messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if user_prompt := st.chat_input("Ask a follow-up question about this topic..."):
            st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)
                
            with st.chat_message("assistant"):
                from langchain_core.messages import SystemMessage, HumanMessage
                from src.config import get_llm
                llm = get_llm(temperature=0.2)
                sys_msg = (
                    "You are a Senior Research Analyst. Answer the user's question accurately using only "
                    "the provided verified research report and accumulated evidence. Cite section headings "
                    "or source URLs where applicable.\n\n"
                    f"RESEARCH REPORT:\n{final_report}"
                )
                try:
                    resp = llm.invoke([
                        SystemMessage(content=sys_msg),
                        HumanMessage(content=user_prompt)
                    ])
                    ans = resp.content
                except Exception as e:
                    ans = f"Error generating answer: {e}"
                st.markdown(ans)
                st.session_state["chat_messages"].append({"role": "assistant", "content": ans})
