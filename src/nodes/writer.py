import re
import logging
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm, OUTPUT_DIR
from src.state import ResearchState

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """You are a World-Class Principal Analyst and Technical Author.
Your mission is to synthesize all accumulated research notes and evidence into a definitive, publication-grade Executive Research Report in GitHub-flavored Markdown.

Report Standards & Format:
1. # Comprehensive Title
2. **Executive Summary**: High-impact synthesis of the core problem, current state, and critical breakthroughs (2-3 paragraphs).
3. **Key Insights & Takeaways**: Bullet points highlighting quantifiable metrics, strategic shifts, or core conclusions.
4. **In-Depth Analysis Sections**:
   - Write thorough, deeply technical, and structured subsections.
   - Use Markdown tables for comparisons (e.g., pros/cons, performance metrics, feature matrices).
   - Integrate in-text citations linking to sources, e.g., [Source Name](URL).
5. **Challenges, Trade-Offs & Future Outlook**: Nuanced evaluation of bottlenecks, limitations, and future trajectory.
6. **Sources & Bibliography**: Numbered list of all unique cited URLs and their titles.

Make the report exhaustive, elegant, authoritative, and fact-driven. Do NOT generate placeholders or truncated content.
"""

def synthesize_report(state: ResearchState) -> dict:
    """Writer node: Synthesizes all gathered evidence into a comprehensive Markdown report."""
    topic = state.get("topic", "")
    sections_data = state.get("sections_data", [])
    
    logger.info(f"Synthesizing final report across {len(sections_data)} research sections.")
    
    # Compile research notes and deduplicate sources
    evidence_blocks = []
    all_sources = {}
    
    for s in sections_data:
        title = s.get("section_title", "Research Findings")
        notes = s.get("notes", "")
        # Include high-density notes
        evidence_blocks.append(f"### {title}\n{notes[:1200]}\n")
        
        for src in s.get("sources", []):
            url = src.get("url")
            if url and url not in all_sources:
                all_sources[url] = src.get("title", url)
                
    compiled_evidence = "\n".join(evidence_blocks)
    
    # Format bibliography reference string
    sources_summary = "\n".join([f"- [{title}]({url})" for url, title in list(all_sources.items())[:10]])
    
    llm = get_llm(temperature=0.2, max_tokens=1000)
    
    prompt = [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(content=f"""Topic: {topic}

RESEARCH EVIDENCE:
{compiled_evidence}

SOURCES:
{sources_summary}

Synthesize a comprehensive, executive-level research report with sections, key findings, and references.""")
    ]
    
    report_content = ""
    try:
        resp = llm.invoke(prompt)
        report_content = resp.content.strip()
    except Exception as e:
        logger.warning(f"Writer LLM timed out or failed: {e}. Compiling structured report directly.")
        report_content = f"""# Deep Research Report: {topic}

## Executive Summary
This report provides an in-depth synthesis of {topic}, compiled from multi-agent web investigation and evidence validation.

## Key Findings & Section Analysis
{compiled_evidence}

## Sources & Bibliography
{sources_summary}
"""
    
    # Append auto-generated Mermaid architecture map
    from src.tools.exporter import generate_mermaid_diagram, markdown_to_pdf_bytes, markdown_to_docx_bytes
    mermaid_block = generate_mermaid_diagram(topic, sections_data)
    if "## Research Architecture & Thematic Map" not in report_content:
        report_content += f"\n\n## Research Architecture & Thematic Map\n\n{mermaid_block}\n"
    
    # Save report to output directory in Markdown, PDF, and DOCX formats
    slug = re.sub(r'[^a-zA-Z0-9_-]', '_', topic.lower()[:40]).strip('_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"research_{slug}_{timestamp}"
    
    md_path = OUTPUT_DIR / f"{base_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Research report saved to {md_path}")
    
    # Save PDF
    try:
        pdf_path = OUTPUT_DIR / f"{base_name}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(markdown_to_pdf_bytes(report_content))
        logger.info(f"PDF report saved to {pdf_path}")
    except Exception as e:
        logger.warning(f"Could not generate PDF: {e}")
        
    # Save DOCX
    try:
        docx_path = OUTPUT_DIR / f"{base_name}.docx"
        with open(docx_path, "wb") as f:
            f.write(markdown_to_docx_bytes(report_content))
        logger.info(f"DOCX report saved to {docx_path}")
    except Exception as e:
        logger.warning(f"Could not generate DOCX: {e}")
        
    # Generate 2-Host Audio Podcast Briefing (MP3)
    audio_path = None
    audio_script = None
    try:
        from src.tools.audio import create_audio_briefing
        mp3_path = OUTPUT_DIR / f"{base_name}_podcast.mp3"
        _, audio_script = create_audio_briefing(report_content, output_path=mp3_path, style="podcast")
        audio_path = str(mp3_path)
        logger.info(f"Podcast MP3 saved to {mp3_path}")
    except Exception as e:
        logger.warning(f"Could not generate audio podcast briefing: {e}")
        
    return {
        "final_report": report_content,
        "audio_script": audio_script,
        "audio_path": audio_path,
        "status_message": f"Report generated and saved to {md_path.name} (MD, PDF, DOCX, MP3)"
    }
