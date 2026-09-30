import os
import sys
import uuid
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

# Ensure root dir is in sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.config import NVIDIA_API_KEY, DEFAULT_MODEL, OUTPUT_DIR
from src.graph import build_research_graph

console = Console(highlight=False)

SAMPLE_TOPICS = [
    "State of Reasoning LLMs in 2026: Architectures, RLVR, Test-Time Compute, and Scaling Laws",
    "Agentic AI Frameworks Comparison: LangGraph vs AutoGen vs CrewAI vs Semantic Kernel",
    "Solid-State Batteries: Commercialization Timelines, Major Competitors, and Physics Bottlenecks",
    "GPU Shortage & Cloud Infrastructure: Custom Silicon (Google TPU, AWS Trainium) vs NVIDIA Dominance"
]

def print_banner():
    banner_text = """
    +=====================================================================+
    |                    AGENTIC DEEP RESEARCH SYSTEM                     |
    |             Powered by LangGraph & NVIDIA NIM / Llama-3.3           |
    +=====================================================================+
    """
    console.print(banner_text, style="bold cyan")

def run_research(topic: str, max_iterations: int = 1, doc_paths: list = None):
    if not os.getenv("NVIDIA_API_KEY"):
        console.print(
            Panel(
                "[bold red]NVIDIA_API_KEY is not set![/bold red]\n\n"
                "Please create a [bold yellow].env[/bold yellow] file in the project root with:\n"
                "[bold green]NVIDIA_API_KEY=nvapi-your-key[/bold green]\n"
                "[bold green]NVIDIA_MODEL=meta/llama-3.3-70b-instruct[/bold green]\n\n"
                "You can get a free API key at [link=https://build.nvidia.com/]https://build.nvidia.com/[/link]",
                title="Configuration Error",
                border_style="red"
            )
        )
        return

    # Ingest private documents if provided (Hybrid RAG)
    private_chunks = []
    if doc_paths:
        from src.tools.document_loader import extract_text_from_path, chunk_document
        for path_str in doc_paths:
            p = Path(path_str)
            if p.exists():
                text = extract_text_from_path(p)
                chunks = chunk_document(text, p.name)
                private_chunks.extend(chunks)
                console.print(f"[bold green][+] Ingested Internal Doc:[/bold green] {p.name} ({len(chunks)} chunks)")
            else:
                console.print(f"[bold yellow][!] Warning: Document path not found:[/bold yellow] {path_str}")

    console.print(Panel(f"[bold cyan]Research Topic:[/bold cyan] {topic}\n[bold cyan]Model:[/bold cyan] {DEFAULT_MODEL}\n[bold cyan]Hybrid RAG:[/bold cyan] {len(private_chunks)} internal chunks loaded", title="Starting Deep Research Session", border_style="blue"))
    
    app = build_research_graph(enable_memory=True, persistent=True)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "topic": topic,
        "plan_summary": "",
        "sections": [],
        "sections_data": [],
        "critique_iteration": 0,
        "max_iterations": max_iterations,
        "is_sufficient": False,
        "review_feedback": "",
        "gap_queries": [],
        "final_report": "",
        "verification_score": 100.0,
        "verification_feedback": "",
        "verified_sources_count": 0,
        "flagged_sources_count": 0,
        "private_docs_context": private_chunks,
        "audio_script": None,
        "audio_path": None,
        "status_message": "Initializing research graph..."
    }
    
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold green]{task.description}"),
        console=console,
        transient=False
    ) as progress:
        task_id = progress.add_task("Decomposing research plan with Planner node...", total=None)
        
        final_state = None
        for event in app.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_update in event.items():
                if node_name == "planner":
                    progress.update(task_id, description="Planner completed. Fanning out parallel researcher workers...")
                    sections = node_update.get("sections", [])
                    console.print(f"\n[bold green][+] Planned {len(sections)} Research Subtopics:[/bold green]")
                    for idx, s in enumerate(sections, 1):
                        console.print(f"  {idx}. [bold]{s.title}[/bold] ({len(s.queries)} search queries)")
                        
                elif node_name == "researcher":
                    data = node_update.get("sections_data", [])
                    if data:
                        latest_section = data[-1]
                        num_scraped = sum(1 for src in latest_section.get("sources", []) if src.get("scraped"))
                        progress.update(
                            task_id,
                            description=f"Synthesized notes for: '{latest_section.get('section_title')}'"
                        )
                        console.print(f"  [cyan][+] Researched:[/cyan] {latest_section.get('section_title')} ([dim]{len(latest_section.get('sources', []))} sources, {num_scraped} deep scraped[/dim])")
                        
                elif node_name == "reviewer":
                    is_suff = node_update.get("is_sufficient", True)
                    iteration = node_update.get("critique_iteration", 1)
                    if is_suff:
                        progress.update(task_id, description=f"Critic approved findings (Pass {iteration}). Synthesizing final report...")
                        console.print(f"\n[bold green][+] Research Verified by Reviewer Node[/bold green]: {node_update.get('review_feedback', 'Comprehensive.')}")
                    else:
                        progress.update(task_id, description=f"Critic identified gaps (Iteration {iteration}). Launching follow-up search...")
                        console.print(f"\n[bold yellow][!] Reviewer Feedback:[/bold yellow] {node_update.get('review_feedback')}")
                        
                elif node_name == "writer":
                    progress.update(task_id, description="Report synthesized! Auditing citations with Verifier node...")
                    console.print("\n[bold green][+] Executive Report Synthesized (MD, PDF, DOCX generated)[/bold green]")
                    
                elif node_name == "verifier":
                    progress.update(task_id, description="Fact-checking complete!")
                    v_score = node_update.get("verification_score", 100.0)
                    v_feedback = node_update.get("verification_feedback", "")
                    color = "green" if v_score >= 80 else ("yellow" if v_score >= 60 else "red")
                    console.print(f"[bold {color}][+] Citation Audit:[/bold {color}] {v_feedback}")
                    final_state = node_update
                    
        progress.stop()

    # Get latest snapshot from checkpointer
    state_snapshot = app.get_state(config)
    final_report = state_snapshot.values.get("final_report", "")
    sections_data = state_snapshot.values.get("sections_data", [])
    v_score = state_snapshot.values.get("verification_score", 100.0)
    v_feedback = state_snapshot.values.get("verification_feedback", "")
    
    # Summary table
    table = Table(title="Research Summary", show_header=True, header_style="bold magenta")
    table.add_column("Subtopic", style="dim")
    table.add_column("Queries Run", justify="right")
    table.add_column("Total Sources", justify="right")
    table.add_column("Deep Scraped", justify="right")
    
    total_sources = 0
    total_scraped = 0
    for s in sections_data:
        sources = s.get("sources", [])
        num_sources = len(sources)
        num_scraped = sum(1 for src in sources if src.get("scraped"))
        total_sources += num_sources
        total_scraped += num_scraped
        table.add_row(
            s.get("section_title", "N/A"),
            str(len(s.get("queries_run", []))),
            str(num_sources),
            f"[green]{num_scraped}[/green]"
        )
    console.print(table)
    console.print(f"[bold green]Total Sources Analyzed:[/bold green] {total_sources} ([cyan]{total_scraped} full pages scraped[/cyan])")
    
    score_color = "green" if v_score >= 80 else ("yellow" if v_score >= 60 else "red")
    console.print(f"[bold {score_color}]Citation Grounding Score:[/bold {score_color}] {v_score:.1f}% ({v_feedback})\n")
    
    # Display preview
    console.print(Panel(Markdown(final_report[:1200] + "\n\n*(Truncated in terminal preview -- see full report in output directory)*"), title="[Report Preview]", border_style="green"))
    console.print(f"[bold yellow]Exports Generated:[/bold yellow] Markdown (.md), Adobe PDF (.pdf), Word (.docx), Audio Podcast (.mp3) in [link=file://{OUTPUT_DIR.resolve()}]{OUTPUT_DIR.resolve()}[/link]\n")

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="Autonomous Deep Research Agent with LangGraph & NVIDIA")
    parser.add_argument("--topic", type=str, help="Research topic to investigate")
    parser.add_argument("--max-iters", type=int, default=1, help="Maximum critique reflection iterations (default: 1)")
    parser.add_argument("--docs", nargs="*", help="Optional path(s) to internal documents to ingest into research context (PDF, DOCX, TXT, MD)")
    args = parser.parse_args()
    
    if args.topic:
        run_research(args.topic, max_iterations=args.max_iters, doc_paths=args.docs)
    else:
        console.print("[bold]Choose an option:[/bold]")
        console.print("  [cyan]0.[/cyan] Enter your own custom research topic")
        for i, sample in enumerate(SAMPLE_TOPICS, 1):
            console.print(f"  [cyan]{i}.[/cyan] {sample}")
            
        choice = console.input("\n[bold yellow]Select [0-4]: [/bold yellow]").strip()
        
        if choice in ["1", "2", "3", "4"]:
            topic = SAMPLE_TOPICS[int(choice) - 1]
        else:
            topic = console.input("\n[bold yellow]Enter your research topic: [/bold yellow]").strip()
            
        if not topic:
            console.print("[red]Topic cannot be empty. Exiting.[/red]")
            return
            
        run_research(topic, max_iterations=args.max_iters, doc_paths=args.docs)

if __name__ == "__main__":
    main()
