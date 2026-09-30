import re
import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import edge_tts
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm, OUTPUT_DIR

logger = logging.getLogger(__name__)

# Natural neural voices
VOICE_ALEX = "en-US-AndrewMultilingualNeural"   # Male host / Narrator
VOICE_SAM = "en-US-AvaMultilingualNeural"       # Female co-host

PODCAST_PROMPT = """You are an Executive Audio Producer creating an engaging NotebookLM-style podcast episode from a deep research report.

Your task is to write a dynamic, spoken dialogue between two intelligent hosts:
- **Alex**: Curious, articulates the big picture, frames strategic questions, and guides the discussion.
- **Sam**: Deep technical specialist who breaks down the architecture, specific metrics, benchmarks, trade-offs, and data.

Rules:
1. Format each spoken line strictly as:
Alex: [Alex's spoken sentence]
Sam: [Sam's spoken sentence]
2. Keep it conversational, engaging, and spoken-friendly (e.g. use natural contractions like "it's", "they've", avoid reading raw markdown or URL links).
3. Cover the core problem, the main breakthroughs, key metrics/statistics, and the future outlook.
4. Total length: 6 to 10 dialogue turns (around 200-350 words).
5. Output ONLY the dialogue lines. Do NOT include sound effect markers like [music fades] or introductory stage directions."""

BRIEFING_PROMPT = """You are a Principal Intelligence Broadcaster.
Write a clear, authoritative 2-minute executive spoken audio briefing based on the provided research report.
Cover the executive summary, primary data breakthroughs, key trade-offs, and strategic conclusions.
Write in a natural, spoken-word cadence suitable for text-to-speech narration (avoid reading raw URLs, markdown syntax, or tables).
Total length: 150-250 words."""

def generate_audio_script(report_content: str, style: str = "podcast") -> str:
    """Generates an engaging audio script or 2-host podcast conversation from research content."""
    llm = get_llm(temperature=0.4, max_tokens=800)
    system_prompt = PODCAST_PROMPT if style == "podcast" else BRIEFING_PROMPT
    
    prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"RESEARCH REPORT CONTENT:\n{report_content[:4000]}")
    ]
    
    try:
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception as e:
        logger.warning(f"Audio script LLM generation failed: {e}. Using fallback.")
        if style == "podcast":
            return (
                "Alex: Welcome to today's deep research summary. We're diving into the core findings of our investigation.\n"
                "Sam: That's right, Alex. The evidence reveals significant technological advancements and crucial trade-offs worth examining.\n"
                "Alex: Let's unpack the key takeaways and what they mean for the future of this field."
            )
        else:
            return (
                "Welcome to the executive research briefing. Today's findings demonstrate rapid advancements "
                "in architecture, performance metrics, and production readiness, accompanied by key engineering trade-offs."
            )

async def _synthesize_async(script: str, style: str = "podcast") -> bytes:
    """Asynchronously generates MP3 audio stream using edge-tts."""
    audio_chunks: List[bytes] = []
    
    if style == "podcast":
        lines = [line.strip() for line in script.split("\n") if line.strip()]
        for line in lines:
            if line.lower().startswith("alex:"):
                voice = VOICE_ALEX
                text = line[5:].strip()
            elif line.lower().startswith("sam:"):
                voice = VOICE_SAM
                text = line[4:].strip()
            else:
                voice = VOICE_ALEX
                text = line
                
            if text:
                comm = edge_tts.Communicate(text, voice)
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        audio_chunks.append(chunk["data"])
    else:
        comm = edge_tts.Communicate(script, VOICE_ALEX)
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
                
    return b"".join(audio_chunks)

def synthesize_audio(script: str, style: str = "podcast") -> bytes:
    """Synchronous wrapper to convert script into MP3 audio bytes."""
    try:
        return asyncio.run(_synthesize_async(script, style=style))
    except Exception as e:
        logger.error(f"Text-to-speech synthesis error: {e}")
        return b""

def create_audio_briefing(
    report_content: str,
    output_path: Optional[Path] = None,
    style: str = "podcast"
) -> Tuple[bytes, str]:
    """Generates an audio script and synthesizes full MP3 audio, optionally saving to disk."""
    logger.info(f"Generating audio briefing (style: {style})...")
    script = generate_audio_script(report_content, style=style)
    audio_bytes = synthesize_audio(script, style=style)
    
    if output_path and audio_bytes:
        try:
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"Saved audio briefing to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write audio file {output_path}: {e}")
            
    return audio_bytes, script
