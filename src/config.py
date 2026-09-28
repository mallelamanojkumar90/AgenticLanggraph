import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
DEFAULT_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")
MAX_RESEARCH_ITERATIONS = int(os.getenv("MAX_RESEARCH_ITERATIONS", "1"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS_PER_QUERY", "4"))
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def get_llm(model: str = None, temperature: float = 0.2, max_tokens: int = 1500):
    """Initializes and returns a ChatNVIDIA instance."""
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    
    api_key = NVIDIA_API_KEY or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY is not set. Please create a .env file with your key: NVIDIA_API_KEY=nvapi-..."
        )
    
    return ChatNVIDIA(
        model=model or DEFAULT_MODEL,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=90,
    )
