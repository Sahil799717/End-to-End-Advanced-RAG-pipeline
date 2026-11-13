
import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env", override=True)

KNOWLEDGE_DIR = PROJECT_ROOT / "Knowledge_source"
DATABASE_DIR = PROJECT_ROOT / "database" / "chroma_db"
GOLDEN_DIR = PROJECT_ROOT / "goldens"
EVAL_DIR = PROJECT_ROOT / "evals"

PDF_PATH = KNOWLEDGE_DIR / "Temples.pdf"

# --------------------------------------------------
# API configuration
# --------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --------------------------------------------------
# Model configuration
# --------------------------------------------------

# Gemini: Query transformation and RAG generation
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-3.6-flash"
)

# Ollama: Local embeddings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "embeddinggemma"
)

# FlashRank: Document reranking
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "ms-marco-MiniLM-L-12-v2"
)

# Groq: DeepEval evaluation judge
GROQ_EVAL_MODEL = os.getenv(
    "GROQ_EVAL_MODEL",
    "openai/gpt-oss-120b"
)

# Prevent provider mismatch
if GROQ_EVAL_MODEL.lower().startswith("gemini"):
    raise ValueError(
        "Invalid GROQ_EVAL_MODEL. "
        "Gemini models cannot be called through the Groq client. "
        "Configure a Groq-supported model in .env."
    )

# --------------------------------------------------
# Vector database
# --------------------------------------------------

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "agentic-rag"
)

# --------------------------------------------------
# Retrieval configuration
# --------------------------------------------------

RETRIEVAL_K = int(
    os.getenv("RETRIEVAL_K", "5")
)

RERANK_TOP_N = int(
    os.getenv("RERANK_TOP_N", "3")
)

MAX_RETRIES = int(
    os.getenv("MAX_RETRIES", "1")
)

# --------------------------------------------------
# Chunking configuration
# --------------------------------------------------

CHUNK_BREAKPOINT_TYPE = os.getenv(
    "CHUNK_BREAKPOINT_TYPE",
    "percentile"
)

CHUNK_BREAKPOINT_AMOUNT = float(
    os.getenv("CHUNK_BREAKPOINT_AMOUNT", "95")
)

# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate_google_config():
    """Validate Google Gemini configuration."""

    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is missing. "
            "Add it to your .env file."
        )


def validate_groq_config():
    """Validate Groq configuration for evaluation."""

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to your .env file."
        )

    if not GROQ_EVAL_MODEL:
        raise ValueError(
            "GROQ_EVAL_MODEL is missing."
        )


def validate_project_config():
    """Validate general project configuration."""

    if RETRIEVAL_K < 1:
        raise ValueError(
            "RETRIEVAL_K must be at least 1."
        )

    if RERANK_TOP_N < 1:
        raise ValueError(
            "RERANK_TOP_N must be at least 1."
        )

    if MAX_RETRIES < 0:
        raise ValueError(
            "MAX_RETRIES cannot be negative."
        )

    if not 0 <= CHUNK_BREAKPOINT_AMOUNT <= 100:
        raise ValueError(
            "CHUNK_BREAKPOINT_AMOUNT must be between 0 and 100."
        )

    KNOWLEDGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# --------------------------------------------------
# Backward-compatible names used by rag_backend.py
# --------------------------------------------------

BREAKPOINT_THRESHOLD_TYPE = CHUNK_BREAKPOINT_TYPE
BREAKPOINT_THRESHOLD_AMOUNT = CHUNK_BREAKPOINT_AMOUNT
