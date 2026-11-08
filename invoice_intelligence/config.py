"""Runtime configuration, overridable through environment variables."""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PACKAGE_DIR / "knowledge_base"
CACHE_DIR = Path(os.environ.get("INVOICE_AI_CACHE_DIR", PACKAGE_DIR.parent / ".cache"))

GENERATION_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.environ.get("GEMINI_EMBEDDING_DIM", "768"))

# Corporate policy knobs used by the deterministic guardrails.
MAX_STANDARD_TERMS_DAYS = 30
APPROVAL_THRESHOLD_USD = 100_000


def get_api_key() -> str:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
