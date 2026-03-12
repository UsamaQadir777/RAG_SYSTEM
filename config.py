"""
Central configuration for the RAG system.
All tunable parameters in one place.
"""
import os

# --- Document Settings ---
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
SUPPORTED_EXTENSIONS = [".pdf", ".txt"]

# --- Chunking Settings ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# --- Embedding Settings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Retrieval Settings ---
TOP_K = int(os.getenv("TOP_K", "3"))

# --- LLM Settings ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
USE_OPENAI = bool(OPENAI_API_KEY)

# --- Logging ---
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Confidence Threshold ---
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "1.5"))