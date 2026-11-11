
"""Configuration partagée pour Olfactive Studio."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Base vectorielle ChromaDB
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "aromes")

# Documents et découpage
STORY_PATH = Path(os.getenv("STORY_PATH", str(BASE_DIR / "story.txt")))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP_PARAGRAPHS = int(
    os.getenv("CHUNK_OVERLAP_PARAGRAPHS", "1")
)

# Recherche
TOP_K = int(os.getenv("TOP_K", "4"))
