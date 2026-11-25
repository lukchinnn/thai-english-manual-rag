import os
from dotenv import load_dotenv
load_dotenv()

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma")
LLM_BACKEND    = os.getenv("LLM_BACKEND", "ollama")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
MONGODB_URI    = os.getenv("MONGODB_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TOP_K = 5