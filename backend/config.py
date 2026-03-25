import os

CHUNK_SIZE = 380
CHUNK_OVERLAP = 75
TOP_K = 4

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "documents")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemma-3-4b-it:free"

OLLAMA_MODEL = "phi3:mini"
OLLAMA_URL = "http://localhost:11434/api/chat"

PORT = 5000
DEBUG = False