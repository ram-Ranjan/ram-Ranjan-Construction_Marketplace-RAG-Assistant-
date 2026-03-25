import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import PORT, DEBUG, OPENROUTER_MODEL
from backend.rag_pipeline import RAGEngine

app = Flask(__name__)
CORS(app)

api_key = os.environ.get("OPENROUTER_API_KEY", "")
engine = RAGEngine(api_key=api_key, model=OPENROUTER_MODEL)
chunk_count = engine.init()
print(f"[startup] indexed {chunk_count} chunks")


#Routes

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "chunks_indexed": chunk_count, "model": OPENROUTER_MODEL})


@app.get("/api/documents")
def list_docs():
    seen = {}
    for c in engine.chunks:
        did = c["doc_id"]
        if did not in seen:
            seen[did] = {"doc_id": did, "title": c["doc_title"], "chunk_count": 0}
        seen[did]["chunk_count"] += 1
    return jsonify(list(seen.values()))


@app.post("/api/query")
def query():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    top_k = int(body.get("top_k", 4))
    use_ollama = bool(body.get("use_ollama", False))
    result = engine.query(question, top_k=top_k, use_ollama=use_ollama)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
