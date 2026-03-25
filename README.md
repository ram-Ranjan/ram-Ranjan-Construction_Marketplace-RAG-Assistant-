---
title: Indecimal RAG Assistant
emoji: 🏗️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Indecimal RAG Assistant

A minimal RAG pipeline for a construction marketplace. Answers user questions strictly from internal documents — no hallucination, no guessing.

## Live Demo

[huggingface.co/spaces/ramranjan/Indecimal_rag_assistant](https://huggingface.co/spaces/ramranjan/Indecimal_rag_assistant)

---

## Architecture
```
User Query
    │
    ▼
TF-IDF vectorization (same vocab as indexed docs)
    │
    ▼
FAISS IndexFlatIP — cosine similarity search
    │
    ▼
Top-4 chunks retrieved
    │
    ▼
Mistral 7B via OpenRouter — grounded answer generation
    │
    ▼
Answer + cited chunks shown in UI
```

---

## Embedding Model — TF-IDF

Went with TF-IDF instead of sentence-transformers for a few practical reasons:

- no model download, works fully offline
- zero GPU dependency  
- easy to debug — you can see exactly which terms drove a high score
- the documents have consistent vocabulary (brand names, prices, policy terms) so lexical matching works well here

Trade-off vs neural embeddings:

| | TF-IDF | Sentence Transformers |
|---|---|---|
| Setup | Instant, no downloads | Requires model download (~90MB) |
| Semantic depth | Lexical matching | True semantic understanding |
| Paraphrase handling | Misses synonyms | Handles paraphrases |
| Speed | Very fast | Slower (neural inference) |
| This corpus | Works well | Marginally better |

For production I'd swap in `all-MiniLM-L6-v2` or the Anthropic embeddings API.

---

## Chunking Strategy

Sliding window over words — 380 words per chunk, 75 word overlap. Each chunk stores its doc title and id for source attribution.

The overlap prevents information loss at boundaries — if a sentence spans two chunks, both will have partial context.
```python
def chunk_doc(doc, size=380, overlap=75):
    words = doc["content"].strip().split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append({
            "chunk_id": f"{doc['id']}_chunk_{len(chunks)}",
            "doc_title": doc["title"],
            "text": " ".join(words[start:end]),
        })
        if end == len(words):
            break
        start += size - overlap
    return chunks
```

---

## Vector Indexing & Retrieval

- Index: `faiss.IndexFlatIP` (inner product on unit-normalized vectors = cosine similarity)
- Query is vectorized using the same TF-IDF vocab built at index time
- Top-4 chunks returned by default (configurable)

---

## Grounding Enforcement

System prompt is strict — the LLM is told to answer only from provided context:
```
STRICT RULES:
1. Answer ONLY using the context chunks provided below.
2. If the answer is not in the context, say so honestly.
3. Mention which document your answer is based on.
4. Do not speculate beyond the context.
```

Retrieved chunks are injected directly into the prompt before the question. The UI also shows every retrieved chunk with its relevance score so the user can verify grounding themselves.

---

## Running Locally
```bash
git clone https://github.com/ramranjan/indecimal-rag
cd indecimal-rag

conda create -n indecimal-rag python=3.11 -y
conda activate indecimal-rag
pip install -r requirements.txt

# Windows CMD
set OPENROUTER_API_KEY=sk-or-v1-...
python backend/app.py
```

Then open `frontend/index.html` in your browser, or serve it:
```bash
cd frontend
python -m http.server 8080
# open http://localhost:8080
```

---

## Running the Evaluation
```bash
set OPENROUTER_API_KEY=sk-or-v1-...
python eval/run_eval.py
```

Results are written to `eval/eval_results.md`.

---

## Project Structure
```
indecimal-rag/
├── backend/
│   ├── app.py           Flask API server
│   ├── rag_engine.py    RAG pipeline (chunking, TF-IDF, FAISS, OpenRouter)
│   └── config.py        Config constants
├── frontend/
│   └── index.html       Single-file chatbot UI (no build step needed)
├── documents/
│   ├── doc1.md          Company overview & customer journey
│   ├── doc2.md          Package comparison & specs
│   └── doc3.md          Policies, quality, guarantees
├── eval/
│   └── run_eval.py      15-question evaluation script
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Evaluation Results

15 test questions derived from the documents:

| Metric | Score |
|---|---|
| Retrieval hits | 15/15 |
| No hallucination risk | 15/15 |
| Complete answers | 13/15 |
| Grounded answers | 13/15 |
| Avg response time | 24.68s |

**Observations:**

Retrieval is solid across all 15 questions — TF-IDF works well here because the document vocabulary is domain-specific and consistent. Brand names, price figures, and policy terms all score high when queried directly.

The 2 incomplete answers were caused by rate limiting on the free OpenRouter tier — the model returned shorter responses after multiple retries. This is an infrastructure constraint, not a retrieval or grounding issue.

Latency is dominated by OpenRouter API response time. The TF-IDF retrieval + FAISS search itself takes under 5ms.

Grounding held up across all meaningful responses — the system prompt successfully prevents the model from adding information outside the retrieved chunks.

---

## Limitations

- Small corpus (3 documents, 7 chunks) — more documents would improve coverage
- TF-IDF misses paraphrases — "protective gear" won't match "PPE"
- No reranking step — a cross-encoder would improve precision
- No chat memory — each query is independent
- Free tier rate limits affect response time

---

## Tech Stack

- **Embedding:** TF-IDF (custom, zero dependencies)
- **Vector Index:** FAISS `IndexFlatIP`
- **LLM:** Mistral 7B via OpenRouter (free tier)
- **Backend:** Python + Flask
- **Frontend:** Vanilla HTML/CSS/JS (single file, no framework)
- **Deployment:** HuggingFace Spaces (Docker)