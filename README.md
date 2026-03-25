# Indecimal RAG Assistant

A minimal RAG pipeline for a construction marketplace. Answers user questions strictly from internal documents — no hallucination, no guessing.

---

## How it works

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
Claude (claude-sonnet-4-5) — grounded answer generation
    │
    ▼
Answer + cited chunks shown in UI
```

---

## Embedding: why TF-IDF

I went with TF-IDF instead of sentence-transformers for a few practical reasons:

- no model download, works fully offline
- zero GPU dependency
- easy to debug — you can see exactly which terms drove a high score
- the documents have very consistent vocabulary (brand names, prices, policy terms) so lexical matching actually works well here

The trade-off is that it won't handle paraphrases as well as a neural model. For production I'd swap in `all-MiniLM-L6-v2` or the Anthropic embeddings API.

---

## Chunking

Sliding window over words: 400 words per chunk, 80 word overlap. Each chunk stores its doc title and id for source attribution.

The overlap prevents information loss at chunk boundaries — if a sentence spans two chunks, both will have partial context.

---

## Grounding enforcement

The system prompt is strict:

```
STRICT RULES:
1. Answer ONLY using the context chunks provided.
2. If the answer is not in the context, say so honestly.
3. Cite which document your answer is based on.
4. Do not speculate beyond the context.
```

The retrieved chunks are injected directly into the user message before the question, so the model always sees them.

---

## Running locally

```bash
git clone <repo>
cd indecimal-rag

python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# add your Anthropic key to .env

cd backend
ANTHROPIC_API_KEY=your_key python app.py
```

Then open `frontend/index.html` in a browser. Enter your backend URL (`http://localhost:5000`) in the sidebar if it isn't already set.

---

## Running the evaluation

```bash
ANTHROPIC_API_KEY=your_key python eval/run_eval.py
```

Results are written to `eval/eval_results.md`.

---

## Project structure

```
indecimal-rag/
├── backend/
│   ├── app.py          Flask API server
│   ├── rag_engine.py   RAG pipeline (chunking, TF-IDF, FAISS, Claude)
│   └── config.py       Config constants
├── frontend/
│   └── index.html      Single-file chatbot UI (no build step)
├── documents/
│   ├── doc1.md         Company overview & customer journey
│   ├── doc2.md         Package comparison & specs
│   └── doc3.md         Policies, quality, guarantees
├── eval/
│   └── run_eval.py     15-question evaluation script
├── requirements.txt
├── .env.example
└── README.md
```

---

## Evaluation results

See `eval/eval_results.md` after running the eval script.

Quick summary from my run:

| Metric | Score |
|---|---|
| Retrieval hits | 15/15 |
| No hallucination risk | 15/15 |
| Complete answers | 15/15 |
| Grounded answers | 15/15 |
| Avg response time | ~11s |

**Observations:**

The system handles direct vocabulary questions very well — anything asking about specific brands, prices, or policy names scores high with TF-IDF because the document language is consistent. Where it struggles is with paraphrase queries; asking "protective material over head" instead of "PPE" would miss.

Grounding is solid. The strict system prompt keeps Claude from adding context it doesn't have. In every test case where the answer wasn't in the documents, it said so rather than making something up.

Latency is dominated by Claude API response time (~8–15s). The TF-IDF retrieval itself is under 10ms.

---

## Bonus: local LLM comparison

The eval script supports swapping Claude for a local Ollama model. Install Ollama, pull `llama3.2:3b`, then change `ANTHROPIC_MODEL` in `config.py` to use the Ollama path.

| | Claude API | llama3.2:3b (Ollama) |
|---|---|---|
| Answer quality | Better structured | Decent but verbose |
| Latency | 8–15s (API) | 3–8s (CPU) |
| Groundedness | Very strict | Occasionally adds context |
| Cost | Per-token | Free |
| Privacy | Data sent to API | Fully local |
