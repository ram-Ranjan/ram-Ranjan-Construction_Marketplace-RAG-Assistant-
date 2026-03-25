import os
import re
import math
import time
import numpy as np
import faiss
import requests

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, DOCS_DIR, TOP_K,
    OPENROUTER_API_URL, OPENROUTER_MODEL,
    OLLAMA_MODEL, OLLAMA_URL,
)

SYSTEM_PROMPT = (
    "You are an AI assistant for Indecimal, a home construction marketplace.\n"
    "STRICT RULES:\n"
    "1. Answer ONLY using the context chunks provided below.\n"
    "2. If the answer is not in the context, say: "
    "\"I don't have enough information in the provided documents to answer that.\"\n"
    "3. Mention which document your answer is based on.\n"
    "4. Do not speculate or add information beyond what is in the context.\n"
    "5. Be clear and direct."
)


def load_docs(docs_dir=DOCS_DIR):
    docs = []
    for fname in sorted(os.listdir(docs_dir)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(docs_dir, fname)
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        m = re.search(r"^#\s+(.+)", raw, re.MULTILINE)
        title = m.group(1).strip() if m else fname
        docs.append({"id": fname.replace(".md", ""), "title": title, "content": raw})
    return docs



def chunk_doc(doc, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = doc["content"].strip().split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append({
            "chunk_id": f"{doc['id']}_chunk_{len(chunks)}",
            "doc_title": doc["title"],
            "doc_id": doc["id"],
            "text": " ".join(words[start:end]),
        })
        if end == len(words):
            break
        start += size - overlap
    return chunks

def chunk_all(docs):
    out = []
    for d in docs:
        out.extend(chunk_doc(d))
    return out


def tokenize(text):
    return re.findall(r"[a-z0-9₹]+", text.lower())

def build_vocab_idf(chunks):
    df = {}
    for c in chunks:
        for tok in set(tokenize(c["text"])):
            df[tok] = df.get(tok, 0) + 1
    N = len(chunks)
    vocab = sorted(df.keys())
    word2idx = {w: i for i, w in enumerate(vocab)}
    idf = np.array([math.log((N + 1) / (df[w] + 1)) + 1.0 for w in vocab], dtype="float32")
    return vocab, word2idx, idf

def tfidf_vector(text, word2idx, idf):
    tokens = tokenize(text)
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    total = len(tokens) if tokens else 1
    vec = np.zeros(len(word2idx), dtype="float32")
    for word, count in tf.items():
        if word in word2idx:
            vec[word2idx[word]] = (count / total) * idf[word2idx[word]]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec



def build_index(chunks, word2idx, idf):
    vecs = np.stack([tfidf_vector(c["text"], word2idx, idf) for c in chunks])
    index = faiss.IndexFlatIP(len(word2idx))
    index.add(vecs) # type: ignore
    return index



def retrieve(query, index, chunks, word2idx, idf, top_k=TOP_K):
    q_vec = tfidf_vector(query, word2idx, idf).reshape(1, -1)
    scores, idxs = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        c = dict(chunks[idx])
        c["relevance_score"] = round(float(score), 4)
        results.append(c)
    return results


def build_prompt(query, chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Chunk {i} — {c['doc_title']}]\n{c['text']}")
    context = "\n\n".join(parts)
    return f"{context}\n\nQuestion: {query}\nAnswer based strictly on the context above:"



def generate(query, retrieved_chunks, api_key, model=OPENROUTER_MODEL):
    if not api_key:
        return "Error: no OpenRouter API key provided."
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://indecimal-rag.app",
        "X-Title": "Indecimal RAG Assistant",
    }
    # merge system prompt into user message — works with all models
    full_prompt = SYSTEM_PROMPT + "\n\n" + build_prompt(query, retrieved_chunks)
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": full_prompt},
        ],
    }
    for attempt in range(5):
        try:
            resp = requests.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=90)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"[rate limit] waiting {wait}s before retry {attempt+1}/5")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                return f"OpenRouter error: {data['error'].get('message', data['error'])}"
            if "choices" not in data:
                return f"Unexpected response: {str(data)[:300]}"
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            if attempt < 3:
                continue
            return "Error: request timed out."
        except Exception as e:
            return f"Error from OpenRouter: {e}"
    return "Error: still rate limited. Wait a minute and try again."



def generate_ollama(query, retrieved_chunks, model=OLLAMA_MODEL):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(query, retrieved_chunks)},
        ],
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "Error: Ollama not running. Start with: ollama serve"
    except Exception as e:
        return f"Error from Ollama: {e}"


class RAGEngine:
    def __init__(self, api_key="", model=OPENROUTER_MODEL):
        self.api_key = api_key
        self.model = model
        self.chunks = []
        self.word2idx = {}
        self.idf = None
        self.index = None
        self.ready = False

    def init(self):
        docs = load_docs()
        self.chunks = chunk_all(docs)
        _, self.word2idx, self.idf = build_vocab_idf(self.chunks)
        self.index = build_index(self.chunks, self.word2idx, self.idf)
        self.ready = True
        return len(self.chunks)

    def query(self, question, top_k=TOP_K, use_ollama=False):
        if not self.ready:
            return {"error": "engine not initialized"}
        chunks = retrieve(question, self.index, self.chunks, self.word2idx, self.idf, top_k)
        t0 = time.time()
        if use_ollama:
            answer = generate_ollama(question, chunks)
        else:
            answer = generate(question, chunks, self.api_key, self.model)
        return {
            "question": question,
            "retrieved_chunks": chunks,
            "answer": answer,
            "response_time": round(time.time() - t0, 2),
        }