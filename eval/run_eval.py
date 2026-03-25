import sys,os,time

from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from rag_pipeline import RAGEngine
from config import OPENROUTER_MODEL

QUESTIONS = [
    "What packages does Indecimal offer and what are their per-sqft prices?",
    "What brand of steel is used in the Essential package?",
    "How does Indecimal's escrow payment model work?",
    "What is the ceiling height across all packages?",
    "What paint brand is used for interior painting in the Pinnacle package?",
    "How does Indecimal handle construction delays?",
    "What are the bathroom sanitary fitting allowances for each package?",
    "What is included in the zero cost maintenance program?",
    "What type of windows are offered in the Premier package?",
    "How many quality checkpoints does Indecimal have?",
    "What is the main door specification for the Infinia package?",
    "How does home financing work with Indecimal?",
    "What flooring options are available for living and dining areas?",
    "What is the partner onboarding process?",
    "What kitchen sink faucet brands are used in the Essential vs Pinnacle package?",
]

GROUNDING_KEYWORDS = [
     "indecimal", "package", "₹", "essential", "premier", "infinia", "pinnacle",
        "escrow", "quality", "checkpoint", "maintenance", "steel", "cement", "wallet",
        "teak", "upvc", "asian paints", "jaquar", "parryware", "chunk", "document",
        "construction", "financing", "floor", "window", "door", "paint", "bathroom",
        "kitchen", "sink", "ceiling", "concrete", "rcc", "block", "aggregate"
]


def check_grounded(answer):
    lower = answer.lower()
    return any(kw in lower for kw in GROUNDING_KEYWORDS)


def check_retrieval(chunks):
    return len(chunks) > 0 and chunks[0]["relevance_score"] > 0.05


def check_no_hallucination(answer):
    bad = ["as of my knowledge", "based on my training", "i believe", "i think", "probably"]
    lower = answer.lower()
    return not any(p in lower for p in bad) and len(answer) > 60


def check_complete(answer):
    return len(answer) > 100


def run():
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY before running")
        sys.exit(1)

    engine = RAGEngine(api_key=api_key, model=OPENROUTER_MODEL)
    n = engine.init()
    print(f"engine ready — {n} chunks indexed\n")

    rows = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i:02d}/15] {q[:65]}...")
        time.sleep(5) 
        t0 = time.time()
        result = engine.query(q)
        elapsed = round(time.time() - t0, 1)

        answer = result.get("answer", "")
        chunks = result.get("retrieved_chunks", [])

        r_ok = check_retrieval(chunks)
        no_hall = check_no_hallucination(answer)
        complete = check_complete(answer)
        grounded = check_grounded(answer)

        rows.append({
            "q": q,
            "retrieval_ok": r_ok,
            "no_hallucination": no_hall,
            "complete": complete,
            "grounded": grounded,
            "time": elapsed,
        })

        status = " ".join([
            "✅" if r_ok else "❌",
            "✅" if no_hall else "❌",
            "✅" if complete else "❌",
            "✅" if grounded else "❌",
            f"{elapsed}s",
        ])
        print(f"       ret  hall  comp  grnd  time")
        print(f"       {status}\n")

    # write markdown report
    out_path = os.path.join(os.path.dirname(__file__), "eval_results.md")
    avg_time = round(sum(r["time"] for r in rows) / len(rows), 2)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# RAG Evaluation Results\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Chunks indexed:** {n}\n\n")
        f.write("| # | Question | Retrieval OK | Hallucination Risk | Complete | Grounded | Time (s) |\n")
        f.write("|---|----------|:---:|:---:|:---:|:---:|---:|\n")
        for i, r in enumerate(rows, 1):
            q_short = r["q"][:50] + "…" if len(r["q"]) > 50 else r["q"]
            f.write(
                f"| {i} | {q_short} "
                f"| {'✅' if r['retrieval_ok'] else '❌'} "
                f"| {'✅' if r['no_hallucination'] else '❌'} "
                f"| {'✅' if r['complete'] else '❌'} "
                f"| {'✅' if r['grounded'] else '❌'} "
                f"| {r['time']} |\n"
            )
        f.write(f"\n## Summary\n")
        f.write(f"- Retrieval hits: **{sum(r['retrieval_ok'] for r in rows)}/15**\n")
        f.write(f"- No hallucination risk: **{sum(r['no_hallucination'] for r in rows)}/15**\n")
        f.write(f"- Complete answers: **{sum(r['complete'] for r in rows)}/15**\n")
        f.write(f"- Grounded answers: **{sum(r['grounded'] for r in rows)}/15**\n")
        f.write(f"- Average response time: **{avg_time}s**\n")

    print(f"\nResults written to {out_path}")
    print(f"Average time: {avg_time}s")


if __name__ == "__main__":
    run()
