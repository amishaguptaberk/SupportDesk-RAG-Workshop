"""Minimal RAG over the support tickets. No vector DB, no framework.

20 docs fit in memory, so retrieval is just cosine similarity + top-k.
Run:  python rag.py "how do I fix login failures after a password reset?"
Self-check (no API key needed):  python rag.py --test
"""
import json
import os
import sys
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def load_tickets(path="data/synthetic_tickets.json"):
    tickets = json.load(open(path))
    # one searchable blob per ticket
    for t in tickets:
        t["text"] = f"{t['title']}\n{t['description']}\nResolution: {t['resolution']}"
    return tickets


def embed(client, texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data])


def top_k(query_vec, doc_matrix, k=3):
    # cosine: vectors from OpenAI aren't unit-normalized, so divide by norms
    sims = doc_matrix @ query_vec / (
        np.linalg.norm(doc_matrix, axis=1) * np.linalg.norm(query_vec) + 1e-9
    )
    return np.argsort(-sims)[:k], sims


def answer(query, k=3):
    client = OpenAI()
    tickets = load_tickets()
    doc_matrix = embed(client, [t["text"] for t in tickets])
    qvec = embed(client, [query])[0]

    idx, sims = top_k(qvec, doc_matrix, k)
    context = "\n\n---\n\n".join(
        f"[{tickets[i]['ticket_id']}] {tickets[i]['text']}" for i in idx
    )

    # anti-hallucination: tell it to refuse when context is thin
    prompt = (
        "You are a support assistant. Answer ONLY from the tickets below. "
        "If they don't cover the question, say so plainly.\n\n"
        f"TICKETS:\n{context}\n\nQUESTION: {query}"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    print(f"\nRetrieved: {[tickets[i]['ticket_id'] for i in idx]} "
          f"(scores {[round(float(sims[i]), 3) for i in idx]})\n")
    print(resp.choices[0].message.content)


def _test():
    # retrieval ranking works without touching the API
    docs = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    idx, _ = top_k(np.array([1.0, 0.0]), docs, k=2)
    assert list(idx) == [0, 2], idx  # nearest two to [1,0]
    print("ok")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test()
    elif len(sys.argv) > 1:
        answer(sys.argv[1])
    else:
        print(__doc__)
