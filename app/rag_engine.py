from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Tuple

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def load_knowledge_base(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [item.embedding for item in resp.data]

def retrieve_relevant_chunks(query: str, kb_chunks: List[str], top_k: int = 3) -> List[str]:
    kb_vectors = embed_texts(kb_chunks)
    query_vector = embed_texts([query])[0]

    scored = []
    for chunk, vec in zip(kb_chunks, kb_vectors):
        sim = _cosine_similarity(query_vector, vec)
        scored.append((chunk, sim))

    scored.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored[:top_k]]