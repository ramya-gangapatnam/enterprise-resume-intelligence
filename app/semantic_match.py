from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    # Pure python cosine similarity (no numpy needed)
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def embed_texts(texts: List[str]) -> List[List[float]]:
    # Embeddings endpoint
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [item.embedding for item in resp.data]

def semantic_skill_match(
    jd_skills: List[str],
    resume_skills: List[str],
    match_threshold: float = 0.80,
    partial_threshold: float = 0.65
) -> List[Dict]:
    """
    For each JD skill, find the best matching resume skill based on cosine similarity.
    """
    if not jd_skills:
        return []
    if not resume_skills:
        return [
            {"jd_skill": s, "best_resume_skill": None, "similarity": 0.0, "status": "missing"}
            for s in jd_skills
        ]

    jd_vecs = embed_texts(jd_skills)
    resume_vecs = embed_texts(resume_skills)

    results = []
    for jd_skill, jd_v in zip(jd_skills, jd_vecs):
        best_idx = -1
        best_sim = -1.0
        for i, r_v in enumerate(resume_vecs):
            sim = _cosine_similarity(jd_v, r_v)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        best_resume_skill = resume_skills[best_idx] if best_idx >= 0 else None

        if best_sim >= match_threshold:
            status = "matched"
        elif best_sim >= partial_threshold:
            status = "partial"
        else:
            status = "missing"

        results.append({
            "jd_skill": jd_skill,
            "best_resume_skill": best_resume_skill,
            "similarity": round(best_sim, 3),
            "status": status
        })

    return results