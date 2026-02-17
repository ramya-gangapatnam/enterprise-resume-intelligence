from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from typing import Optional
import os

from .schemas import EvaluateResponse
from .services import (
    extract_text_from_bytes,
    evaluate_resume,
    extract_skills,
    refine_suggestions_with_rag,
)

app = FastAPI(title="Enterprise Resume Intelligence", version="1.0")

MAX_FILE_MB = 5

@app.get("/")
def root():
    return {
        "service": "Enterprise Resume Intelligence",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "evaluate": "/evaluate (POST)"
        },
        "usage": {
            "evaluate": "POST /evaluate?semantic=true&rag=true with resume + jd files"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    resume: UploadFile = File(...),
    jd: UploadFile = File(...),
    semantic: bool = Query(False, description="Enable embeddings-based semantic matching"),
    rag: bool = Query(False, description="Enable RAG KB retrieval + refinement"),
):
    

    # basic safety checks
    for f in (resume, jd):
        if not f.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        contents = await f.read()
        if len(contents) > MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large (>{MAX_FILE_MB}MB)")
        f._cached_bytes = contents  # store for later

    try:
        resume_text = extract_text_from_bytes(resume.filename, resume._cached_bytes)
        jd_text = extract_text_from_bytes(jd.filename, jd._cached_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not resume_text or not jd_text:
        raise HTTPException(status_code=400, detail="Empty extracted text from resume or JD")

    # Phase 1
    base, usage1 = evaluate_resume(resume_text, jd_text)
    output = base.model_dump()
    output["llm_usage_phase1"] = usage1

    # Phase 2 (Semantic)
    if semantic:
        from .semantic_match import semantic_skill_match  # local import

        skills = extract_skills(resume_text, jd_text)
        sem = semantic_skill_match(skills["jd_skills"], skills["resume_skills"])

        output["semantic_matches"] = sem
        output["resume_skills_extracted"] = skills["resume_skills"]
        output["jd_skills_extracted"] = skills["jd_skills"]

    # Phase 3 (RAG)
    if rag:
        from .rag_engine import load_knowledge_base, retrieve_relevant_chunks

        kb_path = os.getenv("KB_PATH", "knowledge_base/ai_skills.txt")
        kb_chunks = load_knowledge_base(kb_path)

        rag_query = " ".join(base.missing_skills) or "resume improvement suggestions"
        top_chunks = retrieve_relevant_chunks(rag_query, kb_chunks, top_k=3)

        output["rag_enabled"] = True
        output["rag_query"] = rag_query
        output["rag_top_chunks"] = top_chunks
        refined, usage_refine = refine_suggestions_with_rag(base.missing_skills, top_chunks)
        output["rag_refined_suggestions"] = refined
        output["llm_usage_rag_refine"] = usage_refine

    else:
        output["rag_enabled"] = False

    return output
