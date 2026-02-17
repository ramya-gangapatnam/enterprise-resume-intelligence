from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ResumeEvaluation(BaseModel):
    match_score: int = Field(..., ge=0, le=100)
    strengths: List[str]
    missing_skills: List[str]
    improvement_suggestions: List[str]

class EvaluateResponse(ResumeEvaluation):
    semantic_matches: Optional[List[Dict[str, Any]]] = None
    resume_skills_extracted: Optional[List[str]] = None
    jd_skills_extracted: Optional[List[str]] = None

    rag_enabled: Optional[bool] = None
    rag_query: Optional[str] = None
    rag_top_chunks: Optional[List[str]] = None
    rag_refined_suggestions: Optional[List[str]] = None

    llm_usage_phase1: Optional[Dict[str, Any]] = None
    llm_usage_rag_refine: Optional[Dict[str, Any]] = None

