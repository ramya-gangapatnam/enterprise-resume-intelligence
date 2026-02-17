# Enterprise Resume Intelligence  
LLM-Powered Resume vs Job Description Evaluation System  

---

## Overview

Enterprise Resume Intelligence is a production-style AI system that evaluates how well a Resume matches a Job Description.

The system includes:

- Structured LLM evaluation
- Embeddings-based semantic skill matching
- Retrieval-Augmented Generation (RAG)
- Guardrails and schema enforcement
- Token usage monitoring
- CLI mode
- FastAPI service mode

This project demonstrates layered AI system design with production-oriented thinking.

---

# System Architecture

The system is built in 3 phases:

---

## Phase 1 – Structured Evaluation (LLM Core)

- Resume + JD ingestion (PDF/DOCX)
- Strict JSON schema enforcement
- Guardrails using Pydantic
- Low-temperature deterministic scoring
- Token usage tracking

Output:
- match_score (0–100)
- strengths
- missing_skills
- improvement_suggestions

---

## Phase 2 – Embeddings-Based Semantic Matching

Enhances evaluation beyond keyword matching.

### How it works:
1. Extract skills from resume + JD
2. Generate embeddings for each skill phrase
3. Compute cosine similarity
4. Label JD skills as:
   - matched
   - partial
   - missing

Output fields:
- semantic_matches
- resume_skills_extracted
- jd_skills_extracted

This demonstrates semantic understanding instead of string matching.

---

## Phase 3 – Retrieval-Augmented Generation (RAG)

Enhances suggestions using a knowledge base.

### Retrieval Layer
- Missing skills form the query
- Knowledge base chunks are embedded
- Top-k relevant chunks retrieved using cosine similarity

### Generation Layer
- Retrieved context injected into second LLM call
- Generates refined, grounded improvement suggestions

Additional output fields:
- rag_enabled
- rag_query
- rag_top_chunks
- rag_refined_suggestions

---

# Tech Stack

- Python
- OpenAI Responses API
- OpenAI Embeddings API
- FastAPI
- Pydantic
- PyPDF
- python-docx
- argparse
- cosine similarity (pure Python)

---

# Project Structure

enterprise-resume-intelligence/
│
├── app.py                      # CLI entry point
│
├── app/                        # FastAPI application
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry
│   ├── services.py             # Core LLM + embedding logic
│   ├── schemas.py              # Pydantic models
│   ├── semantic_match.py       # Phase 2 semantic similarity
│   ├── rag_engine.py           # Phase 3 retrieval engine
│
├── knowledge_base/
│   └── ai_skills.txt           # RAG knowledge source
│
├── requirements.txt
├── README.md
├── .env
└── outputs/

# CLI Mode (Local Execution)

## Setup

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Create .env:
OPENAI_API_KEY=your_key_here

Run Phase 1 Only
python app.py --resume data/resume.pdf --jd data/job_description.pdf

Run Phase 2 (Semantic Matching)
python app.py --resume data/resume.pdf --jd data/job_description.pdf --semantic

Run Phase 3 (RAG Enabled)
python app.py --resume data/resume.pdf --jd data/job_description.pdf --semantic --rag

FastAPI Service Mode
Start Server
uvicorn app.main:app --reload

Access:
Root: http://127.0.0.1:8000/
Health: http://127.0.0.1:8000/health
Swagger UI: http://127.0.0.1:8000/docs

POST /evaluate
Upload:
-resume (.pdf or .docx)
-jd (.pdf or .docx)

Query parameters:
-semantic=true
-rag=true

Example API Response Structure
{
  "match_score": 80,
  "strengths": [...],
  "missing_skills": [...],
  "improvement_suggestions": [...],

  "semantic_matches": [...],
  "resume_skills_extracted": [...],
  "jd_skills_extracted": [...],

  "rag_enabled": true,
  "rag_query": "...",
  "rag_top_chunks": [...],
  "rag_refined_suggestions": [...],

  "llm_usage_phase1": {
    "input_tokens": 555,
    "output_tokens": 112,
    "total_tokens": 667
  },
  "llm_usage_rag_refine": {
    "input_tokens": 200,
    "output_tokens": 54,
    "total_tokens": 254
  }
}

Token Usage Monitoring
Each LLM call tracks:
-input_tokens
-output_tokens
-total_tokens
-cached_tokens

This enables:
-Cost awareness
-Latency awareness
-Prompt optimization
-Production scalability thinking

Guardrails Implemented
-Strict JSON schema enforcement
-Pydantic validation
-Score range constraint (0–100)
-No inferred skills rule

Design Decisions
-CLI + API dual-mode architecture
-Embeddings for semantic comparison
-RAG for grounded refinement
-Token tracking for cost transparency
-Modular service layer (shared logic)