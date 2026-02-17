# Enterprise Resume Intelligence
LLM-Powered Resume vs Job Description Evaluation System

---

## Overview

Enterprise Resume Intelligence is a production-style Applied AI system that evaluates how well a Resume matches a Job Description.

The system returns:

- Match Score (0–100)
- Strengths
- Missing Skills
- Improvement Suggestions
- (Optional) Semantic Skill Matching using Embeddings

This project demonstrates structured LLM integration, guardrails, and semantic matching — built with production mindset.

---

## Why This Project?

Recruiters and hiring teams often manually evaluate resumes against job descriptions. This system automates that evaluation using:

- LLM-based structured scoring
- Guardrails for reliability
- Embeddings for semantic skill matching
- Token logging for cost awareness

The goal is not just generation — but controlled, validated AI output.

---

# Architecture Overview

The system follows layered AI system design:

### 1️⃣ Input Layer
- PDF/DOCX ingestion
- Resume and JD text extraction
- Input validation

### 2️⃣ Reasoning Layer (LLM)
- Low-temperature evaluation for consistency
- Strict JSON schema enforcement
- No inference beyond explicit text

### 3️⃣ Guardrail Layer
- Pydantic validation
- Score constrained to 0–100
- Deterministic output structure

### 4️⃣ Semantic Matching Layer (Phase 2)
- Skill extraction via structured output
- Embedding generation
- Cosine similarity calculation
- Matched / Partial / Missing classification

### 5️⃣ Monitoring Layer
- Token usage logging
- Output persistence

---

# Tech Stack

- Python
- OpenAI Responses API
- OpenAI Embeddings API
- Pydantic (Validation & Guardrails)
- PyPDF (PDF Parsing)
- python-docx (DOCX Parsing)
- argparse (CLI Interface)
- rich (Console Output)

---

# Project Structure
enterprise-resume-intelligence/
│
├── app.py
├── semantic_match.py
├── README.md
├── requirements.txt
├── .env # Not committed
│
├── data/ # Ignored in Git
│ ├── resume.pdf
│ └── job_description.pdf
│
└── outputs/
├── result.json
└── sample_result.json


---

# How to Run

1️⃣ Setup Virtual Environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2️⃣ Add Your API Key
Create a .env file:
OPENAI_API_KEY=your_key_here

3️⃣ Add Input Files
Place your resume and job description inside:
data/
Example:
data/resume.pdf
data/job_description.pdf

4️⃣ Run Phase 1 (Structured Evaluation)
python app.py --resume data/resume.pdf --jd data/job_description.pdf

Output:
Structured JSON printed to console
Saved to outputs/result.json

5️⃣ Run Phase 2 (Semantic Matching Enabled)
python app.py --resume data/resume.pdf --jd data/job_description.pdf --semantic

How to Run Phase 3
python app.py --resume data/resume.pdf --jd data/job_description.pdf --semantic --rag

This adds:
-semantic_matches
-resume_skills_extracted
-jd_skills_extracted

to the final output.

Example Output Structure:
{
  "match_score": 80,
  "strengths": [...],
  "missing_skills": [...],
  "improvement_suggestions": [...],
  "semantic_matches": [
    {
      "jd_skill": "Python",
      "best_resume_skill": "Python",
      "similarity": 1.0,
      "status": "matched"
    }
  ],  
  "rag_enabled": true,
  "rag_query": "...",
  "rag_top_chunks": [...],
  "rag_refined_suggestions": [...]
}