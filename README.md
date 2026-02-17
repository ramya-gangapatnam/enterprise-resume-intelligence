# Enterprise Resume Intelligence (LLM-Powered Resume vs JD Evaluator)

A lightweight, production-style resume evaluation tool that compares a **Resume** against a **Job Description** and returns a **strict JSON report** with:

- Match score (0–100)
- Strengths (from resume text)
- Missing skills (from job description not found in resume)
- Improvement suggestions

This project focuses on **structured outputs, validation, and cost awareness** (token usage) — the practical foundations of Applied AI Engineering.

---

## What it does

1. Reads **Resume** and **Job Description** from **PDF or DOCX**
2. Extracts text reliably
3. Sends both to an LLM (OpenAI API)
4. Enforces **strict JSON schema output**
5. Validates the output with **Pydantic**
6. Saves results to `outputs/result.json`
7. Prints token usage for cost monitoring

---

## Tech Stack

- Python
- OpenAI Responses API
- Pydantic (validation + guardrails)
- PyPDF (PDF parsing)
- python-docx (DOCX parsing)
- dotenv (.env secrets)
- rich (clean terminal output)

---

## Project Structure

```txt
enterprise-resume-intelligence/
  app.py
  README.md
  requirements.txt
  .env
  data/
    resume.pdf
    job_description.pdf
  outputs/
    result.json              # ignored (changes every run)
    sample_result.json       # committed example output

How to Run
1) Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2) Add your API key
Create .env:
OPENAI_API_KEY=your_key_here

3) Add input files
Place your files here:
data/resume.pdf (or .docx)
data/job_description.pdf (or .docx)

4) Run
python app.py --resume data/resume.pdf --jd data/job_description.pdf

Outputs:
Console: JSON + token usage
File: outputs/result.json

Example Output
See: outputs/sample_result.json
Example format:

{
  "match_score": 80,
  "strengths": ["..."],
  "missing_skills": ["..."],
  "improvement_suggestions": ["..."]
}