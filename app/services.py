import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document

from .schemas import ResumeEvaluation

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_bytes(filename: str, content: bytes) -> str:
    name = filename.lower()

    if name.endswith(".pdf"):
        # pypdf needs a file-like object
        import io
        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()

    if name.endswith(".docx"):
        import io
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs).strip()

    raise ValueError("Unsupported file format. Use .pdf or .docx")

def build_eval_prompt(resume_text: str, jd_text: str) -> str:
    return f"""
You are an evaluator. Compare the resume against the job description.

RULES:
- Only list strengths explicitly mentioned in the resume.
- Only list missing skills explicitly mentioned in the job description AND not present in the resume.
- If unclear, do NOT assume it.
- Keep each list item short (max 12 words).

Resume:
{resume_text}

Job Description:
{jd_text}

Return strict JSON with:
- match_score (0-100)
- strengths (list)
- missing_skills (list)
- improvement_suggestions (list)
""".strip()

def evaluate_resume(resume_text: str, jd_text: str):
    prompt = build_eval_prompt(resume_text, jd_text)

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.2,
        text={
            "format": {
                "type": "json_schema",
                "name": "resume_eval",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "match_score": {"type": "integer"},
                        "strengths": {"type": "array", "items": {"type": "string"}},
                        "missing_skills": {"type": "array", "items": {"type": "string"}},
                        "improvement_suggestions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["match_score", "strengths", "missing_skills", "improvement_suggestions"],
                    "additionalProperties": False,
                },
            }
        },
    )

    data = json.loads(response.output_text)
    validated = ResumeEvaluation(**data)

    usage = None
    if getattr(response, "usage", None):
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
            "cached_tokens": getattr(response.usage.input_tokens_details, "cached_tokens", 0),
        }

    return validated, usage


def extract_skills(resume_text: str, jd_text: str) -> dict:
    skill_prompt = f"""
Extract skills from the Resume and Job Description.

RULES:
- Return ONLY skills explicitly present in each text.
- Keep skills short phrases (1-4 words).
- Exclude job titles or role names (e.g., "AI Engineer", "Software Engineer").
- No duplicates.
- Do not infer.

Resume:
{resume_text}

Job Description:
{jd_text}
""".strip()

    skill_response = client.responses.create(
        model="gpt-4o-mini",
        input=skill_prompt,
        temperature=0,
        text={
            "format": {
                "type": "json_schema",
                "name": "skills_extract",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "resume_skills": {"type": "array", "items": {"type": "string"}},
                        "jd_skills": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["resume_skills", "jd_skills"],
                    "additionalProperties": False,
                },
            }
        },
    )

    return json.loads(skill_response.output_text)

def refine_suggestions_with_rag(missing_skills: list[str], retrieved_chunks: list[str]):
    refine_prompt = f"""
You are improving resume suggestions using ONLY the retrieved knowledge context.

Missing skills:
{missing_skills}

Retrieved knowledge context:
{chr(10).join(retrieved_chunks)}

Rules:
- Make suggestions specific and actionable (max 5).
- Do NOT invent experience.
- Keep each suggestion under 14 words.
- Return strict JSON.

Return JSON with:
- refined_suggestions (list of strings)
""".strip()

    refine_response = client.responses.create(
        model="gpt-4o-mini",
        input=refine_prompt,
        temperature=0.2,
        text={
            "format": {
                "type": "json_schema",
                "name": "rag_refine",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "refined_suggestions": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["refined_suggestions"],
                    "additionalProperties": False,
                },
            }
        },
    )

    refined = json.loads(refine_response.output_text)["refined_suggestions"]

    usage = None
    if getattr(refine_response, "usage", None):
        usage = {
            "input_tokens": refine_response.usage.input_tokens,
            "output_tokens": refine_response.usage.output_tokens,
            "total_tokens": refine_response.usage.total_tokens,
            "cached_tokens": getattr(refine_response.usage.input_tokens_details, "cached_tokens", 0),
        }

    return refined, usage


    