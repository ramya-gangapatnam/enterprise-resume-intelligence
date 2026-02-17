from openai import OpenAI
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from pydantic import BaseModel, Field
import json
from rich import print
import argparse

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# Text Extraction
# ---------------------------
def extract_text(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()

    if file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs).strip()

    raise ValueError("Unsupported file format. Use .pdf or .docx")


# ---------------------------
# Pydantic Schema
# ---------------------------
class ResumeEvaluation(BaseModel):
    match_score: int = Field(..., ge=0, le=100)
    strengths: list[str]
    missing_skills: list[str]
    improvement_suggestions: list[str]


# ---------------------------
# Prompt
# ---------------------------
def build_prompt(resume_text: str, jd_text: str) -> str:
    return f"""
You are an evaluator. Compare the resume against the job description.

RULES:
- Only list strengths that are explicitly mentioned in the resume.
- Only list missing skills that are explicitly mentioned in the job description AND not present in the resume.
- If something is unclear, do NOT assume it. Say it is not mentioned.
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


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Enterprise Resume Intelligence: Resume vs JD evaluator")
    parser.add_argument("--resume", required=True, help="Path to resume file (.pdf or .docx)")
    parser.add_argument("--jd", required=True, help="Path to job description file (.pdf or .docx)")
    parser.add_argument("--out", default="outputs/result.json", help="Output JSON path (default: outputs/result.json)")
    args = parser.parse_args()

    resume_text = extract_text(args.resume)
    jd_text = extract_text(args.jd)

    if not resume_text:
        raise ValueError("Resume text extraction returned empty content.")
    if not jd_text:
        raise ValueError("Job description text extraction returned empty content.")

    prompt = build_prompt(resume_text, jd_text)

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

    print("[bold green]LLM Output:[/bold green]")
    print(response.output_text)

    print("\n[bold blue]Token Usage:[/bold blue]")
    print(response.usage)

    # Validate with Pydantic
    data = json.loads(response.output_text)
    validated = ResumeEvaluation(**data)

    # Save output
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(validated.model_dump(), f, indent=4)

    print(f"\n[bold green]Saved to {args.out}[/bold green]")


if __name__ == "__main__":
    main()