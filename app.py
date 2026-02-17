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
                        "jd_skills": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["resume_skills", "jd_skills"],
                    "additionalProperties": False
                }
            }
        }
    )
    return json.loads(skill_response.output_text)

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
    parser.add_argument("--semantic", action="store_true", help="Enable embeddings-based semantic matching")
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

        # Start with base output
    output = validated.model_dump()

    # Optional: semantic matching
    if args.semantic:
        from semantic_match import semantic_skill_match

        skills = extract_skills(resume_text, jd_text)
        sem = semantic_skill_match(
            jd_skills=skills["jd_skills"],
            resume_skills=skills["resume_skills"]
        )

        output["semantic_matches"] = sem
        output["resume_skills_extracted"] = skills["resume_skills"]
        output["jd_skills_extracted"] = skills["jd_skills"]

        print("\n[bold magenta]Semantic Matching Enabled[/bold magenta]")

    # Save ONE final output (no overwrite)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"\n[bold green]Saved to {args.out}[/bold green]")


if __name__ == "__main__":
    main()