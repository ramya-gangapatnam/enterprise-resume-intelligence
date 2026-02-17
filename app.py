from openai import OpenAI
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from pydantic import BaseModel, Field
import json
from rich import print

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# Text Extraction
# ---------------------------

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    else:
        raise ValueError("Unsupported file format")


# ---------------------------
# Pydantic Schema
# ---------------------------

class ResumeEvaluation(BaseModel):
    match_score: int = Field(..., ge=0, le=100)
    strengths: list[str]
    missing_skills: list[str]
    improvement_suggestions: list[str]


# ---------------------------
# Main Logic
# ---------------------------

resume_text = extract_text("data/resume.pdf")
jd_text = extract_text("data/job_description.pdf")

prompt = f"""
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
"""

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
                    "improvement_suggestions": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["match_score", "strengths", "missing_skills", "improvement_suggestions"],
                "additionalProperties": False
            }
        }
    }
)

print("[bold green]LLM Output:[/bold green]")
print(response.output_text)

print("\n[bold blue]Token Usage:[/bold blue]")
print(response.usage)

# Validate with Pydantic
data = json.loads(response.output_text)
validated = ResumeEvaluation(**data)

# Save output
os.makedirs("outputs", exist_ok=True)
with open("outputs/result.json", "w") as f:
    json.dump(validated.model_dump(), f, indent=4)

print("\n[bold green]Saved to outputs/result.json[/bold green]")