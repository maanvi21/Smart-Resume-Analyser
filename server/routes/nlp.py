# routes/resume.py
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
import json
import re
import pdfplumber

router = APIRouter()

# Known tech/skill terms to always capture even if NER misses them
TECH_PATTERN = re.compile(
    r'\b(python|javascript|typescript|java|c\+\+|c#|ruby|go|rust|swift|kotlin|php|scala|'
    r'react|angular|vue|next\.?js|nuxt|svelte|node\.?js|express|django|flask|fastapi|spring|'
    r'postgresql|mysql|mongodb|redis|sqlite|cassandra|dynamodb|firebase|supabase|'
    r'docker|kubernetes|aws|azure|gcp|terraform|ansible|jenkins|github|gitlab|ci/cd|'
    r'machine learning|deep learning|nlp|computer vision|tensorflow|pytorch|keras|scikit-learn|'
    r'pandas|numpy|matplotlib|sql|nosql|graphql|rest|api|microservices|'
    r'html|css|sass|tailwind|bootstrap|figma|git|linux|bash|'
    r'agile|scrum|jira|confluence|system design|data structures|algorithms)\b',
    re.IGNORECASE
)

def extract_skills_from_jd(jd_text: str) -> list[str]:
    """
    Extract actual skills and technologies from a job description using:
    1. Regex for known tech terms
    2. BERT NER for org/misc entities (tools, frameworks often tagged as ORG/MISC)
    Returns a deduplicated, clean list.
    """
    found = set()

    # ── 1. Regex: catch known tech stack terms ───────────────────────────────
    for match in TECH_PATTERN.finditer(jd_text):
        found.add(match.group(0).lower().strip())

    # ── 2. BERT NER: catch tool/framework names NER recognises ──────────────
    try:
        from transformers import pipeline
        ner = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
        )
        MAX_CHARS = 2000
        chunks = [jd_text[i:i+MAX_CHARS] for i in range(0, len(jd_text), MAX_CHARS)]
        for chunk in chunks:
            entities = ner(chunk)
            for ent in entities:
                word = ent.get("word", "").strip()
                group = ent.get("entity_group", "")
                # ORG and MISC often contain tool/product names in JDs
                if group in ("ORG", "MISC") and len(word) > 1:
                    found.add(word.lower())
    except Exception:
        pass  # if NER fails, regex results still carry on

    # ── 3. Clean up ──────────────────────────────────────────────────────────
    NOISE = {"and", "or", "the", "with", "for", "a", "an", "in", "of", "to",
             "is", "are", "we", "you", "our", "your", "its", "be", "will"}
    keywords = sorted([
        k for k in found
        if k not in NOISE and len(k) > 1 and not k.isdigit()
    ])

    return keywords


@router.post("/extract-features")
async def extract_features(
    file: UploadFile = File(...),
    requirements: str = Form(...)
):
    """
    Accept a PDF resume + free-form job description paragraph.
    Uses NER + regex to extract real skills/tech keywords from the JD.
    """
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are accepted.")

        # ── Extract resume text ──────────────────────────────────────────────
        with pdfplumber.open(file.file) as pdf:
            pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
            extracted_text = "\n".join(pages)

        if not extracted_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract any text from the PDF.")

        # ── Extract keywords from job description ────────────────────────────
        keywords = extract_skills_from_jd(requirements)

        if not keywords:
            raise HTTPException(status_code=422, detail="Could not extract any skills from the job description.")

        data = {
            "filename":        file.filename,
            "text":            extracted_text,
            "requirements":    keywords,
            "job_description": requirements,
        }

        with open("uploaded_resume.json", "w", encoding="utf-8") as jf:
            json.dump(data, jf, indent=4)

        return {
            "message":            "Resume processed successfully",
            "pages_extracted":    len(pages),
            "keywords_extracted": keywords,
            "requirements_count": len(keywords),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))