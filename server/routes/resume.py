from fastapi import APIRouter, File, Form, UploadFile, HTTPException
import json
import re
import pdfplumber
import pandas as pd
import os

router = APIRouter()

# ── Path to the resume dataset CSV ────────────────────────────────────────────
RESUME_CSV_PATH = os.environ.get("RESUME_CSV_PATH", "Resume.csv")

# ── Cache the dataset in memory after first load ───────────────────────────────
_resume_df: pd.DataFrame | None = None

def get_resume_df() -> pd.DataFrame | None:
    """Load and cache the resume CSV dataset (ID, Resume_str, Resume_html, Category)."""
    global _resume_df
    if _resume_df is not None:
        return _resume_df
    if not os.path.exists(RESUME_CSV_PATH):
        return None
    try:
        df = pd.read_csv(RESUME_CSV_PATH)
        # Normalise column names to lower-case for robustness
        df.columns = [c.strip().lower() for c in df.columns]
        # We only need the plain text and category
        required = {"resume_str", "category"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {required}. Found: {list(df.columns)}")
        df = df[["resume_str", "category"]].dropna(subset=["resume_str", "category"])
        df["resume_str"] = df["resume_str"].astype(str).str.strip()
        df["category"] = df["category"].astype(str).str.strip()
        _resume_df = df
        print(f"[dataset] Loaded {len(df):,} resumes across {df['category'].nunique()} categories.")
        return _resume_df
    except Exception as exc:
        print(f"[dataset] Failed to load resume CSV: {exc}")
        return None


# ── Known tech/skill terms to always capture even if NER misses them ──────────
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
        chunks = [jd_text[i:i + MAX_CHARS] for i in range(0, len(jd_text), MAX_CHARS)]
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


def predict_category(resume_text: str) -> dict:
    """
    Predict the most likely job category for the uploaded resume by
    semantically comparing it against the reference dataset.

    Returns the top-3 predicted categories with confidence scores.
    Gracefully falls back to an empty result if the dataset is unavailable.
    """
    df = get_resume_df()
    if df is None or df.empty:
        return {"predicted_category": None, "top_categories": [], "dataset_used": False}

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Encode the uploaded resume (truncated to 512 tokens worth of chars)
        resume_emb = model.encode(
            resume_text[:3000], convert_to_numpy=True, normalize_embeddings=True
        )

        # Build category-level embeddings: average of up to 30 samples per category
        categories = df["category"].unique().tolist()
        cat_scores: dict[str, float] = {}

        for cat in categories:
            cat_texts = df[df["category"] == cat]["resume_str"].tolist()
            # Sample up to 30 to keep inference fast
            sample = cat_texts[:30]
            cat_embs = model.encode(
                [t[:1500] for t in sample],
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=16,
                show_progress_bar=False,
            )
            # Cosine similarity: since embeddings are L2-normalised, dot product = cosine
            centroid = cat_embs.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            cat_scores[cat] = float(np.dot(resume_emb, centroid))

        sorted_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
        top3 = [
            {"category": cat, "confidence": round(score, 4)}
            for cat, score in sorted_cats[:3]
        ]

        return {
            "predicted_category": top3[0]["category"] if top3 else None,
            "top_categories": top3,
            "dataset_used": True,
        }

    except Exception as exc:
        print(f"[predict_category] Error: {exc}")
        return {"predicted_category": None, "top_categories": [], "dataset_used": False}


def get_dataset_skill_benchmark(category: str | None, keywords: list[str]) -> dict:
    """
    Using the dataset, find what skills are most common in resumes of the
    predicted category and surface any high-frequency skills that are absent
    from both the JD keywords and the uploaded resume.

    Returns a dict with benchmark insight.
    """
    df = get_resume_df()
    if df is None or df.empty or not category:
        return {"benchmark_skills": [], "category_sample_size": 0}

    cat_df = df[df["category"].str.lower() == category.lower()]
    if cat_df.empty:
        return {"benchmark_skills": [], "category_sample_size": 0}

    # Count skill occurrences across category resumes
    skill_counts: dict[str, int] = {}
    for text in cat_df["resume_str"].tolist()[:200]:  # cap at 200 for speed
        for match in TECH_PATTERN.finditer(str(text)):
            skill = match.group(0).lower().strip()
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

    total = len(cat_df[:200])
    # Skills present in ≥30 % of category resumes but absent from JD keywords
    jd_set = {k.lower() for k in keywords}
    benchmark = [
        {"skill": sk, "frequency_pct": round(cnt / total * 100, 1)}
        for sk, cnt in sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        if sk not in jd_set and cnt / total >= 0.30
    ][:10]

    return {
        "benchmark_skills": benchmark,
        "category_sample_size": total,
    }


@router.post("/extract-features")
async def extract_features(
    file: UploadFile = File(...),
    requirements: str = Form(...)
):
    """
    Accept a PDF resume + free-form job description paragraph.
    Uses NER + regex to extract real skills/tech keywords from the JD,
    predicts the resume's job category from the reference dataset, and
    surfaces category-level skill benchmarks.
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
            raise HTTPException(
                status_code=422,
                detail="Could not extract any skills from the job description."
            )

        # ── Dataset-powered category prediction ──────────────────────────────
        print("[extract-features] Predicting resume category from dataset...")
        category_info = predict_category(extracted_text)

        # ── Dataset-powered skill benchmark ──────────────────────────────────
        benchmark = get_dataset_skill_benchmark(
            category_info.get("predicted_category"), keywords
        )

        data = {
            "filename":           file.filename,
            "text":               extracted_text,
            "requirements":       keywords,
            "job_description":    requirements,
            "predicted_category": category_info.get("predicted_category"),
            "top_categories":     category_info.get("top_categories", []),
            "dataset_used":       category_info.get("dataset_used", False),
            "benchmark":          benchmark,
        }

        with open("uploaded_resume.json", "w", encoding="utf-8") as jf:
            json.dump(data, jf, indent=4)

        return {
            "message":              "Resume processed successfully",
            "pages_extracted":      len(pages),
            "keywords_extracted":   keywords,
            "requirements_count":   len(keywords),
            "predicted_category":   category_info.get("predicted_category"),
            "top_categories":       category_info.get("top_categories", []),
            "dataset_used":         category_info.get("dataset_used", False),
            "benchmark_skills":     benchmark.get("benchmark_skills", []),
            "category_sample_size": benchmark.get("category_sample_size", 0),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dataset-info")
async def dataset_info():
    """Return summary statistics about the loaded resume dataset."""
    df = get_resume_df()
    if df is None:
        return {
            "loaded": False,
            "message": f"Dataset not found at '{RESUME_CSV_PATH}'. "
                       "Place resume.csv in the project root or set RESUME_CSV_PATH env var.",
        }
    category_counts = df["category"].value_counts().to_dict()
    return {
        "loaded":          True,
        "total_resumes":   len(df),
        "total_categories": df["category"].nunique(),
        "category_counts": category_counts,
    }