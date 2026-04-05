from fastapi import APIRouter, File, Form, UploadFile, HTTPException
import json
import re
import pdfplumber
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

router = APIRouter()

RESUME_CSV_PATH = os.environ.get("RESUME_CSV_PATH", "resume.csv")
_resume_df: pd.DataFrame | None = None


def get_resume_df() -> pd.DataFrame | None:
    global _resume_df
    if _resume_df is not None:
        return _resume_df
    if not os.path.exists(RESUME_CSV_PATH):
        return None
    try:
        df = pd.read_csv(RESUME_CSV_PATH)
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"resume_str", "category"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain: {required}. Found: {list(df.columns)}")
        df = df[["resume_str", "category"]].dropna(subset=["resume_str", "category"])
        df["resume_str"] = df["resume_str"].astype(str).str.strip()
        df["category"]   = df["category"].astype(str).str.strip()
        _resume_df = df
        print(f"[dataset] Loaded {len(df):,} resumes, {df['category'].nunique()} categories.")
        return _resume_df
    except Exception as exc:
        print(f"[dataset] Failed to load CSV: {exc}")
        return None


# ── Strict skill-only regex — no sentence fragments ───────────────────────────
TECH_PATTERN = re.compile(
    r'\b('
    r'python|javascript|typescript|java|c\+\+|c#|ruby|golang|go|rust|swift|'
    r'kotlin|php|scala|perl|matlab|bash|shell|powershell|'
    r'react(?:\.js)?|angular(?:js)?|vue(?:\.js)?|next\.js|nuxt(?:\.js)?|svelte|'
    r'html5?|css3?|sass|scss|tailwind(?:css)?|bootstrap|jquery|'
    r'node(?:\.js)?|express(?:\.js)?|django|flask|fastapi|spring(?:boot)?|'
    r'laravel|rails|asp\.net|graphql|grpc|'
    r'postgresql|postgres|mysql|mongodb|redis|sqlite|cassandra|dynamodb|'
    r'firebase|elasticsearch|neo4j|mariadb|mssql|'
    r'docker|kubernetes|k8s|aws|azure|gcp|terraform|ansible|jenkins|'
    r'gitlab|helm|nginx|linux|unix|'
    r'tensorflow|pytorch|keras|scikit-learn|pandas|numpy|matplotlib|'
    r'spark|hadoop|airflow|dbt|tableau|'
    r'selenium|cypress|jest|pytest|junit|mocha|postman|jmeter|'
    r'agile|scrum|kanban|jira|confluence|git|svn|'
    r'microservices|devops|devsecops|'
    r'sql|nosql|api|sdk|rest|restful|qa|sdet|tdd|bdd|'
    r'machine\s?learning|deep\s?learning|computer\s?vision|'
    r'system\s?design|data\s?structures|algorithms|'
    r'unit\s?testing|integration\s?testing|'
    r'github(?:\s?actions)?|power\s?bi|ci/cd|'
    r')'
    r'\b',
    re.IGNORECASE
)

BLOCKLIST = {
    '##k', '##s', '##d', '##r', '##n', '##m', '##t', '##l', '##e', '##i',
    '##a', '##b', '##c', '##f', '##g', '##h', '##j', '##o', '##p', '##q',
    '##u', '##v', '##w', '##x', '##y', '##z',
    'ji', 'sc', 'sd', 'devo', 'opskube', 'applications', 'services',
    'education', 'technology', 'computer science and technology',
    'quality assurance and testing education', 'software & q',
    'and', 'or', 'the', 'with', 'for', 'a', 'an', 'in', 'of', 'to',
    'is', 'are', 'we', 'you', 'our', 'your', 'its', 'be', 'will', 'can',
    'this', 'that', 'from', 'as', 'on', 'at', 'by', 'it', 'if', 'not',
    'but', 'also', 'so', 'do', 'use', 'used', 'using', 'based', 'strong',
    'experience', 'knowledge', 'skills', 'ability', 'understanding',
    'good', 'great', 'solid', 'excellent', 'proven', 'required', 'preferred',
}

_SUBWORD_RE = re.compile(r'^#')
_SYMBOL_RE  = re.compile(r'[&%@$]')


def _is_valid_skill(token: str) -> bool:
    t = token.strip()
    if not t or len(t) < 2:
        return False
    if _SUBWORD_RE.match(t):
        return False
    if t.lower() in BLOCKLIST:
        return False
    if t.isdigit():
        return False
    if len(t.split()) > 3:
        return False
    if _SYMBOL_RE.search(t):
        return False
    return True


def extract_skills_from_jd(jd_text: str) -> list[str]:
    found: set[str] = set()
    for match in TECH_PATTERN.finditer(jd_text):
        skill = match.group(0).lower().strip()
        if _is_valid_skill(skill):
            found.add(skill)
    try:
        from transformers import pipeline as hf_pipeline
        ner = hf_pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="first",
        )
        for chunk in [jd_text[i:i + 2000] for i in range(0, len(jd_text), 2000)]:
            for ent in ner(chunk):
                word  = re.sub(r'^#+', '', ent.get("word", "")).strip()
                group = ent.get("entity_group", "")
                if (group in ("ORG", "MISC")
                        and _is_valid_skill(word)
                        and len(word.split()) <= 2
                        and word.lower() not in found):
                    found.add(word.lower())
    except Exception as exc:
        print(f"[NER] Skipped: {exc}")
    return sorted(k for k in found if _is_valid_skill(k))


SKILL_SYNONYMS: dict[str, list[str]] = {
    "computer science":        ["computer engineering", "software engineering",
                                "information technology", "cs", "btech", "b.tech",
                                "information systems", "cse"],
    "computer engineering":    ["computer science", "software engineering",
                                "electronics", "ece", "btech", "b.tech", "cse"],
    "software engineering":    ["computer science", "computer engineering",
                                "software development", "swe", "cse"],
    "information technology":  ["computer science", "computer engineering",
                                "it", "information systems"],
    "machine learning":        ["ml", "ai", "artificial intelligence",
                                "deep learning", "data science", "tensorflow", "pytorch"],
    "deep learning":           ["machine learning", "neural networks", "ai",
                                "tensorflow", "pytorch", "keras"],
    "data science":            ["machine learning", "statistics", "data analysis",
                                "analytics", "python", "r"],
    "devops":                  ["sre", "platform engineering", "ci/cd",
                                "infrastructure", "docker", "kubernetes"],
    "rest":                    ["restful", "rest api", "api", "http", "web services"],
    "restful":                 ["rest", "api", "http", "web services"],
    "node.js":                 ["node", "nodejs", "express", "javascript"],
    "react":                   ["react.js", "reactjs", "react native", "frontend"],
    "kubernetes":              ["k8s", "container orchestration", "docker", "helm"],
    "postgresql":              ["postgres", "sql", "rdbms", "relational database"],
    "mysql":                   ["sql", "rdbms", "mariadb", "relational database"],
    "mongodb":                 ["nosql", "document database", "mongo"],
    "aws":                     ["amazon web services", "cloud", "ec2", "s3", "lambda", "gcp", "azure"],
    "azure":                   ["microsoft azure", "cloud", "aws", "gcp"],
    "gcp":                     ["google cloud", "cloud", "aws", "azure", "bigquery"],
    "agile":                   ["scrum", "kanban", "sprint", "agile methodology"],
    "scrum":                   ["agile", "kanban", "sprint"],
    "qa":                      ["quality assurance", "testing", "sdet", "tdd", "bdd", "selenium"],
    "sdet":                    ["qa", "test automation", "selenium", "quality assurance", "pytest"],
    "ci/cd":                   ["continuous integration", "continuous deployment",
                                "jenkins", "github actions", "gitlab", "devops"],
    "java":                    ["jvm", "spring", "springboot", "kotlin", "backend"],
    "python":                  ["django", "flask", "fastapi", "pandas", "numpy", "scripting"],
    "javascript":              ["typescript", "js", "node.js", "react", "vue", "angular", "frontend"],
    "typescript":              ["javascript", "ts", "angular", "react"],
    "git":                     ["github", "gitlab", "version control", "svn"],
    "docker":                  ["containerization", "containers", "kubernetes", "devops"],
    "sql":                     ["postgresql", "mysql", "sqlite", "mssql", "database", "rdbms"],
    "linux":                   ["unix", "bash", "shell", "ubuntu", "centos", "os"],
    "system design":           ["architecture", "distributed systems", "scalability", "microservices"],
    "data structures":         ["algorithms", "dsa", "competitive programming"],
    "algorithms":              ["data structures", "dsa"],
    "jira":                    ["agile", "project management", "confluence", "atlassian"],
    "microservices":           ["distributed systems", "system design", "docker", "kubernetes", "api"],
    "api":                     ["rest", "restful", "graphql", "grpc", "web services", "sdk"],
    "tdd":                     ["test driven development", "unit testing", "pytest", "jest", "qa"],
    "bdd":                     ["behavior driven", "cucumber", "selenium", "qa", "testing"],
}


def expand_with_synonyms(keyword: str) -> list[str]:
    kw = keyword.lower()
    return list(dict.fromkeys([kw] + [s.lower() for s in SKILL_SYNONYMS.get(kw, [])]))


def predict_category_semantic(resume_text: str) -> dict:
    df = get_resume_df()
    if df is None or df.empty:
        return {"predicted_category": None, "top_categories": [], "dataset_used": False}
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model      = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        resume_emb = model.encode(resume_text[:3000], convert_to_numpy=True, normalize_embeddings=True)
        cat_scores: dict[str, float] = {}
        for cat in df["category"].unique():
            texts    = df[df["category"] == cat]["resume_str"].tolist()[:30]
            embs     = model.encode([t[:1500] for t in texts], convert_to_numpy=True,
                                    normalize_embeddings=True, batch_size=16, show_progress_bar=False)
            centroid = embs.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            cat_scores[cat] = float(np.dot(resume_emb, centroid))
        top3 = [
            {"category": c, "confidence": round(s, 4)}
            for c, s in sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        ]
        return {"predicted_category": top3[0]["category"] if top3 else None,
                "top_categories": top3, "dataset_used": True}
    except Exception as exc:
        print(f"[predict_category_semantic] {exc}")
        return {"predicted_category": None, "top_categories": [], "dataset_used": False}


def get_dataset_skill_benchmark(category: str | None, keywords: list[str]) -> dict:
    df = get_resume_df()
    if df is None or df.empty or not category:
        return {"benchmark_skills": [], "category_sample_size": 0}
    cat_df = df[df["category"].str.lower() == category.lower()].head(200)
    if cat_df.empty:
        return {"benchmark_skills": [], "category_sample_size": 0}
    counts: dict[str, int] = {}
    for text in cat_df["resume_str"]:
        for m in TECH_PATTERN.finditer(str(text)):
            sk = m.group(0).lower().strip()
            if _is_valid_skill(sk):
                counts[sk] = counts.get(sk, 0) + 1
    total  = len(cat_df)
    jd_set = {k.lower() for k in keywords}
    result = [
        {"skill": sk, "frequency_pct": round(cnt / total * 100, 1)}
        for sk, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if sk not in jd_set and cnt / total >= 0.30
    ][:10]
    return {"benchmark_skills": result, "category_sample_size": total}


def _extract_text_from_pdf(file) -> tuple[str, int]:
    """
    Extract text from a PDF preserving complete lines.

    pdfplumber's default extract_text() wraps long lines at the PDF's
    internal line breaks, so a single bullet point spanning two visual
    lines arrives as two separate strings. We reconstruct full lines by
    grouping words that share the same vertical (y) position using their
    bounding boxes, then join them left-to-right. This ensures bullets
    are complete before they reach the NLP pipeline.
    """
    page_texts: list[str] = []

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            words = page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=False,
            )
            if not words:
                continue

            # Group words by their top-edge y coordinate (rounded to 1dp
            # to absorb sub-pixel variation between words on the same line)
            line_map: dict[float, list[dict]] = {}
            for w in words:
                y = round(float(w["top"]), 1)
                line_map.setdefault(y, []).append(w)

            # Sort lines top-to-bottom; within each line sort words left-to-right
            sorted_lines = [
                " ".join(w["text"] for w in sorted(ws, key=lambda w: w["x0"]))
                for _, ws in sorted(line_map.items())
            ]
            page_texts.append("\n".join(sorted_lines))

    text = "\n".join(page_texts)
    return text, len(page_texts)


@router.post("/extract-features")
async def extract_features(file: UploadFile = File(...), requirements: str = Form(...)):
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDFs are accepted.")

        text, num_pages = _extract_text_from_pdf(file.file)

        if not text.strip():
            raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

        keywords = extract_skills_from_jd(requirements)
        if not keywords:
            raise HTTPException(status_code=422, detail="No skills found in job description.")

        keyword_expansions = {kw: expand_with_synonyms(kw) for kw in keywords}

        category_info = predict_category_semantic(text)
        benchmark     = get_dataset_skill_benchmark(category_info.get("predicted_category"), keywords)

        data = {
            "filename":          file.filename,
            "text":              text,
            "requirements":      keywords,
            "keyword_expansions": keyword_expansions,
            "job_description":   requirements,
            "predicted_category": category_info.get("predicted_category"),
            "top_categories":    category_info.get("top_categories", []),
            "dataset_used":      category_info.get("dataset_used", False),
            "benchmark":         benchmark,
        }
        with open("uploaded_resume.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        return {
            "message":            "Resume processed successfully",
            "pages_extracted":    num_pages,
            "keywords_extracted": keywords,
            "requirements_count": len(keywords),
            "predicted_category": category_info.get("predicted_category"),
            "top_categories":     category_info.get("top_categories", []),
            "dataset_used":       category_info.get("dataset_used", False),
            "benchmark_skills":   benchmark.get("benchmark_skills", []),
            "category_sample_size": benchmark.get("category_sample_size", 0),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dataset-info")
async def dataset_info():
    df = get_resume_df()
    if df is None:
        return {"loaded": False, "message": f"Dataset not found at '{RESUME_CSV_PATH}'."}
    return {
        "loaded":           True,
        "total_resumes":    len(df),
        "total_categories": df["category"].nunique(),
        "category_counts":  df["category"].value_counts().to_dict(),
    }