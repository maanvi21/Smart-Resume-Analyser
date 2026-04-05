import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import APIRouter, HTTPException
import json
import re
import torch
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

router = APIRouter()

# ── lazy-load heavy models once per worker ───────────────────────────────────
_sentence_model = None
_ner_pipeline   = None
_t5_pipeline    = None

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _sentence_model = _sentence_model.to(DEVICE)
    return _sentence_model

def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline
        _ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1,
        )
    return _ner_pipeline

def get_t5_pipeline():
    global _t5_pipeline
    if _t5_pipeline is None:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        model     = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(DEVICE)
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        _t5_pipeline = (model, tokenizer)
    return _t5_pipeline


# ── helpers ───────────────────────────────────────────────────────────────────

def semantic_match(resume_text: str, requirements: list[str]) -> dict:
    model = get_sentence_model()

    resume_emb = model.encode(resume_text, convert_to_numpy=True, normalize_embeddings=True)
    req_embs   = model.encode(requirements, convert_to_numpy=True, normalize_embeddings=True)

    scores  = (req_embs @ resume_emb).tolist()
    per_req = {req: round(float(s), 4) for req, s in zip(requirements, scores)}

    threshold = 0.40
    matched   = [r for r, s in per_req.items() if s >= threshold]
    pct       = int(len(matched) / len(requirements) * 100) if requirements else 0

    return {
        "per_requirement_scores":    per_req,
        "matched_requirements":      matched,
        "unmatched_requirements":    [r for r in requirements if r not in matched],
        "semantic_match_percentage": pct,
    }


def dataset_calibrated_match(
    resume_text: str,
    requirements: list[str],
    predicted_category: str | None,
) -> dict:
    """
    Augments the basic semantic match with a dataset-aware calibration:

    1. Pull the base cosine scores per requirement.
    2. If a predicted category is available, compute a 'category alignment bonus'
       by comparing the resume to the average embedding of resumes in that category
       from the dataset.  This nudges the overall score to reflect how well the
       candidate fits the role family—not just keyword overlap.
    3. Return adjusted scores alongside the raw ones.
    """
    from routes.resume import get_resume_df   # shared cache
    import numpy as np

    model = get_sentence_model()

    resume_emb = model.encode(resume_text[:3000], convert_to_numpy=True, normalize_embeddings=True)
    req_embs   = model.encode(requirements, convert_to_numpy=True, normalize_embeddings=True)
    raw_scores = (req_embs @ resume_emb).tolist()
    per_req    = {req: round(float(s), 4) for req, s in zip(requirements, raw_scores)}

    # ── Category alignment bonus (0–10 %) ────────────────────────────────────
    category_alignment = 0.0
    category_bonus_applied = False

    if predicted_category:
        df = get_resume_df()
        if df is not None and not df.empty:
            cat_texts = df[df["category"].str.lower() == predicted_category.lower()][
                "resume_str"
            ].tolist()[:30]
            if cat_texts:
                cat_embs = model.encode(
                    [t[:1500] for t in cat_texts],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=16,
                    show_progress_bar=False,
                )
                centroid = cat_embs.mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
                category_alignment = float(np.dot(resume_emb, centroid))
                category_bonus_applied = True

    # Bonus: up to +10 % of the raw match percentage
    bonus_pct = round(category_alignment * 10, 1) if category_bonus_applied else 0.0

    threshold = 0.40
    matched   = [r for r, s in per_req.items() if s >= threshold]
    base_pct  = int(len(matched) / len(requirements) * 100) if requirements else 0
    adjusted_pct = min(100, int(base_pct + bonus_pct))

    return {
        "per_requirement_scores":      per_req,
        "matched_requirements":        matched,
        "unmatched_requirements":      [r for r in requirements if r not in matched],
        "semantic_match_percentage":   base_pct,
        "adjusted_match_percentage":   adjusted_pct,
        "category_alignment_score":    round(category_alignment, 4),
        "category_bonus_pct":          bonus_pct,
        "category_bonus_applied":      category_bonus_applied,
    }


def extract_entities(text: str) -> dict:
    ner       = get_ner_pipeline()
    MAX_CHARS = 2000
    chunks    = [text[i:i + MAX_CHARS] for i in range(0, len(text), MAX_CHARS)]

    all_entities: list[dict] = []
    for chunk in chunks:
        all_entities.extend(ner(chunk))

    buckets: dict[str, list[str]] = {}
    for ent in all_entities:
        group = ent.get("entity_group", "MISC")
        word  = ent.get("word", "").strip()
        if word:
            buckets.setdefault(group, [])
            if word not in buckets[group]:
                buckets[group].append(word)

    return buckets


def run_t5(prompt: str, max_new_tokens: int = 80) -> str:
    model, tokenizer = get_t5_pipeline()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def rewrite_bullets(bullets: list[str]) -> list[dict]:
    results = []
    for bullet in bullets[:10]:
        prompt = (
            "Rewrite the following resume bullet point to be more impactful, "
            "quantified, and professional:\n\n" + bullet
        )
        output = run_t5(prompt, max_new_tokens=80)
        results.append({"original": bullet, "improved": output.strip()})
    return results


def suggest_missing_skills(
    resume_text: str,
    requirements: list[str],
    unmatched: list[str],
    predicted_category: str | None = None,
) -> list[dict]:
    """
    Generate actionable suggestions for unmatched JD skills.
    When a predicted category is available, the T5 prompt is enriched with
    category context so the advice is more role-specific.
    """
    if not unmatched:
        return []

    category_ctx = f" for a {predicted_category} role" if predicted_category else ""

    suggestions = []
    for req in unmatched[:5]:
        prompt = (
            f"A software job{category_ctx} requires: {req}. "
            f"The candidate does not mention {req} in their resume. "
            f"Give one specific, actionable suggestion to add {req} experience to a resume. "
            "Be concrete and brief. Do not say 'make sure' or 'build a strong resume'."
        )
        out = run_t5(prompt, max_new_tokens=80)
        suggestions.append({"skill": req, "suggestion": out.strip()})
    return suggestions


def find_similar_resumes_in_dataset(resume_text: str, predicted_category: str | None, top_n: int = 3) -> list[dict]:
    """
    Find the most similar resumes in the dataset to the uploaded one.
    Returns anonymised snippets (first 200 chars) and similarity scores.
    Useful for the frontend to show 'resumes in this category typically highlight…'
    """
    from routes.resume import get_resume_df
    import numpy as np

    df = get_resume_df()
    if df is None or df.empty or not predicted_category:
        return []

    try:
        model = get_sentence_model()
        cat_df = df[df["category"].str.lower() == predicted_category.lower()].head(100)
        if cat_df.empty:
            return []

        resume_emb = model.encode(
            resume_text[:3000], convert_to_numpy=True, normalize_embeddings=True
        )
        texts = cat_df["resume_str"].tolist()
        embs  = model.encode(
            [t[:1500] for t in texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        )
        sims = (embs @ resume_emb).tolist()
        ranked = sorted(zip(sims, texts), key=lambda x: x[0], reverse=True)[:top_n]

        return [
            {
                "similarity": round(sim, 4),
                "snippet":    text[:200].replace("\n", " ").strip() + "…",
            }
            for sim, text in ranked
        ]
    except Exception as exc:
        print(f"[similar_resumes] Error: {exc}")
        return []


# ── endpoint ──────────────────────────────────────────────────────────────────

@router.get("/nlp-module")
async def nlp_module():
    try:
        file_path   = Path("uploaded_resume.json").absolute()
        output_file = Path("nlp_results.json").absolute()

        print("Starting enhanced NLP processing")
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Resume data file not found")

        with open(file_path, "r", encoding="utf-8") as f:
            resume_data = json.load(f)

        resume_text        = resume_data.get("text", "").strip()
        requirements       = resume_data.get("requirements", [])
        predicted_category = resume_data.get("predicted_category")   # injected by /extract-features
        top_categories     = resume_data.get("top_categories", [])
        dataset_used       = resume_data.get("dataset_used", False)
        benchmark          = resume_data.get("benchmark", {})

        if not resume_text:
            raise HTTPException(status_code=400, detail="No text content found in the resume")

        # ── Semantic matching (dataset-calibrated if category is known) ───────
        print("Running dataset-calibrated semantic matching...")
        semantic = dataset_calibrated_match(resume_text, requirements, predicted_category)

        print("Running NER...")
        entities = extract_entities(resume_text)

        # Detect bullet lines
        all_lines = [line.strip() for line in resume_text.splitlines()]
        bullet_lines = []
        for line in all_lines:
            if len(line) < 20:
                continue
            if line[0] in "-•*◦▪–":
                bullet_lines.append(line.lstrip("-•*◦▪– ").strip())
            elif re.match(
                r"^(Developed|Built|Designed|Led|Managed|Created|Implemented|"
                r"Improved|Optimized|Increased|Reduced|Delivered|Worked|"
                r"Collaborated|Maintained|Deployed|Automated|Integrated|"
                r"Architected|Established|Launched|Mentored|Spearheaded)", line
            ):
                bullet_lines.append(line)

        print(f"Rewriting {len(bullet_lines[:10])} bullets...")
        rewritten = rewrite_bullets(bullet_lines)

        print("Generating suggestions for unmatched requirements...")
        suggestions = suggest_missing_skills(
            resume_text,
            requirements,
            semantic["unmatched_requirements"],
            predicted_category=predicted_category,
        )

        # ── Dataset: find similar resumes in predicted category ───────────────
        print("Finding similar resumes in dataset...")
        similar_resumes = find_similar_resumes_in_dataset(resume_text, predicted_category)

        # Use adjusted_match_percentage for the suitability gate when available
        display_pct = semantic.get("adjusted_match_percentage", semantic["semantic_match_percentage"])

        results = {
            # ── Core match info ──────────────────────────────────────────────
            "semantic_match_percentage":  semantic["semantic_match_percentage"],
            "adjusted_match_percentage":  semantic.get("adjusted_match_percentage"),
            "category_bonus_pct":         semantic.get("category_bonus_pct", 0),
            "category_bonus_applied":     semantic.get("category_bonus_applied", False),
            "matched_requirements":       semantic["matched_requirements"],
            "unmatched_requirements":     semantic["unmatched_requirements"],
            "per_requirement_scores":     semantic["per_requirement_scores"],
            "suitable":                   display_pct >= 70,

            # ── NER entities ─────────────────────────────────────────────────
            "entities":                   entities,

            # ── AI rewrites & suggestions ────────────────────────────────────
            "rewritten_bullets":          rewritten,
            "missing_skill_suggestions":  suggestions,

            # ── Dataset-powered insights ──────────────────────────────────────
            "predicted_category":         predicted_category,
            "top_categories":             top_categories,
            "dataset_used":               dataset_used,
            "category_alignment_score":   semantic.get("category_alignment_score", 0),
            "benchmark_skills":           benchmark.get("benchmark_skills", []),
            "category_sample_size":       benchmark.get("category_sample_size", 0),
            "similar_resumes":            similar_resumes,
        }

        with open(output_file, "w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=4)

        print("NLP processing complete.")
        return results

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in NLP processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))