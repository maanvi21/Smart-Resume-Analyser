import os
from dotenv import load_dotenv
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import APIRouter, HTTPException
import json
import re
import torch
import httpx
from pathlib import Path

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

print(f"Using device: {DEVICE}")
print(f"Gemini enabled: {bool(GEMINI_KEY)}")

router = APIRouter()

_sentence_model = None
_ner_pipeline   = None


def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        ).to(DEVICE)
    return _sentence_model


def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline
        _ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="first",
            device=0 if torch.cuda.is_available() else -1,
        )
    return _ner_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI FLASH
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1/models/"
    "gemini-2.5-flash:generateContent"
)


async def gemini(prompt: str, *, temperature: float = 0.3, max_tokens: int = 2048) -> str:
    if not GEMINI_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Export it before starting the server:\n"
            "  export GEMINI_API_KEY=your_key_here"
        )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":     temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            GEMINI_URL,
            params={"key": GEMINI_KEY},
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response shape: {data}") from e


async def gemini_json(prompt: str, **kwargs) -> dict | list:
    raw   = await gemini(prompt, **kwargs)
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean.strip())
    return json.loads(clean)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CONTEXT BUILDER
#
# For each missing skill, scan real resumes in the Kaggle dataset that belong
# to the same predicted category and extract sentences that mention that skill.
# These become grounding examples in the Gemini prompt, replacing the hollow
# "Add X to your resume" pattern with role-specific, evidence-backed advice.
# ─────────────────────────────────────────────────────────────────────────────

def _extract_skill_context_from_dataset(
    skill: str,
    category: str,
    max_snippets: int = 3,
    snippet_len: int = 300,
) -> list[str]:
    """
    Return up to `max_snippets` real lines from category-matched Kaggle resumes
    that mention `skill`. Lines are trimmed to `snippet_len` characters.
    """
    try:
        from routes.resume import get_resume_df
        df = get_resume_df()
        if df is None or df.empty:
            return []

        cat_df = df[df["category"].str.lower() == category.lower()]
        if cat_df.empty:
            return []

        pattern  = re.compile(re.escape(skill), re.IGNORECASE)
        snippets: list[str] = []

        for resume_text in cat_df["resume_str"].tolist():
            for line in resume_text.splitlines():
                line = line.strip()
                # Skip headers, section titles, and very short fragments
                if len(line) < 40:
                    continue
                if pattern.search(line):
                    snippets.append(line[:snippet_len])
                if len(snippets) >= max_snippets:
                    break
            if len(snippets) >= max_snippets:
                break

        return snippets
    except Exception as exc:
        print(f"[dataset_context:{skill}] {exc}")
        return []


def _build_skill_context_block(
    unmatched: list[str],
    category: str,
) -> dict[str, list[str]]:
    """
    Build a {skill: [snippet, ...]} map for every unmatched skill by scanning
    the Kaggle dataset. Used to ground Gemini's suggestions in real examples.
    """
    return {
        skill: _extract_skill_context_from_dataset(skill, category)
        for skill in unmatched
    }


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI TASKS
# ─────────────────────────────────────────────────────────────────────────────

async def gemini_classify_category(
    resume_text: str,
    semantic_top3: list[dict],
    job_description: str,
) -> dict:
    """
    Confirm or override the embedding model's category prediction.
    Returns {"category": str, "reasoning": str, "confidence": "high"|"medium"|"low"}
    """
    top3_str = "\n".join(
        f"  {i+1}. {x['category']} (embedding score: {x['confidence']})"
        for i, x in enumerate(semantic_top3)
    ) or "  (no dataset available)"

    prompt = f"""You are an expert recruiter and resume classifier.

Below are the top predicted job categories from an embedding model, along with the candidate's resume and the job description they are applying for.

Embedding model top predictions:
{top3_str}

JOB DESCRIPTION:
{job_description[:800]}

RESUME (excerpt):
{resume_text[:1500]}

Based on ALL of the above, determine the single most appropriate job category for this candidate.

Return ONLY valid JSON (no markdown):
{{
  "category": "<best matching category>",
  "reasoning": "<1-2 sentence explanation>",
  "confidence": "high" | "medium" | "low"
}}"""

    try:
        result = await gemini_json(prompt, temperature=0.1)
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        print(f"[gemini_classify] {exc}")
        return {
            "category":   semantic_top3[0]["category"] if semantic_top3 else "Unknown",
            "reasoning":  "Gemini classification unavailable; using embedding model.",
            "confidence": "low",
        }


async def gemini_rewrite_bullets(bullets: list[str], job_role: str) -> list[dict]:
    """
    Rewrite all detected resume bullets — no fixed cap.
    """
    if not bullets:
        return []

    bullets_json = json.dumps(bullets, ensure_ascii=False)
    prompt = f"""You are an expert resume writer helping a candidate apply for a {job_role} role.

For EVERY bullet below you MUST produce a genuinely rewritten version — never copy the original unchanged.

Rules:
1. Start with a strong past-tense action verb (replace weak openers like "Gained", "Worked on", "Helped").
2. Add a concrete result or metric if one is missing — infer a realistic one from context (e.g. "reducing load time by ~40%", "serving 200+ users").
3. If a metric already exists, keep it and sharpen the impact framing instead.
4. Complete any sentence that is cut off — do not leave trailing commas or dangling clauses.
5. Keep to 1-2 lines. Do not pad with filler words.
6. The improved version MUST be meaningfully different from the original — never return the original as-is.

Return ONLY valid JSON (no markdown), one object per bullet, same order as input:
[
  {{"original": "<exact original>", "improved": "<rewritten version>"}},
  ...
]

Bullets:
{bullets_json}"""

    try:
        result = await gemini_json(prompt, temperature=0.4, max_tokens=4096)
        if isinstance(result, list):
            # Drop entries where Gemini returned the original unchanged
            return [
                r for r in result
                if r.get("improved", "").strip() != r.get("original", "").strip()
                and r.get("improved", "").strip()
            ]
        return []
    except Exception as exc:
        print(f"[gemini_rewrite_bullets] {exc}")
        return []


async def gemini_suggest_skills(
    resume_text: str,
    unmatched: list[str],
    job_role: str,
    job_description: str,
    category: str,
) -> list[dict]:
    """
    Generate specific, grounded suggestions for genuinely missing skills.

    What makes this accurate:
    - Before calling Gemini, we scan the Kaggle dataset for real lines from
      resumes in the same category that mention each missing skill.
    - Those snippets are injected into the prompt so Gemini can see how real
      candidates in this exact field describe the skill — not generic advice.
    - Gemini is instructed to skip skills that aren't meaningfully absent,
      and to tie every suggestion back to what the candidate already has.
    """
    if not unmatched:
        return []

    # Pull real dataset evidence per skill before touching Gemini
    print(f"  Fetching dataset context for {len(unmatched)} unmatched skills…")
    skill_context = _build_skill_context_block(unmatched, category)

    # Format the per-skill evidence block for the prompt
    skill_lines: list[str] = []
    for skill in unmatched:
        snippets = skill_context.get(skill, [])
        if snippets:
            examples = "\n".join(f'      • "{s}"' for s in snippets)
            skill_lines.append(
                f"  - {skill}\n"
                f"    How real {category} candidates phrase this on their resumes:\n"
                f"{examples}"
            )
        else:
            # Skill isn't present in similar resumes at all — flag it
            skill_lines.append(
                f"  - {skill}\n"
                f"    (not commonly found in {category} resumes in the dataset)"
            )

    skills_block = "\n".join(skill_lines)

    prompt = f"""You are a senior career coach with access to a dataset of real {category} resumes.

A candidate is applying for a {job_role} role. The skills below are flagged as missing from their resume. For each skill, you have real examples of how other {category} professionals describe that skill on their resumes.

MISSING SKILLS (with real dataset examples):
{skills_block}

JOB DESCRIPTION (excerpt):
{job_description[:500]}

CANDIDATE'S RESUME (excerpt):
{resume_text[:1200]}

Instructions:
1. Only include a suggestion for skills that are GENUINELY absent AND meaningfully important for this specific role. Skip skills that are implied by what the candidate already has, rarely appear in the dataset for this category, or are peripheral to the role.
2. For each skill you keep, write a specific, actionable suggestion of 2–3 sentences:
   - Ground it in the dataset examples — recommend the kind of phrasing, project context, or tool combination that real {category} candidates actually use.
   - Tie it to the candidate's existing background where possible (e.g. "Since you have Django experience, adding a note about...").
   - Be concrete: name a specific project type, certification, or tool to add — never say "Add X to your resume" or "Make sure to mention".
3. If a skill is not found in the dataset for this category, only suggest it if the job description makes it clearly critical.

Return ONLY valid JSON (no markdown), one object per skill worth acting on:
[
  {{"skill": "<skill name>", "suggestion": "<specific, grounded, actionable advice>"}},
  ...
]"""

    try:
        result = await gemini_json(prompt, temperature=0.4, max_tokens=2048)
        if isinstance(result, list):
            return result
        return []
    except Exception as exc:
        print(f"[gemini_suggest_skills] {exc}")
        return [{"skill": s, "suggestion": f"Consider adding {s} to your resume."} for s in unmatched]


# ─────────────────────────────────────────────────────────────────────────────
# SYNONYM-AWARE SEMANTIC MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def synonym_aware_match(
    resume_text: str,
    requirements: list[str],
    keyword_expansions: dict[str, list[str]],
    predicted_category: str | None,
) -> dict:
    import numpy as np
    model = get_sentence_model()

    resume_emb = model.encode(
        resume_text[:4000], convert_to_numpy=True, normalize_embeddings=True
    )

    per_req: dict[str, float] = {}
    for req in requirements:
        variants = keyword_expansions.get(req, [req])
        phrases  = [req] + [f"{s} experience" for s in variants[1:]]
        var_embs = model.encode(
            phrases, convert_to_numpy=True, normalize_embeddings=True
        )
        scores       = (var_embs @ resume_emb).tolist()
        per_req[req] = round(float(max(scores)), 4)

    category_alignment     = 0.0
    category_bonus_applied = False

    if predicted_category:
        try:
            from routes.resume import get_resume_df
            df = get_resume_df()
            if df is not None and not df.empty:
                cat_texts = (
                    df[df["category"].str.lower() == predicted_category.lower()]["resume_str"]
                    .tolist()[:30]
                )
                if cat_texts:
                    cat_embs = model.encode(
                        [t[:1500] for t in cat_texts],
                        convert_to_numpy=True, normalize_embeddings=True,
                        batch_size=16, show_progress_bar=False,
                    )
                    centroid = cat_embs.mean(axis=0)
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
                    category_alignment     = float(np.dot(resume_emb, centroid))
                    category_bonus_applied = True
        except Exception as exc:
            print(f"[category_bonus] {exc}")

    threshold = 0.40
    matched   = [r for r, s in per_req.items() if s >= threshold]
    base_pct  = int(len(matched) / len(requirements) * 100) if requirements else 0
    bonus_pct = round(category_alignment * 10, 1) if category_bonus_applied else 0.0
    adj_pct   = min(100, int(base_pct + bonus_pct))

    return {
        "per_requirement_scores":    per_req,
        "matched_requirements":      matched,
        "unmatched_requirements":    [r for r in requirements if r not in matched],
        "semantic_match_percentage": base_pct,
        "adjusted_match_percentage": adj_pct,
        "category_alignment_score":  round(category_alignment, 4),
        "category_bonus_pct":        bonus_pct,
        "category_bonus_applied":    category_bonus_applied,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NER entity extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_entities(text: str) -> dict:
    ner    = get_ner_pipeline()
    chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
    all_ents: list[dict] = []
    for chunk in chunks:
        all_ents.extend(ner(chunk))

    buckets: dict[str, list[str]] = {}
    for ent in all_ents:
        group = ent.get("entity_group", "MISC")
        word  = re.sub(r'^#+', '', ent.get("word", "")).strip()
        if word and len(word) > 1 and not word.startswith('#'):
            buckets.setdefault(group, [])
            if word not in buckets[group]:
                buckets[group].append(word)
    return buckets


def find_similar_resumes(
    resume_text: str,
    predicted_category: str | None,
    top_n: int = 3,
) -> list[dict]:
    from routes.resume import get_resume_df
    import numpy as np
    df = get_resume_df()
    if df is None or df.empty or not predicted_category:
        return []
    try:
        model   = get_sentence_model()
        cat_df  = df[df["category"].str.lower() == predicted_category.lower()].head(100)
        if cat_df.empty:
            return []
        res_emb = model.encode(resume_text[:3000], convert_to_numpy=True, normalize_embeddings=True)
        texts   = cat_df["resume_str"].tolist()
        embs    = model.encode(
            [t[:1500] for t in texts], convert_to_numpy=True,
            normalize_embeddings=True, batch_size=16, show_progress_bar=False,
        )
        sims   = (embs @ res_emb).tolist()
        ranked = sorted(zip(sims, texts), key=lambda x: x[0], reverse=True)[:top_n]
        return [
            {"similarity": round(sim, 4),
             "snippet": text[:200].replace("\n", " ").strip() + "…"}
            for sim, text in ranked
        ]
    except Exception as exc:
        print(f"[similar_resumes] {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Bullet detection helper
# ─────────────────────────────────────────────────────────────────────────────

_ACTION_VERB_RE = re.compile(
    r"^(Developed|Built|Designed|Led|Managed|Created|Implemented|Improved|"
    r"Optimized|Increased|Reduced|Delivered|Worked|Collaborated|Maintained|"
    r"Deployed|Automated|Integrated|Architected|Established|Launched|Mentored|"
    r"Spearheaded|Engineered|Devised|Eliminated|Gained|Conducted|Analyzed|"
    r"Resolved|Migrated|Scaled|Streamlined|Revamped|Introduced|Directed)",
    re.IGNORECASE,
)


def _is_continuation(line: str) -> bool:
    """
    True if a line is a wrapped continuation of the previous bullet rather
    than a new bullet or section header.

    A line is a continuation when it:
    - does NOT start with a bullet symbol
    - does NOT start with an action verb (which would be a new bullet)
    - does NOT look like a section header (Title Case short line, all-caps, etc.)
    - is not empty
    """
    if not line:
        return False
    if line[0] in "-•*◦▪–":
        return False
    if _ACTION_VERB_RE.match(line):
        return False
    # Section headers: short Title Case or ALL CAPS lines
    if len(line) <= 40 and (line.istitle() or line.isupper()):
        return False
    # Lines that start with a digit are likely standalone (dates, rankings)
    if line[0].isdigit():
        return False
    return True


def detect_bullet_lines(resume_text: str) -> list[str]:
    """
    Extract complete bullet points from resume text, stitching wrapped
    continuation lines back onto the bullet they belong to.

    PDFs frequently wrap long bullets across multiple lines — the naive
    approach of treating each line independently sends truncated fragments
    to Gemini, which then has nothing to improve and returns them unchanged.
    """
    raw_lines = [l.strip() for l in resume_text.splitlines()]
    bullets: list[str] = []
    current: str | None = None

    for line in raw_lines:
        if not line:
            # Blank line always ends the current bullet
            if current:
                bullets.append(current)
                current = None
            continue

        is_bullet_start = (
            (len(line) > 1 and line[0] in "-•*◦▪–")
            or _ACTION_VERB_RE.match(line)
        )

        if is_bullet_start:
            # Save previous bullet before starting a new one
            if current:
                bullets.append(current)
            current = line.lstrip("-•*◦▪– ").strip()

        elif current is not None and _is_continuation(line):
            # Stitch the continuation onto the in-progress bullet
            # Use a space unless the previous fragment ends mid-word (rare)
            joiner = "" if current.endswith("-") else " "
            current = current.rstrip("-") + joiner + line

        else:
            # Not a bullet start and not a continuation — flush and skip
            if current:
                bullets.append(current)
                current = None

    if current:
        bullets.append(current)

    # Drop any bullet that is still obviously truncated (ends mid-sentence
    # with a comma and is very short) — these are section titles that leaked
    return [b for b in bullets if len(b) >= 30]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/nlp-module")
async def nlp_module():
    try:
        file_path   = Path("uploaded_resume.json").absolute()
        output_file = Path("nlp_results.json").absolute()

        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Resume data not found. Call /extract-features first."
            )

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        resume_text        = data.get("text", "").strip()
        requirements       = data.get("requirements", [])
        keyword_expansions = data.get("keyword_expansions", {})
        job_description    = data.get("job_description", "")
        semantic_top3      = data.get("top_categories", [])
        dataset_used       = data.get("dataset_used", False)
        benchmark          = data.get("benchmark", {})

        if not resume_text:
            raise HTTPException(status_code=400, detail="No text content in resume.")

        # ── 1. Gemini: classify category ──────────────────────────────────────
        print("Gemini: classifying job category…")
        gemini_cat     = await gemini_classify_category(resume_text, semantic_top3, job_description)
        final_cat      = gemini_cat.get("category") or (semantic_top3[0]["category"] if semantic_top3 else None)
        cat_reasoning  = gemini_cat.get("reasoning", "")
        cat_confidence = gemini_cat.get("confidence", "low")

        # ── 2. Synonym-aware semantic matching ────────────────────────────────
        print("Running synonym-aware semantic matching…")
        semantic = synonym_aware_match(
            resume_text, requirements, keyword_expansions, final_cat
        )

        # ── 3. NER entity extraction ──────────────────────────────────────────
        print("Running NER…")
        entities = extract_entities(resume_text)

        # ── 4. Gemini: rewrite all detected bullets ───────────────────────────
        bullets = detect_bullet_lines(resume_text)
        print(f"Gemini: rewriting {len(bullets)} bullets…")
        rewritten = await gemini_rewrite_bullets(bullets, final_cat or "software engineering")

        # ── 5. Dataset-grounded skill suggestions ─────────────────────────────
        # _build_skill_context_block scans the Kaggle CSV for real sentences
        # from same-category resumes that mention each missing skill, then
        # those snippets are injected into the Gemini prompt so suggestions
        # are grounded in how real candidates in this field write about it.
        print("Building dataset context for skill suggestions…")
        suggestions = await gemini_suggest_skills(
            resume_text,
            semantic["unmatched_requirements"],
            final_cat or "software engineering",
            job_description,
            category=final_cat or "",
        )

        # ── 6. Similar resumes from dataset ───────────────────────────────────
        print("Finding similar resumes in dataset…")
        similar = find_similar_resumes(resume_text, final_cat)

        display_pct = semantic.get("adjusted_match_percentage", semantic["semantic_match_percentage"])

        results = {
            "semantic_match_percentage": semantic["semantic_match_percentage"],
            "adjusted_match_percentage": semantic["adjusted_match_percentage"],
            "category_bonus_pct":        semantic["category_bonus_pct"],
            "category_bonus_applied":    semantic["category_bonus_applied"],
            "matched_requirements":      semantic["matched_requirements"],
            "unmatched_requirements":    semantic["unmatched_requirements"],
            "per_requirement_scores":    semantic["per_requirement_scores"],
            "suitable":                  display_pct >= 70,
            "entities":                  entities,
            "rewritten_bullets":         rewritten,
            "missing_skill_suggestions": suggestions,
            "predicted_category":        final_cat,
            "category_reasoning":        cat_reasoning,
            "category_confidence":       cat_confidence,
            "top_categories":            semantic_top3,
            "dataset_used":              dataset_used,
            "category_alignment_score":  semantic["category_alignment_score"],
            "benchmark_skills":          benchmark.get("benchmark_skills", []),
            "category_sample_size":      benchmark.get("category_sample_size", 0),
            "similar_resumes":           similar,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        print("NLP processing complete.")
        return results

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in NLP processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))