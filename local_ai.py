import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_EMBEDDINGS_MODEL = os.getenv("LOCAL_AI_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_OLLAMA_MODEL = os.getenv("LOCAL_AI_OLLAMA_MODEL", "llama3.1")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


@dataclass(frozen=True)
class LocalAISettings:
    embeddings_model: str = DEFAULT_EMBEDDINGS_MODEL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_URL
    request_timeout_seconds: int = 30


def _json_post(url: str, payload: dict, timeout_seconds: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to call local Ollama server at {url}: {e}") from e


def is_ollama_available(settings: Optional[LocalAISettings] = None) -> bool:
    cfg = settings or LocalAISettings()
    payload = {"model": cfg.ollama_model}
    try:
        _json_post(f"{cfg.ollama_base_url}/api/show", payload, timeout_seconds=min(5, cfg.request_timeout_seconds))
        return True
    except Exception:
        return False


def ollama_generate(prompt: str, settings: Optional[LocalAISettings] = None) -> str:
    cfg = settings or LocalAISettings()
    payload = {
        "model": cfg.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    try:
        result = _json_post(f"{cfg.ollama_base_url}/api/generate", payload, cfg.request_timeout_seconds)
    except ConnectionError:
        return ""
    text = result.get("response", "")
    return (text or "").strip()


@lru_cache(maxsize=1)
def _get_sentence_transformer(model_name: str):
    try:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "sentence-transformers is not installed. Install it (pip install sentence-transformers) "
            "or rely on the TF-IDF fallback by keeping semantic scoring enabled."
        ) from e

    model_path = model_name
    if not os.path.isdir(model_name):
        try:
            model_path = snapshot_download(model_name, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Embedding model '{model_name}' is not cached locally. Download it separately with user approval, then retry."
            ) from exc
    return SentenceTransformer(model_path)


def embed_texts(texts: List[str], model_name: str = DEFAULT_EMBEDDINGS_MODEL) -> np.ndarray:
    model = _get_sentence_transformer(model_name)
    embeddings = model.encode(texts or [""], normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def _tfidf_similarity_score(resume_text: str, job_text: str) -> float:
    resume = (resume_text or "").strip()
    job = (job_text or "").strip()
    if not resume or not job:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
    try:
        mat = vectorizer.fit_transform([resume, job])
    except ValueError:
        return 0.0
    sim = float((mat[0] @ mat[1].T).toarray()[0][0])
    return round(max(0.0, min(100.0, sim * 100.0)), 2)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return np.matmul(a, b.T)


def semantic_match_score(
    resume_text: str,
    job_text: str,
    model_name: str = DEFAULT_EMBEDDINGS_MODEL,
    use_embeddings: bool = False,
) -> float:
    resume = (resume_text or "").strip()
    job = (job_text or "").strip()
    if not resume or not job:
        return 0.0

    if not use_embeddings:
        return _tfidf_similarity_score(resume, job)

    try:
        vecs = embed_texts([resume, job], model_name=model_name)
        sim = float(np.matmul(vecs[0], vecs[1].T))
        return round(max(0.0, min(100.0, sim * 100.0)), 2)
    except ModuleNotFoundError:
        return _tfidf_similarity_score(resume, job)


def _extract_json_value(text: str):
    """Parse a JSON object or list from a model response, including code fences."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start_positions = [pos for pos in (cleaned.find("{"), cleaned.find("[")) if pos >= 0]
    if not start_positions:
        return None
    start = min(start_positions)
    for end in range(len(cleaned), start, -1):
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            continue
    return None


def _extract_json_object(text: str) -> dict:
    value = _extract_json_value(text)
    return value if isinstance(value, dict) else {}


def extract_structured_profile(resume_text: str, job_text: str, settings: Optional[LocalAISettings] = None) -> dict:
    resume = (resume_text or "").strip()
    job = (job_text or "").strip()
    if not resume:
        return {}

    prompt = (
        "You are an expert resume analyst. Extract a structured profile from the resume and job description. "
        "Return ONLY valid JSON, no markdown.\n\n"
        "Schema:\n"
        "{\n"
        "  \"profession\": string,\n"
        "  \"target_role\": string,\n"
        "  \"seniority\": string,\n"
        "  \"skills\": [string],\n"
        "  \"tools\": [string],\n"
        "  \"certifications\": [string],\n"
        "  \"education\": [string],\n"
        "  \"experience_highlights\": [string],\n"
        "  \"keywords_from_job\": [string]\n"
        "}\n\n"
        f"JOB DESCRIPTION:\n{job[:6000]}\n\n"
        f"RESUME:\n{resume[:12000]}\n"
    )
    output = ollama_generate(prompt, settings=settings)
    if not output:
        return {}
    return _extract_json_object(output)


def generate_gap_analysis(structured: dict) -> dict:
    skills = set([s.strip() for s in structured.get("skills", []) if isinstance(s, str) and s.strip()])
    job_keywords = set(
        [s.strip() for s in structured.get("keywords_from_job", []) if isinstance(s, str) and s.strip()]
    )
    missing = sorted(job_keywords - skills)
    return {
        "profession": structured.get("profession", ""),
        "target_role": structured.get("target_role", ""),
        "seniority": structured.get("seniority", ""),
        "missing_keywords": missing[:25],
    }


def generate_tailored_resume_suggestions(
    resume_text: str,
    job_text: str,
    settings: Optional[LocalAISettings] = None,
) -> List[dict]:
    resume = (resume_text or "").strip()
    job = (job_text or "").strip()
    if not resume or not job:
        return []

    prompt = (
        "You are a resume coach. Suggest improvements tailored to the job description. "
        "Do NOT invent experience, metrics, qualifications, tools, or outcomes. Keep facts unchanged. "
        "Only suggest rewriting exact text quoted from the resume; if no safe rewrite is possible, return an empty list. Return ONLY valid JSON list.\n\n"
        "Schema: [ {\"issue\": string, \"recommendation\": string, \"section\": string, \"original_text\": string, \"example_rewrite\": string} ]\n\n"
        f"JOB DESCRIPTION:\n{job[:6000]}\n\n"
        f"RESUME:\n{resume[:12000]}\n"
    )
    output = ollama_generate(prompt, settings=settings)
    if not output:
        return []
    data = _extract_json_value(output)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)][:8]
    return []


def generate_cover_letter(
    resume_text: str,
    job_text: str,
    company: str = "",
    settings: Optional[LocalAISettings] = None,
) -> str:
    resume = (resume_text or "").strip()
    job = (job_text or "").strip()
    if not resume or not job:
        return ""

    prompt = (
        "Write a professional cover letter based on the resume and job description. "
        "Do NOT invent employers, degrees, or achievements. Use a concise tone.\n\n"
        f"Company: {company or 'the company'}\n\n"
        f"JOB DESCRIPTION:\n{job[:6000]}\n\n"
        f"RESUME:\n{resume[:12000]}\n"
    )
    output = ollama_generate(prompt, settings=settings)
    return output or ""


def generate_career_coach_response(
    question: str,
    analysis: dict,
    settings: Optional[LocalAISettings] = None,
) -> str:
    """Answer from a compact analysis summary, not the raw resume document."""
    skill_details = analysis.get("skill_details", {})
    context = {
        "target_role": skill_details.get("target_role_context", ""),
        "job_fit_score": analysis.get("overall_score"),
        "resume_quality_score": analysis.get("resume_quality", {}).get("score"),
        "matched_qualifications": [item.get("skill") for item in skill_details.get("matched_details", [])],
        "unmatched_qualifications": [item.get("skill") for item in skill_details.get("missing_details", [])],
        "related_but_unconfirmed_skills": {
            item.get("skill"): item.get("related_resume_skills", [])
            for item in skill_details.get("missing_details", [])
            if item.get("related_resume_skills")
        },
        "other_qualification_evidence": {
            category: [
                {"requirement": item.get("requirement"), "status": item.get("status")}
                for item in analysis.get("qualification_evidence", {}).get(category, [])
            ]
            for category in ("education", "experience", "certifications")
        },
        "resume_quality_recommendations": [item.get("action") for item in analysis.get("resume_quality", {}).get("recommendations", [])],
        "job_recommendations": analysis.get("ats_details", {}).get("suggestions", [])[:8],
    }
    prompt = (
        "You are a practical career coach. Answer using only the supplied analysis context. "
        "Do not infer experience, proficiency, achievements, or qualifications that are absent. Treat related technologies as transferable context only, never as direct proficiency. "
        "Treat experience duration as self-reported and credentials as unverified. "
        "State when context is insufficient, distinguish suggestions from confirmed evidence, "
        "and give a concise next step. Do not describe scores as hiring probabilities.\n\n"
        f"ANALYSIS CONTEXT:\n{json.dumps(context, ensure_ascii=False)}\n\n"
        f"USER QUESTION:\n{(question or '').strip()[:1500]}\n"
    )
    return ollama_generate(prompt, settings=settings)
