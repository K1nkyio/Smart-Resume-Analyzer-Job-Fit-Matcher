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
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "sentence-transformers is not installed. Install it (pip install sentence-transformers) "
            "or rely on the TF-IDF fallback by keeping semantic scoring enabled."
        ) from e

    return SentenceTransformer(model_name)


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
    mat = vectorizer.fit_transform([resume, job])
    sim = float((mat[0] @ mat[1].T).A[0][0])
    return round(max(0.0, min(100.0, sim * 100.0)), 2)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return np.matmul(a, b.T)


def semantic_match_score(resume_text: str, job_text: str, model_name: str = DEFAULT_EMBEDDINGS_MODEL) -> float:
    resume = (resume_text or "").strip()
    job = (job_text or "").strip()
    if not resume or not job:
        return 0.0

    try:
        vecs = embed_texts([resume, job], model_name=model_name)
        sim = float(np.matmul(vecs[0], vecs[1].T))
        return round(max(0.0, min(100.0, sim * 100.0)), 2)
    except ModuleNotFoundError:
        return _tfidf_similarity_score(resume, job)


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json_object(text: str) -> dict:
    match = _JSON_RE.search(text or "")
    if not match:
        return {}
    blob = match.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return {}


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
        "Do NOT invent experience. Keep facts unchanged. Return ONLY valid JSON list.\n\n"
        "Schema: [ {\"issue\": string, \"recommendation\": string, \"example_rewrite\": string} ]\n\n"
        f"JOB DESCRIPTION:\n{job[:6000]}\n\n"
        f"RESUME:\n{resume[:12000]}\n"
    )
    output = ollama_generate(prompt, settings=settings)
    if not output:
        return []
    data = _extract_json_object(output)
    if isinstance(data, list):
        return data
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
