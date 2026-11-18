"""Evidence-led resume document quality checks, independent of a job posting."""

import re
from typing import Any, Dict, List, Optional


# Product owners can tune these weights. Optional dimensions are omitted and the
# remaining weights are normalized, so an absent degree or projects section is
# not an automatic penalty.
RESUME_QUALITY_WEIGHTS = {
    "structure_organization": 0.15,
    "professional_profile": 0.10,
    "work_experience": 0.25,
    "skills_presentation": 0.15,
    "education_certifications": 0.10,
    "projects_achievements": 0.15,
    "language_readability": 0.10,
}

_ACTION_VERBS = re.compile(
    r"\b(built|created|designed|developed|delivered|implemented|improved|"
    r"increased|reduced|led|launched|managed|organized|produced|resolved|"
    r"streamlined|tested|wrote|analyzed|supported|coordinated|maintained)\b", re.I
)
_DATE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?:19|20)\d{2}\b",
    re.I,
)
_METRIC = re.compile(r"(?:\b\d+(?:\.\d+)?\s?%|\$\s?\d[\d,.]*|\b\d+\s+(?:users|clients|projects|hours|days|weeks)\b)", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•▪] |\d+[.)]\s+)")


def _score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def _observed_sections(sections: Dict[str, Any]) -> List[str]:
    return [key for key, value in sections.items() if str(value or "").strip()]


def assess_resume_quality(
    resume_text: str,
    sections: Dict[str, Any],
    contact_info: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> dict:
    """Score resume presentation from extracted evidence, not job alignment.

    Grammar/spelling judgments are deliberately not guessed. The deterministic
    checks assess text readability signals and explicitly expose their coverage.
    """
    text = resume_text or ""
    sections = sections or {}
    contact_info = contact_info or {}
    if not text.strip():
        return {
            "score": None,
            "score_type": "heuristic_resume_presentation_quality",
            "dimensions": {},
            "category_scores": {},
            "category_details": {},
            "weights_used": {},
            "evidence_coverage": {"available_categories": 0, "total_categories": len(RESUME_QUALITY_WEIGHTS)},
            "contact_coverage": 0.0,
            "observed_sections": [],
            "experience_evidence": {},
            "project_evidence": {},
            "recommendations": [],
            "limitations": "No readable resume text was available, so resume quality cannot be assessed.",
            "method": "Weighted mean of applicable resume presentation categories; no score is produced without extracted text.",
        }
    observed = _observed_sections(sections)
    active_weights = dict(RESUME_QUALITY_WEIGHTS)
    if weights:
        active_weights.update({key: float(value) for key, value in weights.items() if key in active_weights})

    present_count = len(observed)
    word_count = len(text.split())
    heading_score = min(100.0, present_count / 5 * 100)
    length_score = 100.0 if 150 <= word_count <= 1000 else (65.0 if 80 <= word_count < 150 or 1000 < word_count <= 1500 else 35.0)
    structure_score = _score(heading_score * 0.75 + length_score * 0.25)

    summary = str(sections.get("summary") or "")
    summary_words = len(summary.split())
    if not summary:
        profile_score = None
        profile_note = "No summary was extracted; this is optional and excluded from the score."
    else:
        concise = 100 if 25 <= summary_words <= 100 else (70 if 12 <= summary_words <= 140 else 40)
        declared_skills = set(re.findall(r"[a-z][a-z0-9+#.\-]{2,}", summary.lower()))
        body = " ".join(str(sections.get(name) or "") for name in ("experience", "projects", "education")).lower()
        claim_terms = {term for term in declared_skills if term in {"python", "java", "javascript", "react", "sql", "aws", "leadership", "analytics", "sales", "finance"}}
        support = 100 if not claim_terms else 100 * sum(term in body for term in claim_terms) / len(claim_terms)
        profile_score = _score(concise * 0.55 + support * 0.45)
        profile_note = "Summary clarity is estimated from length and whether a small set of explicit claims also appears elsewhere in the resume."

    experience = str(sections.get("experience") or "")
    if not experience:
        experience_score = None
        experience_note = "No employment experience section was extracted; project and academic experience are assessed separately."
        experience_facts = {}
    else:
        lines = [line.strip() for line in experience.splitlines() if line.strip()]
        bullets = [line for line in lines if _BULLET.search(line)]
        evidence_lines = bullets or lines
        action_ratio = sum(bool(_ACTION_VERBS.search(line)) for line in evidence_lines) / max(1, len(evidence_lines))
        outcome_ratio = sum(bool(_METRIC.search(line)) for line in evidence_lines) / max(1, len(evidence_lines))
        date_lines = sum(bool(_DATE.search(line)) for line in lines)
        role_detail = min(100.0, (sum(bool(re.search(r"\b(?:inc|llc|ltd|corp|company|university|hospital|agency|studio)\b", line, re.I)) for line in lines) * 30) + (date_lines * 20) + (len(lines) > 1) * 20)
        # Metrics are one outcome signal, not a mandatory condition.
        experience_score = _score(35 + action_ratio * 35 + min(30, outcome_ratio * 30) + min(10, role_detail * 0.1))
        experience_note = "Responsibilities, action-oriented wording, role/date signals, and stated outcomes are checked separately. Metrics are a useful signal but are not required."
        experience_facts = {
            "content_lines": len(lines),
            "action_oriented_lines": sum(bool(_ACTION_VERBS.search(line)) for line in evidence_lines),
            "outcome_lines_with_explicit_metrics": sum(bool(_METRIC.search(line)) for line in evidence_lines),
            "lines_with_date_signals": date_lines,
        }

    skills = str(sections.get("skills") or "")
    project = str(sections.get("projects") or "")
    if not skills:
        skills_score = None
        skills_note = "No skills section was extracted; demonstrated skills remain visible in other sections."
    else:
        skill_items = [item for item in re.split(r"[,;|\n•]+", skills) if item.strip()]
        demonstrated_text = " ".join(str(sections.get(name) or "") for name in ("experience", "projects")).lower()
        meaningful = [item.strip() for item in skill_items if len(item.strip()) > 1]
        corroborated = sum(any(token in demonstrated_text for token in re.findall(r"[a-z0-9+#.\-]+", item.lower()) if len(token) > 2) for item in meaningful)
        consistency = 100 if not meaningful else 100 * corroborated / len(meaningful)
        skills_score = _score(65 + consistency * 0.35)
        skills_note = "Organization and corroboration of listed skills are signals; an uncorroborated list is not treated as proof of proficiency."

    education = str(sections.get("education") or "")
    certifications = str(sections.get("certifications") or "")
    if not education and not certifications:
        education_score = None
        education_note = "No education or certification section was extracted; this optional category is excluded."
    else:
        qualification_text = "\n".join((education, certifications))
        has_status = bool(re.search(r"\b(in progress|currently pursuing|expected|candidate for|completed|graduated|certified|issued)\b", qualification_text, re.I))
        has_institution = bool(re.search(r"\b(university|college|institute|school|academy)\b", education, re.I))
        has_date = bool(_DATE.search(qualification_text))
        education_score = _score(55 + 15 * has_status + 15 * has_institution + 15 * has_date)
        education_note = "Completion status, institution and date signals are reported; a degree is not required for general resume quality."

    if not project:
        projects_score = None
        projects_note = "No projects or achievements section was extracted; this optional category is excluded."
        project_facts = {}
    else:
        lines = [line.strip() for line in project.splitlines() if line.strip()]
        technology_or_contribution = sum(bool(_ACTION_VERBS.search(line)) for line in lines)
        outcome_count = sum(bool(_METRIC.search(line)) for line in lines)
        projects_score = _score(55 + min(30, technology_or_contribution * 12) + min(15, outcome_count * 5))
        projects_note = "Projects count as practical evidence but are kept distinct from formal employment. Contribution and stated outcomes are surfaced."
        project_facts = {"project_lines": len(lines), "action_or_contribution_lines": technology_or_contribution, "outcome_lines_with_explicit_metrics": outcome_count}

    all_lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated_ratio = 1 - (len(set(line.lower() for line in all_lines)) / max(1, len(all_lines)))
    avg_line_words = sum(len(line.split()) for line in all_lines) / max(1, len(all_lines))
    readability_score = _score(100 - min(40, repeated_ratio * 100) - (20 if avg_line_words > 35 else 0) - (10 if avg_line_words < 2 else 0))

    dimensions: Dict[str, Optional[float]] = {
        "structure_organization": structure_score,
        "section_completeness": heading_score,
        "professional_profile": profile_score,
        "work_experience": experience_score,
        "skills_presentation": skills_score,
        "education_certifications": education_score,
        "projects_achievements": projects_score,
        "language_readability": readability_score,
    }
    available = [key for key in active_weights if dimensions.get(key) is not None and active_weights[key] > 0]
    weight_total = sum(active_weights[key] for key in available)
    quality_score = _score(sum(float(dimensions[key]) * active_weights[key] for key in available) / weight_total) if weight_total else 0.0

    contact_fields = ("email", "phone", "linkedin")
    contact_coverage = round(sum(bool(contact_info.get(key)) for key in contact_fields) / len(contact_fields) * 100, 1)
    recommendations: List[dict] = []
    if experience and experience_facts.get("content_lines", 0) and experience_facts.get("action_oriented_lines", 0) == 0:
        recommendations.append({"priority": "high", "action": "Make work contributions specific", "why": "Role responsibilities are easier to understand when the resume states what you personally did.", "how": "Start bullets with an accurate action and name the system, task, or people involved. Add a verified result when available; do not invent metrics."})
    if skills and skills_score is not None and skills_score < 80:
        recommendations.append({"priority": "medium", "action": "Connect listed skills to evidence", "why": "A reader cannot infer proficiency from a standalone keyword list.", "how": "For skills you have used, point to a project, course, or work example. Keep untested skills clearly labeled."})
    if not summary:
        recommendations.append({"priority": "low", "action": "Consider adding a concise professional profile", "why": "A short profile can quickly orient readers to your direction.", "how": "Summarize your target work and strongest supported experience in a few sentences. A summary is optional."})
    if not any(contact_info.get(key) for key in contact_fields):
        recommendations.append({"priority": "medium", "action": "Check contact information", "why": "The extracted document did not reveal a contact route.", "how": "If appropriate for your privacy and application context, include an email or professional link."})

    category_notes = {
        "structure_organization": "Section heading coverage and broad length signals; formatting consistency is limited by text extraction.",
        "professional_profile": profile_note,
        "work_experience": experience_note,
        "skills_presentation": skills_note,
        "education_certifications": education_note,
        "projects_achievements": projects_note,
        "language_readability": "Line length and repetition signals only. Grammar and spelling are not reliably evaluated by this deterministic model.",
    }
    category_scores = {key: dimensions[key] for key in active_weights}
    category_details = {
        key: {"score": dimensions[key], "weight": active_weights[key], "applicable": dimensions[key] is not None, "explanation": category_notes[key]}
        for key in active_weights
    }
    return {
        "score": quality_score,
        "score_type": "heuristic_resume_presentation_quality",
        "dimensions": dimensions,
        "category_scores": category_scores,
        "category_details": category_details,
        "weights_used": {key: active_weights[key] / weight_total for key in available} if weight_total else {},
        "evidence_coverage": {"available_categories": len(available), "total_categories": len(active_weights)},
        "contact_coverage": contact_coverage,
        "observed_sections": observed,
        "experience_evidence": experience_facts,
        "project_evidence": project_facts,
        "recommendations": recommendations,
        "limitations": "Heuristic text assessment, not a measure of candidate ability or hiring likelihood. Grammar, truth, layout fidelity, career progression and nuanced accomplishment quality need human review; optional categories are excluded when not extracted.",
        "method": "Weighted mean of applicable categories: structure/organization 15%, professional profile 10%, work experience presentation 25%, skills presentation 15%, education/certifications 10%, projects/achievements 15%, language/readability 10%. Optional categories are excluded and the remaining weights renormalized. Contact completeness is reported separately and not used to penalize privacy choices.",
    }
