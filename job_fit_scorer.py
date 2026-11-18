"""Transparent job-fit scoring from extracted requirements and resume evidence."""

from typing import Dict, Optional

from skill_matcher import AccurateSkillMatcher


JOB_FIT_WEIGHTS = {
    "required_skills": 30.0,
    "relevant_experience": 25.0,
    "responsibilities": 15.0,
    "education_qualifications": 10.0,
    "preferred_skills": 10.0,
    "projects_achievements": 10.0,
}


def _all_skills(grouped: Dict[str, list]) -> list:
    return [item for items in (grouped or {}).values() for item in items]


def _evidence_value(item: dict) -> float:
    if item.get("matched"):
        # A skills-list mention is useful evidence but weaker than an example in
        # work, projects, or education. One exact requirement never counts twice.
        return 1.0 if set(item.get("where_found", [])) & {"experience", "projects", "education"} else 0.55
    if item.get("related_resume_skills"):
        return 0.35
    return 0.0


def calculate_job_fit(
    skill_details: dict,
    job_requirements: dict,
    qualification_evidence: dict,
    resume_sections: Optional[dict] = None,
    weights: Optional[dict] = None,
) -> dict:
    """Return a job-specific score and per-requirement evidence classifications.

    Requirements with no extracted evidence are reported as not demonstrated;
    a missing candidate section makes non-skill qualifications unverifiable and
    excludes that category from the score. The score is not a hiring probability.
    """
    effective_weights = dict(JOB_FIT_WEIGHTS)
    if weights:
        effective_weights.update({key: float(value) for key, value in weights.items() if key in effective_weights})
    sections = resume_sections or {}
    matched_by_name = {item["skill"]: item for item in skill_details.get("matched_details", [])}
    missing_by_name = {item["skill"]: item for item in skill_details.get("missing_details", [])}
    job_skills = _all_skills(skill_details.get("job_skills", {}))

    evidence_rows = []
    for requirement in job_skills:
        name = requirement["name"]
        hit = matched_by_name.get(name)
        gap = missing_by_name.get(name, {})
        related = gap.get("related_resume_skills", [])
        where = hit.get("where_found", []) if hit else []
        if hit and set(where) & {"experience", "projects", "education"}:
            classification = "direct match"
        elif hit:
            classification = "partial match: listed without contextual evidence"
        elif related:
            classification = "strong related match" if len(related) == 1 else "transferable match"
        else:
            classification = "not demonstrated"
        evidence_rows.append({
            "requirement": name,
            "requirement_type": requirement.get("requirement_type", "important"),
            "classification": classification,
            "resume_sources": where or related,
            "evidence_weight": round(_evidence_value({"matched": bool(hit), "where_found": where, "related_resume_skills": related}), 2),
            "interpretation": "Not demonstrated means this resume text did not show evidence; it does not establish that the candidate lacks the ability." if classification == "not demonstrated" else "",
        })

    component_values = {}
    component_evidence = {}
    for category, labels in (
        ("required_skills", {"required", "important"}),
        ("preferred_skills", {"preferred"}),
    ):
        rows = [row for row in evidence_rows if row["requirement_type"] in labels]
        if rows:
            component_values[category] = sum(row["evidence_weight"] for row in rows) / len(rows) * 100
            component_evidence[category] = {"assessable": len(rows), "total": len(rows)}

    # If the posting did not label skills, treat them as contextual requirements
    # with lower weight rather than silently upgrading them to mandatory.
    unlabeled = [row for row in evidence_rows if row["requirement_type"] not in {"required", "important", "preferred"}]
    if unlabeled:
        component_values["responsibilities"] = sum(row["evidence_weight"] for row in unlabeled) / len(unlabeled) * 100
        component_evidence["responsibilities"] = {"assessable": len(unlabeled), "total": len(unlabeled)}

    exp_rows = qualification_evidence.get("experience", [])
    exp_values = []
    for item in exp_rows:
        status = item.get("status", "")
        evidence = item.get("resume_evidence", "")
        if "cannot determine" in status:
            continue
        if "meets the posted threshold" in status:
            exp_values.append(100.0)
        elif "below the posted threshold" in status:
            try:
                stated, asked = [int(value) for value in __import__("re").findall(r"\d+", evidence)[:2]]
                exp_values.append(min(100.0, stated / max(1, asked) * 100.0))
            except (ValueError, TypeError):
                exp_values.append(0.0)
    if exp_values:
        component_values["relevant_experience"] = sum(exp_values) / len(exp_values)
        component_evidence["relevant_experience"] = {"assessable": len(exp_values), "total": len(exp_rows)}
    elif sections.get("experience"):
        relevant = [row for row in evidence_rows if row["requirement_type"] in {"required", "important"}]
        if relevant:
            source_rows = [row for row in relevant if "experience" in row["resume_sources"]]
            component_values["relevant_experience"] = sum(row["evidence_weight"] for row in source_rows) / len(relevant) * 100
            component_evidence["relevant_experience"] = {"assessable": len(relevant), "total": len(relevant), "source": "requirement skills found in extracted work experience"}

    education_rows = qualification_evidence.get("education", []) + qualification_evidence.get("certifications", [])
    education_values = []
    education_present = bool(sections.get("education") or sections.get("certifications"))
    for item in education_rows:
        status = item.get("status", "")
        if "cannot determine" in status:
            continue
        if "not found" in status:
            education_values.append(0.0)
        elif "different degree level" in status or "in progress" in status:
            education_values.append(50.0)
        elif "resume states a" in status or "phrase found" in status:
            education_values.append(100.0)
    if education_values:
        component_values["education_qualifications"] = sum(education_values) / len(education_values)
        component_evidence["education_qualifications"] = {"assessable": len(education_values), "total": len(education_rows)}
    elif education_rows and education_present:
        component_values["education_qualifications"] = 0.0
        component_evidence["education_qualifications"] = {"assessable": len(education_rows), "total": len(education_rows)}

    project_skills = set()
    if sections.get("projects"):
        matcher = AccurateSkillMatcher()
        extracted_projects = matcher.extract_skills_from_text(sections["projects"])
        project_skills = {item["name"] for item in _all_skills(extracted_projects)}
        if evidence_rows:
            component_values["projects_achievements"] = sum(1.0 if row["requirement"] in project_skills else 0.0 for row in evidence_rows) / len(evidence_rows) * 100
            component_evidence["projects_achievements"] = {"assessable": len(evidence_rows), "total": len(evidence_rows), "source": "detected job skills stated in the projects section"}

    available = [key for key in effective_weights if key in component_values and effective_weights[key] > 0]
    weight_total = sum(effective_weights[key] for key in available)
    score = round(sum(component_values[key] * effective_weights[key] for key in available) / weight_total, 2) if weight_total else None
    evidence_weight = sum(effective_weights[key] for key in available)
    confidence_coverage = round(evidence_weight / sum(effective_weights.values()) * 100, 1) if sum(effective_weights.values()) else 0.0
    if score is None:
        score_status = "not assessable: no job requirements could be compared with available resume evidence"
    else:
        score_status = "heuristic score based on assessable extracted requirements"

    return {
        "score": score,
        "score_status": score_status,
        "category_scores": {key: round(component_values[key], 1) for key in available},
        "category_evidence": {key: component_evidence.get(key, {}) for key in available},
        "weights_used": {key: round(effective_weights[key] / weight_total, 4) for key in available} if weight_total else {},
        "evidence_coverage_percent": confidence_coverage,
        "requirements": evidence_rows,
        "unknown_qualification_checks": [item for category in ("education", "experience", "certifications") for item in qualification_evidence.get(category, []) if "cannot determine" in item.get("status", "")],
        "method": "Weighted mean of assessable job-specific categories: required skills 30%, relevant experience 25%, responsibilities/contextual skills 15%, education and certifications 10%, preferred skills 10%, projects and achievements 10%. Categories with no extracted or assessable requirements are omitted and the remaining weights renormalized. Exact contextual evidence is valued above skills-list mentions; related technologies receive limited partial credit and are labeled. This is a heuristic, not a hiring probability.",
        "limitations": "Requirement extraction and evidence classification are deterministic heuristics. A not-demonstrated result refers only to the supplied resume text and is not proof the candidate lacks that qualification. Review source text and unknown evidence before making decisions.",
    }
