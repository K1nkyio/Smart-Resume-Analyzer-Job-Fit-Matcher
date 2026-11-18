"""Deterministic, explainable extraction of common job-description fields."""

import re
from typing import Dict, List, Optional

from skill_matcher import AccurateSkillMatcher


_FIELD_PATTERNS = {
    "title": re.compile(r"^(?:job title|position|role)\s*:\s*(.+)$", re.I),
    "company": re.compile(r"^(?:company|employer|organization)\s*:\s*(.+)$", re.I),
    "location": re.compile(r"^(?:location|work location)\s*:\s*(.+)$", re.I),
}
_REQUIRED = re.compile(r"\b(required|must[- ]have|essential|mandatory|minimum qualification)\b", re.I)
_PREFERRED = re.compile(r"\b(preferred|nice[- ]to[- ]have|desirable|bonus|asset|advantageous)\b", re.I)
_EDUCATION = re.compile(r"\b(bachelor|master|degree|diploma|phd|doctorate|education)\b", re.I)
_EXPERIENCE = re.compile(r"\b(\d+\+?\s+years?|years? of experience|experience in|experience with)\b", re.I)
_CERTIFICATION = re.compile(r"\b(certification|certified|license|licence|credential)\b", re.I)
_DEGREE_LEVELS = {
    "doctorate": re.compile(r"\b(?:ph\.?d\.?|doctorate|doctoral)\b", re.I),
    "master": re.compile(r"\b(?:master(?:'s)?|mba|msc|m\.sc\.?|ma|m\.a\.?)\b", re.I),
    "bachelor": re.compile(r"\b(?:bachelor(?:'s)?|bsc|b\.sc\.?|ba|b\.a\.?|bs|b\.s\.?)\b", re.I),
    "associate": re.compile(r"\b(?:associate(?:'s)?|aa|as)\b", re.I),
    "diploma": re.compile(r"\b(?:diploma|certificate program)\b", re.I),
}
_YEARS_IN_EXPERIENCE = re.compile(
    r"\b(\d+)\s*(?:\+|plus)?\s+years?\s+(?:of\s+)?(?:(?:professional|relevant|work)\s+)?experience\b",
    re.I,
)
_CERTIFICATION_SUFFIX = re.compile(
    r"\b(?:is\s+)?(?:required|preferred|mandatory|essential|a\s+plus|nice\s+to\s+have)\b.*$",
    re.I,
)


def _clean_line(line: str) -> str:
    return re.sub(r"^\s*(?:[-*•▪]|\d+[.)])\s*", "", line).strip()


def extract_job_requirements(text: str) -> Dict[str, object]:
    """Extract explicit fields and requirement bullets without guessing missing data."""
    lines = [_clean_line(line) for line in (text or "").splitlines() if _clean_line(line)]
    fields = {name: "" for name in _FIELD_PATTERNS}
    responsibilities: List[str] = []
    required_items: List[str] = []
    preferred_items: List[str] = []
    education_items: List[str] = []
    experience_items: List[str] = []
    certification_items: List[str] = []
    section = ""

    for line in lines:
        for name, pattern in _FIELD_PATTERNS.items():
            match = pattern.match(line)
            if match and not fields[name]:
                fields[name] = match.group(1).strip()
                break

        heading = line.lower().strip(": ")
        if re.fullmatch(r"(?:key )?responsibilities|what you(?:'|’)ll do|duties", heading):
            section = "responsibilities"
            continue
        if re.fullmatch(r"(?:requirements|qualifications|what you(?:'|’)ll need|minimum qualifications)", heading):
            section = "requirements"
            continue
        if re.fullmatch(r"(?:preferred qualifications|preferred skills|nice to have)", heading):
            section = "preferred"
            continue

        if section == "responsibilities":
            responsibilities.append(line)
        if _PREFERRED.search(line) or section == "preferred":
            preferred_items.append(line)
        elif _REQUIRED.search(line) or section == "requirements":
            required_items.append(line)
        if _EDUCATION.search(line):
            education_items.append(line)
        if _EXPERIENCE.search(line):
            experience_items.append(line)
        if _CERTIFICATION.search(line):
            certification_items.append(line)

    matcher = AccurateSkillMatcher()
    extracted = matcher.extract_skills_from_text(text or "")
    skills = sorted({item["name"] for group in extracted.values() for item in group})
    skill_types = {skill: matcher._requirement_type(skill, text or "") for skill in skills}
    return {
        **fields,
        "responsibilities": list(dict.fromkeys(responsibilities)),
        "required_items": list(dict.fromkeys(required_items)),
        "preferred_items": list(dict.fromkeys(preferred_items)),
        "education_requirements": list(dict.fromkeys(education_items)),
        "experience_requirements": list(dict.fromkeys(experience_items)),
        "certifications": list(dict.fromkeys(certification_items)),
        "skills": [{"name": skill, "requirement_type": skill_types[skill]} for skill in skills],
        "extraction_method": "rule-based text extraction; review the source description for context",
    }


def assess_qualification_evidence(
    resume_text: str,
    resume_sections: Optional[Dict[str, str]],
    requirements: Dict[str, object],
) -> Dict[str, List[dict]]:
    """Compare explicit education, experience-duration, and certificate evidence.

    These checks report what the extracted resume states; they do not validate
    credentials or add points to the job-fit score.
    """
    resume_text = resume_text or ""
    sections = resume_sections or {}
    education_text = str(sections.get("education") or "")
    experience_text = "\n".join(
        str(sections.get(name) or "") for name in ("experience", "summary")
    )
    certification_text = str(sections.get("certifications") or "")
    if not certification_text:
        certification_text = resume_text

    education_results = []
    for line in requirements.get("education_requirements", []):
        required_level = next((name for name, pattern in _DEGREE_LEVELS.items() if pattern.search(line)), None)
        education_lines = [item.strip() for item in education_text.splitlines() if item.strip()]
        matching_lines = [
            item for item in education_lines
            if required_level and _DEGREE_LEVELS[required_level].search(item)
        ]
        candidate_levels = [name for name, pattern in _DEGREE_LEVELS.items() if pattern.search(education_text)]
        if not education_text:
            status, evidence = "cannot determine: no education section extracted", ""
        elif not required_level:
            status, evidence = "cannot determine: requirement level is not explicit", ""
        elif matching_lines:
            matched_text = "\n".join(matching_lines)
            in_progress = bool(re.search(r"\b(in progress|currently pursuing|pursuing|expected|candidate for|enrolled)\b", matched_text, re.I))
            completed = bool(re.search(r"\b(completed|graduated|earned|awarded|degree conferred)\b", matched_text, re.I))
            if in_progress:
                status, evidence = f"resume states a {required_level} qualification in progress; not completed", matched_text
            elif completed:
                status, evidence = f"resume states a completed {required_level} qualification; self-reported", matched_text
            else:
                status, evidence = f"resume states a {required_level} qualification; completion status not explicit", matched_text
        elif candidate_levels:
            status, evidence = "a different degree level is stated; requirement not confirmed", ", ".join(candidate_levels)
        else:
            status, evidence = "requirement not found in extracted education text", ""
        education_results.append({"requirement": line, "status": status, "resume_evidence": evidence})

    experience_results = []
    resume_year_values = [int(value) for value in _YEARS_IN_EXPERIENCE.findall(experience_text)]
    for line in requirements.get("experience_requirements", []):
        required_match = _YEARS_IN_EXPERIENCE.search(line)
        if not required_match:
            status, evidence = "cannot determine: no numeric duration extracted from posting", ""
        elif not resume_year_values:
            status, evidence = "cannot determine: resume does not explicitly state years of experience", ""
        else:
            required_years = int(required_match.group(1))
            stated_years = max(resume_year_values)
            meets = stated_years >= required_years
            status = (
                "resume-stated duration meets the posted threshold; self-reported"
                if meets else "resume-stated duration is below the posted threshold; self-reported"
            )
            evidence = f"{stated_years} years stated; posting asks for {required_years}"
        experience_results.append({"requirement": line, "status": status, "resume_evidence": evidence})

    certification_results = []
    candidate_cert_text = _normalize_credential(certification_text)
    for line in requirements.get("certifications", []):
        credential = _CERTIFICATION_SUFFIX.sub("", line).strip(" :-")
        credential = re.sub(r"^(?:required|preferred|certification in|certified in)\s+", "", credential, flags=re.I).strip()
        normalized_credential = _normalize_credential(credential)
        generic_credential = normalized_credential in {
            "certification", "certifications", "certified", "license", "licence", "credential",
            "professional certification", "relevant certification",
        }
        if len(normalized_credential) < 3 or generic_credential:
            status, evidence = "cannot determine: credential is not named in the posting", ""
        elif normalized_credential in candidate_cert_text:
            status, evidence = "credential phrase found in resume text; not independently verified", credential
        else:
            status, evidence = "named credential not found in resume text; verify before treating as missing", ""
        certification_results.append({"requirement": line, "status": status, "resume_evidence": evidence})

    return {
        "education": education_results,
        "experience": experience_results,
        "certifications": certification_results,
        "score_inclusion": "These evidence checks are not included in the numeric job-fit score.",
    }


def _normalize_credential(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
