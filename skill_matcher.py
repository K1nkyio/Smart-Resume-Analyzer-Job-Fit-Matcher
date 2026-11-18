import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from config import EXPERIENCE_LEVELS, INDUSTRY_SKILLS, ROLE_PLAYBOOKS, SKILLS, SKILL_SYNONYMS


SECTION_WEIGHTS = {
    "skills": 1.30,
    "projects": 1.20,
    "experience": 1.15,
    "summary": 1.05,
    "education": 0.95,
    "general": 1.00,
}

SECTION_ALIASES = {
    "summary": ["summary", "profile", "objective", "about"],
    "experience": ["experience", "employment", "work history", "professional experience"],
    "education": ["education", "academic", "qualification"],
    "skills": ["skills", "technical skills", "competencies", "expertise"],
    "projects": ["projects", "portfolio", "case studies"],
}

GENERIC_SKILL_IMPACT = {
    "SQL": "SQL is commonly required for reporting, analysis, and production data work.",
    "Python": "Python is frequently expected for automation, analytics, and backend workflows.",
    "JavaScript": "JavaScript is required across modern frontend and full-stack roles.",
    "AWS": "Cloud platform experience is a frequent screening requirement.",
    "Git": "Version control is a baseline collaboration requirement in engineering teams.",
    "Communication": "Clear communication is required for collaboration and stakeholder alignment.",
}

# Curated adjacency only: these technologies may provide transferable concepts,
# but are never counted as direct equivalence or direct skill coverage.
RELATED_SKILL_FAMILIES = (
    {"SQL", "PostgreSQL", "MySQL", "SQLite", "SQL Server", "Oracle"},
    {"AWS", "Azure", "Google Cloud"},
    {"React", "Vue.js", "Angular", "Next.js"},
    {"Laravel", "Express.js", "Django", "Flask", "FastAPI"},
    {"REST API", "API Development"},
    {"PyTorch", "TensorFlow"},
)


def _related_skill_index():
    index = defaultdict(set)
    for family in RELATED_SKILL_FAMILIES:
        for skill in family:
            index[skill].update(family - {skill})
    return index


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


class AccurateSkillMatcher:
    def __init__(self):
        self.skills_db = SKILLS
        self.industry_db = INDUSTRY_SKILLS
        self.skills_catalog = self._build_skills_catalog()
        self.skill_variations = self._build_skill_variations()
        self.related_skills = _related_skill_index()

    def _build_skills_catalog(self) -> Dict[str, str]:
        catalog = {}
        for category, skills_map in self.skills_db.items():
            for skill in skills_map:
                catalog[skill] = category
        return catalog

    def _build_skill_variations(self) -> Dict[str, str]:
        variations = {}
        for canonical_skill in self.skills_catalog:
            base_key = _safe_slug(canonical_skill)
            if base_key:
                variations[base_key] = canonical_skill

            compact = base_key.replace(" ", "")
            if compact:
                variations[compact] = canonical_skill

            for synonym in SKILL_SYNONYMS.get(canonical_skill, []):
                syn_key = _safe_slug(synonym)
                if syn_key:
                    variations[syn_key] = canonical_skill
                    variations[syn_key.replace(" ", "")] = canonical_skill
        return variations

    def _canonical_role(self, role: Optional[str]) -> str:
        if not role:
            return "Unspecified role"
        role_slug = _safe_slug(role)
        for known_role in ROLE_PLAYBOOKS:
            if _safe_slug(known_role) == role_slug:
                return known_role
        for known_role in ROLE_PLAYBOOKS:
            if role_slug in _safe_slug(known_role) or _safe_slug(known_role) in role_slug:
                return known_role
        return role.strip()

    def _infer_sections(self, text: str) -> Dict[str, str]:
        sections = defaultdict(list)
        current_section = "general"

        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            line_slug = _safe_slug(line)
            section_hit = None
            for section_name, aliases in SECTION_ALIASES.items():
                if line_slug in {_safe_slug(alias) for alias in aliases}:
                    section_hit = section_name
                    break

            if section_hit:
                current_section = section_hit
                continue

            sections[current_section].append(line)

        if not sections:
            sections["general"] = [text]

        return {name: " ".join(lines) for name, lines in sections.items()}

    def _is_skill_in_text(self, skill_variant: str, text: str) -> bool:
        escaped_skill = re.escape(skill_variant)
        if skill_variant in ["c++", "c#", ".net", "asp.net"]:
            pattern = escaped_skill
        else:
            pattern = r"\b" + escaped_skill + r"\b"
        return bool(re.search(pattern, text, re.IGNORECASE))

    def _extract_skill_mentions(self, text: str) -> Dict[str, List[str]]:
        mentions = defaultdict(list)
        text_clean = _safe_slug(text)

        for skill_variant, canonical_skill in self.skill_variations.items():
            if canonical_skill == "C#":
                found = bool(re.search(r"\bc\s*\#|\bc\s+sharp\b", text or "", re.I))
            elif canonical_skill in {".NET", "ASP.NET"}:
                found = bool(re.search(r"\b(?:asp\s*\.\s*net|aspnet|dotnet|\.net)\b", text or "", re.I))
            else:
                found = self._is_skill_in_text(skill_variant, text_clean)
            if found:
                mentions[canonical_skill].append(skill_variant)

        return mentions

    def _requirement_type(self, skill: str, job_text: str) -> str:
        """Classify explicit cues and headings without turning every bullet mandatory."""
        patterns = (
            ("preferred", r"\b(preferred|nice[- ]to[- ]have|desirable|bonus|asset|advantageous)\b"),
            ("required", r"\b(required|must[- ]have|essential|mandatory|minimum qualification)\b"),
        )
        current_section = ""
        found_types = []
        for raw_line in (job_text or "").splitlines():
            line = raw_line.strip().lstrip("-*• ")
            slug = _safe_slug(line)
            if re.fullmatch(r"(?:minimum|required|mandatory) qualifications?", slug):
                current_section = "required"
                continue
            if re.fullmatch(r"(?:preferred )?(?:qualifications|requirements|skills)", slug):
                current_section = "preferred" if slug.startswith("preferred") else "important"
                continue
            if re.fullmatch(r"(?:key )?responsibilities|duties|what you will do", slug):
                current_section = "contextual"
                continue
            clauses = [clause.strip() for clause in re.split(r"(?<=[.!?;])\s+", line) if clause.strip()]
            for clause in clauses:
                if skill not in self._extract_skill_mentions(clause):
                    continue
                explicit = next((kind for kind, cue in patterns if re.search(cue, clause, re.I)), None)
                found_types.append(explicit or current_section or "unspecified")
        for kind in ("required", "preferred", "important", "contextual"):
            if kind in found_types:
                return kind
        return found_types[0] if found_types else "unspecified"

    def extract_skills_from_text(self, text: str, sections: Optional[Dict[str, str]] = None) -> Dict[str, List[dict]]:
        if not text:
            return {}

        section_map = sections if sections else self._infer_sections(text)
        combined = {}

        for section_name, section_text in section_map.items():
            found_in_section = self._extract_skill_mentions(section_text)
            for canonical_skill, matches in found_in_section.items():
                if canonical_skill not in combined:
                    combined[canonical_skill] = {
                        "name": canonical_skill,
                        "category": self.skills_catalog.get(canonical_skill, "technical"),
                        "matches": set(),
                        "sections": set(),
                        "weighted_total": 0.0,
                    }

                combined[canonical_skill]["matches"].update(matches)
                combined[canonical_skill]["sections"].add(section_name)
                combined[canonical_skill]["weighted_total"] += SECTION_WEIGHTS.get(section_name, 1.0)

        grouped = defaultdict(list)
        for skill_name, data in combined.items():
            section_count = max(1, len(data["sections"]))
            avg_weight = data["weighted_total"] / section_count
            confidence = min(1.0, 0.60 + (0.08 * section_count) + (0.08 * max(0.0, avg_weight - 1.0)))
            grouped[data["category"]].append(
                {
                    "name": skill_name,
                    "confidence": round(confidence, 3),
                    "matched_as": sorted(data["matches"])[0],
                    "sections_found": sorted(data["sections"]),
                    "weighted_score": round(avg_weight, 3),
                }
            )

        for category in grouped:
            grouped[category] = sorted(grouped[category], key=lambda item: item["confidence"], reverse=True)

        return dict(grouped)

    def _skill_set(self, extracted_skills: Dict[str, List[dict]]) -> Set[str]:
        return {skill["name"] for skills in extracted_skills.values() for skill in skills}

    def _skill_index(self, extracted_skills: Dict[str, List[dict]]) -> Dict[str, dict]:
        index = {}
        for skills in extracted_skills.values():
            for skill in skills:
                index[skill["name"]] = skill
        return index

    def _skill_impact(self, skill: str, target_role: Optional[str]) -> str:
        role_name = self._canonical_role(target_role)
        role_impact = ROLE_PLAYBOOKS.get(role_name, {}).get("skill_impact", {})
        if skill in role_impact:
            return role_impact[skill]
        return GENERIC_SKILL_IMPACT.get(skill, "This skill appears frequently in screened job requirements.")

    def _fix_example(self, skill: str) -> dict:
        return {
            "before": "Add evidence only if you have used this skill.",
            "after": f"If accurate, describe where you used {skill}, your contribution, and any verified result. Otherwise, leave it out.",
        }

    def compare_skills_accurately(
        self,
        resume_text: str,
        job_text: str,
        industry: Optional[str] = None,
        experience_level: Optional[str] = None,
        target_role: Optional[str] = None,
        location: Optional[str] = None,
        resume_sections: Optional[Dict[str, str]] = None,
    ) -> Tuple[Set[str], Set[str], float, dict]:
        if not resume_text or not job_text:
            return set(), set(), 0.0, {
                "matched_details": [],
                "missing_details": [],
                "resume_skills": {},
                "job_skills": {},
                "industry_alignment": 0.0,
            }

        resume_skills = self.extract_skills_from_text(resume_text, sections=resume_sections)
        job_skills = self.extract_skills_from_text(job_text)

        resume_skill_set = self._skill_set(resume_skills)
        job_skill_set = self._skill_set(job_skills)
        resume_skill_index = self._skill_index(resume_skills)

        matched_skills = resume_skill_set.intersection(job_skill_set)
        missing_skills = job_skill_set - resume_skill_set

        requirement_types = {skill: self._requirement_type(skill, job_text) for skill in job_skill_set}
        requirement_weights = {
            skill: 2.0 if requirement_types[skill] == "required" else 1.0
            for skill in job_skill_set
        }
        total_requirement_weight = sum(requirement_weights.values())
        direct_score = (
            sum(requirement_weights[skill] for skill in matched_skills) / total_requirement_weight
            if total_requirement_weight else 0.0
        )

        role_name = self._canonical_role(target_role)
        role_playbook = ROLE_PLAYBOOKS.get(role_name, {})
        role_must_have = set(role_playbook.get("must_have_skills", []))
        role_alignment = (len(resume_skill_set.intersection(role_must_have)) / len(role_must_have)) if role_must_have else 0.0

        if matched_skills:
            avg_context_weight = sum(
                resume_skill_index.get(skill, {}).get("weighted_score", 1.0) for skill in matched_skills
            ) / len(matched_skills)
        else:
            avg_context_weight = 1.0

        section_bonus = max(0.0, min(1.0, (avg_context_weight - 1.0) / 0.30))
        final_score = min(100.0, (direct_score * 70.0) + (role_alignment * 20.0) + (section_bonus * 10.0))

        matched_details = []
        for skill in sorted(matched_skills):
            entry = resume_skill_index.get(skill, {})
            matched_details.append(
                {
                    "skill": skill,
                    "where_found": entry.get("sections_found", ["general"]),
                    "confidence": entry.get("confidence", 0.7),
                    "why_it_matters": self._skill_impact(skill, target_role),
                    "requirement_type": requirement_types[skill],
                    "evidence_level": (
                        "demonstrated" if set(entry.get("sections_found", [])) & {"experience", "projects"}
                        else "mentioned without experience or project evidence"
                    ),
                }
            )

        missing_details = []
        for skill in sorted(missing_skills):
            related_resume_skills = sorted(self.related_skills.get(skill, set()).intersection(resume_skill_set))
            missing_details.append(
                {
                    "skill": skill,
                    "why_it_matters": self._skill_impact(skill, target_role),
                    "fix_example": self._fix_example(skill),
                    "requirement_type": requirement_types[skill],
                    "evidence_level": (
                        "related skill found; direct requirement not demonstrated"
                        if related_resume_skills else "not detected in the resume text"
                    ),
                    "related_resume_skills": related_resume_skills,
                    "related_skill_note": (
                        "Related technologies can transfer concepts, but this does not establish experience with the requested technology."
                        if related_resume_skills else ""
                    ),
                }
            )

        details = {
            "matched_details": matched_details,
            "missing_details": missing_details,
            "requirement_coverage": {
                kind: {
                    "detected": sum(1 for skill in job_skill_set if requirement_types[skill] == kind),
                    "matched": sum(1 for skill in matched_skills if requirement_types[skill] == kind),
                }
                for kind in ("required", "important", "preferred", "contextual", "unspecified")
            },
            "resume_skills": resume_skills,
            "job_skills": job_skills,
            "industry_alignment": round(role_alignment * 100.0, 2),
            "location_context": location or "Not specified",
            "experience_level_context": experience_level or "Not specified",
            "target_role_context": role_name,
            "score_components": {
                "direct_match": round(direct_score * 100.0, 2),
                "required_skill_coverage": round(
                    sum(1 for skill in matched_skills if requirement_types[skill] == "required") /
                    max(1, sum(1 for skill in job_skill_set if requirement_types[skill] == "required")) * 100.0,
                    2,
                ) if any(requirement_types[skill] == "required" for skill in job_skill_set) else None,
                "role_alignment": round(role_alignment * 100.0, 2),
                "section_bonus": round(section_bonus * 100.0, 2),
            },
            "score_method": "Catalog skill coverage diagnostic only. Final job fit is calculated separately from assessable job-specific skills, experience, responsibilities, education, preferred-skill and project evidence. The catalog is incomplete and this diagnostic is not a hiring probability.",
        }

        if industry and industry in self.industry_db:
            must_have = set(self.industry_db[industry].get("must_have", []))
            details["industry_must_have_coverage"] = round(
                (len(resume_skill_set.intersection(must_have)) / len(must_have) * 100.0), 2
            ) if must_have else 0.0

        return matched_skills, missing_skills, round(final_score, 2), details


def analyze_job_trends(job_descriptions: List[str]) -> dict:
    matcher = AccurateSkillMatcher()
    trend_counter = Counter()

    for desc in job_descriptions:
        if not desc:
            continue
        extracted = matcher.extract_skills_from_text(desc)
        for skills in extracted.values():
            for skill in skills:
                trend_counter[skill["name"]] += 1

    return dict(trend_counter.most_common(25))


def compare_resume_to_job(
    resume_text: str,
    job_text: str,
    industry: Optional[str] = None,
    experience_level: Optional[str] = None,
    target_role: Optional[str] = None,
    location: Optional[str] = None,
    resume_sections: Optional[Dict[str, str]] = None,
):
    matcher = AccurateSkillMatcher()
    matched, missing, score, _ = matcher.compare_skills_accurately(
        resume_text=resume_text,
        job_text=job_text,
        industry=industry,
        experience_level=experience_level,
        target_role=target_role,
        location=location,
        resume_sections=resume_sections,
    )
    return matched, missing, score


def compare_resume_to_job_detailed(
    resume_text: str,
    job_text: str,
    industry: Optional[str] = None,
    experience_level: Optional[str] = None,
    target_role: Optional[str] = None,
    location: Optional[str] = None,
    resume_sections: Optional[Dict[str, str]] = None,
) -> dict:
    matcher = AccurateSkillMatcher()
    matched, missing, score, details = matcher.compare_skills_accurately(
        resume_text=resume_text,
        job_text=job_text,
        industry=industry,
        experience_level=experience_level,
        target_role=target_role,
        location=location,
        resume_sections=resume_sections,
    )
    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "skill_score": score,
        **details,
    }


def get_skill_recommendations(
    missing_skills: Set[str],
    industry: Optional[str] = None,
    target_role: Optional[str] = None,
    experience_level: Optional[str] = None,
    location: Optional[str] = None,
) -> dict:
    matcher = AccurateSkillMatcher()
    role_name = matcher._canonical_role(target_role)
    playbook = ROLE_PLAYBOOKS.get(role_name, {})
    role_impact = playbook.get("skill_impact", {})

    if not missing_skills:
        return {
            "priority_skills": [],
            "learning_resources": {},
            "time_investment": {},
            "career_impact": {},
            "role_specific_tips": [],
        }

    priority_order = {"high": 3, "medium": 2, "low": 1}
    role_must_have = set(playbook.get("must_have_skills", []))
    role_nice_to_have = set(playbook.get("nice_to_have_skills", []))
    expected_count = EXPERIENCE_LEVELS.get(experience_level or "", {}).get("expected_skills_count", 10)

    prioritized = []
    for skill in sorted(missing_skills):
        if skill in role_must_have:
            priority = "high"
            time_estimate = "3-6 weeks"
        elif skill in role_nice_to_have:
            priority = "medium"
            time_estimate = "2-5 weeks"
        else:
            priority = "medium" if expected_count <= 10 else "low"
            time_estimate = "2-4 weeks" if priority == "medium" else "1-3 weeks"

        prioritized.append(
            {
                "skill": skill,
                "priority": priority,
                "time": time_estimate,
                "impact": role_impact.get(skill, "Improves your screening match rate for this role."),
            }
        )

    prioritized.sort(key=lambda item: priority_order[item["priority"]], reverse=True)
    prioritized = prioritized[:8]

    resource_map = {
        "SQL": ["SQLBolt", "Mode SQL Tutorial", "LeetCode SQL practice"],
        "Python": ["Python docs", "Real Python", "Kaggle notebooks"],
        "React": ["React docs", "Frontend Mentor projects", "Testing Library examples"],
        "AWS": ["AWS Skill Builder", "AWS workshops", "Cloud Resume Challenge"],
        "Docker": ["Docker docs", "Play with Docker", "Containerized project template"],
    }

    learning_resources = {}
    time_investment = {}
    career_impact = {}
    for item in prioritized:
        skill = item["skill"]
        learning_resources[skill] = resource_map.get(skill, ["Official docs", "Hands-on portfolio project"])
        time_investment[skill] = item["time"]
        career_impact[skill] = item["impact"]

    role_tips = [
        f"Target role: {role_name}",
        f"Location context: {location or 'Not specified'}",
        f"Prioritize evidence bullets that mention: {', '.join(playbook.get('keywords', [])[:4]) or 'impact metrics'}",
    ]

    if industry and industry in INDUSTRY_SKILLS:
        role_tips.append(
            "Industry must-have focus: " + ", ".join(INDUSTRY_SKILLS[industry].get("must_have", [])[:5])
        )

    return {
        "priority_skills": prioritized,
        "learning_resources": learning_resources,
        "time_investment": time_investment,
        "career_impact": career_impact,
        "role_specific_tips": role_tips,
    }


def generate_job_fit_plan(
    missing_skills: Set[str],
    target_role: Optional[str],
    experience_level: Optional[str],
    location: Optional[str],
    current_skill_score: float,
) -> dict:
    matcher = AccurateSkillMatcher()
    role_name = matcher._canonical_role(target_role)
    playbook = ROLE_PLAYBOOKS.get(role_name, {})

    top_missing = sorted(list(missing_skills))[:5]
    certifications = playbook.get("certifications", [])[:3]
    portfolio_projects = playbook.get("portfolio_projects", [])[:3]

    plan_30 = [
        f"Close foundational gap in: {', '.join(top_missing[:2]) or 'core role skills'}",
        "Where accurate, clarify action, verified result, and impact in relevant resume bullets. Never estimate metrics.",
        "Publish one mini-project artifact (GitHub README or dashboard screenshot).",
    ]
    plan_60 = [
        f"Add intermediate capability in: {', '.join(top_missing[2:4]) or 'advanced role topics'}",
        "Complete one role-specific portfolio project with measurable outcome.",
        "Run 3 mock interviews and refine weak answer areas.",
    ]
    plan_90 = [
        f"Strengthen final priority skill: {top_missing[4] if len(top_missing) > 4 else 'role specialization'}",
        "Apply to targeted roles using tailored resume variants.",
        "Track callback rate and improve based on recruiter feedback.",
    ]

    return {
        "target_role": role_name,
        "experience_level": experience_level or "Not specified",
        "location": location or "Not specified",
        "top_missing_skills": top_missing,
        "certification_suggestions": certifications,
        "portfolio_project_ideas": portfolio_projects,
        "plan_30_days": plan_30,
        "plan_60_days": plan_60,
        "plan_90_days": plan_90,
    }


def debug_skill_extraction(text: str, expected_skills: Optional[List[str]] = None):
    matcher = AccurateSkillMatcher()
    found_skills = matcher.extract_skills_from_text(text)

    print("=== SKILL EXTRACTION DEBUG ===")
    print(f"Text length: {len(text)} characters")
    print("\nFound skills:")

    total_found = 0
    found_names = set()
    for category, skills in found_skills.items():
        print(f"\n{category.upper()}:")
        for skill in skills:
            print(
                f"  - {skill['name']} "
                f"(confidence: {skill['confidence']:.2f}, sections: {skill['sections_found']})"
            )
            total_found += 1
            found_names.add(skill["name"])

    print(f"\nTotal skills found: {total_found}")

    if expected_skills:
        missed = set(expected_skills) - found_names
        if missed:
            print(f"Missed skills: {sorted(missed)}")
        else:
            print("All expected skills found.")

    return found_skills
