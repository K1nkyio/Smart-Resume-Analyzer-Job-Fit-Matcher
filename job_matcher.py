import os
import statistics
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Set

import requests

from config import INDUSTRY_SKILLS, JOB_TRENDS, ROLE_PLAYBOOKS, SALARY_DATA, SKILLS


def _safe_slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9+#\.]+", " ", (text or "").lower()).strip()


class JobMatcher:
    def __init__(self, local_only: bool = False, country_code: str = "us"):
        self.salary_data = SALARY_DATA
        self.job_trends = JOB_TRENDS
        self.industry_skills = INDUSTRY_SKILLS
        self.local_only = local_only
        self.country_code = country_code.lower()
        self.known_skills = self._build_known_skill_list()

    def _build_known_skill_list(self) -> List[str]:
        skills = []
        for _, skills_map in SKILLS.items():
            skills.extend(list(skills_map.keys()))
        return sorted(set(skills))

    def _extract_skills_from_text(self, text: str) -> Set[str]:
        import re

        text_slug = _safe_slug(text)
        found = set()
        for skill in self.known_skills:
            skill_slug = _safe_slug(skill)
            if not skill_slug:
                continue
            pattern = r"\b" + re.escape(skill_slug) + r"\b"
            if re.search(pattern, text_slug):
                found.add(skill)
        return found

    def _normalize_resume_skills(self, resume_skills) -> Set[str]:
        normalized = set()
        if isinstance(resume_skills, dict):
            for category_skills in resume_skills.values():
                if isinstance(category_skills, list):
                    for skill_info in category_skills:
                        if isinstance(skill_info, dict) and "name" in skill_info:
                            normalized.add(skill_info["name"])
                        elif isinstance(skill_info, str):
                            normalized.add(skill_info)
        elif isinstance(resume_skills, list):
            for skill in resume_skills:
                if isinstance(skill, str):
                    normalized.add(skill)
                elif isinstance(skill, dict) and "name" in skill:
                    normalized.add(skill["name"])
        return normalized

    def _fetch_adzuna_jobs(self, query: str, location: str, results: int = 30) -> List[dict]:
        app_id = os.getenv("ADZUNA_APP_ID")
        app_key = os.getenv("ADZUNA_APP_KEY")
        if self.local_only or not app_id or not app_key:
            return []

        url = f"https://api.adzuna.com/v1/api/jobs/{self.country_code}/search/1"
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "results_per_page": results,
            "what": query,
            "where": location,
            "content-type": "application/json",
        }

        try:
            response = requests.get(url, params=params, timeout=12)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []

        normalized_jobs = []
        for job in payload.get("results", []):
            normalized_jobs.append(
                {
                    "title": job.get("title") or query,
                    "company": (job.get("company") or {}).get("display_name", "Unknown Company"),
                    "location": ((job.get("location") or {}).get("display_name") or location),
                    "description": job.get("description", ""),
                    "salary_min": job.get("salary_min"),
                    "salary_max": job.get("salary_max"),
                    "experience_level": "Unknown",
                    "industry": "General",
                    "source": "live_api_adzuna",
                    "url": job.get("redirect_url", ""),
                }
            )
        return normalized_jobs

    def _fallback_jobs(self, query: str, location: str) -> List[dict]:
        fallback_roles = [query] + [role for role in list(ROLE_PLAYBOOKS.keys()) if role != query]
        fallback_jobs = []

        for role in fallback_roles[:8]:
            role_salary = self.salary_data.get(role, self.salary_data.get("Software Engineer", {}))
            mid = role_salary.get("Mid Level", {"min": 85000, "max": 120000, "average": 100000, "growth": 6})
            playbook = ROLE_PLAYBOOKS.get(role, {})
            skills = playbook.get("must_have_skills", ["Communication", "Problem Solving", "SQL"])
            keywords = playbook.get("keywords", ["impact", "delivery", "ownership"])
            description = (
                f"We are hiring a {role}. Required skills include {', '.join(skills[:5])}. "
                f"The role focuses on {', '.join(keywords[:3])}."
            )
            fallback_jobs.append(
                {
                    "title": role,
                    "company": "Market Benchmark Employer",
                    "location": location or "United States",
                    "description": description,
                    "salary_min": mid["min"],
                    "salary_max": mid["max"],
                    "experience_level": "Mid Level",
                    "industry": "General",
                    "source": "fallback_dataset",
                    "url": "",
                }
            )

        return fallback_jobs

    def _get_market_jobs(self, query: str, location: str, results: int = 30, local_only: Optional[bool] = None) -> List[dict]:
        if local_only is None:
            local_only = self.local_only

        original_mode = self.local_only
        self.local_only = local_only
        try:
            jobs = self._fetch_adzuna_jobs(query=query, location=location, results=results)
        finally:
            self.local_only = original_mode

        if jobs:
            return jobs
        return self._fallback_jobs(query=query, location=location)

    def find_similar_jobs(
        self,
        resume_skills,
        target_role: Optional[str] = None,
        industry: Optional[str] = None,
        experience_level: Optional[str] = None,
        location: str = "United States",
        local_only: Optional[bool] = None,
    ) -> List[dict]:
        query = target_role or "Software Engineer"
        user_skills = self._normalize_resume_skills(resume_skills)
        market_jobs = self._get_market_jobs(query=query, location=location, results=35, local_only=local_only)

        matched_jobs = []
        for job in market_jobs:
            required_skills = self._extract_skills_from_text(job.get("description", "") + " " + job.get("title", ""))

            if industry in self.industry_skills:
                required_skills.update(self.industry_skills[industry].get("must_have", [])[:3])

            if not required_skills:
                required_skills = set(self._fallback_jobs(query, location)[0]["description"].split()[:4])

            matched_skills = user_skills.intersection(required_skills)
            missing_skills = required_skills - user_skills
            fit = len(matched_skills) / len(required_skills) if required_skills else 0.0

            if experience_level and job.get("experience_level") not in [experience_level, "Unknown"]:
                fit *= 0.9

            if industry and job.get("industry") not in [industry, "General"]:
                fit *= 0.9

            if fit >= 0.15:
                matched_jobs.append(
                    {
                        **job,
                        "required_skills": sorted(required_skills),
                        "matched_skills": sorted(matched_skills),
                        "missing_skills": sorted(missing_skills),
                        "match_score": round(fit * 100.0, 2),
                    }
                )

        matched_jobs.sort(key=lambda item: item["match_score"], reverse=True)
        return matched_jobs[:10]

    def get_salary_insights(
        self,
        target_role: str,
        experience_level: str,
        industry: Optional[str] = None,
        location: str = "United States",
        local_only: Optional[bool] = None,
    ) -> dict:
        jobs = self._get_market_jobs(query=target_role, location=location, results=50, local_only=local_only)
        ranges = []
        for job in jobs:
            if job.get("salary_min") and job.get("salary_max"):
                ranges.append((job["salary_min"], job["salary_max"]))

        if ranges:
            mins = [item[0] for item in ranges]
            maxs = [item[1] for item in ranges]
            medians = [statistics.median(item) for item in ranges]
            return {
                "min": int(min(mins)),
                "max": int(max(maxs)),
                "average": int(statistics.mean(medians)),
                "median": int(statistics.median(medians)),
                "growth": self._growth_estimate(experience_level),
                "sample_size": len(ranges),
                "source": jobs[0].get("source", "live_api"),
                "location": location,
            }

        if target_role in self.salary_data and experience_level in self.salary_data[target_role]:
            base = self.salary_data[target_role][experience_level]
        else:
            base = {"min": 70000, "max": 110000, "average": 90000, "growth": self._growth_estimate(experience_level)}

        return {
            "min": int(base["min"]),
            "max": int(base["max"]),
            "average": int(base["average"]),
            "median": int((base["min"] + base["max"]) / 2),
            "growth": int(base.get("growth", self._growth_estimate(experience_level))),
            "sample_size": 0,
            "source": "fallback_dataset",
            "location": location,
        }

    def _growth_estimate(self, experience_level: str) -> int:
        growth_map = {
            "Entry Level": 14,
            "Mid Level": 9,
            "Senior Level": 6,
            "Executive": 4,
        }
        return growth_map.get(experience_level, 7)

    def analyze_job_market_trends(
        self,
        skills,
        location: Optional[str] = None,
        target_role: Optional[str] = None,
        local_only: Optional[bool] = None,
    ) -> dict:
        user_skills = self._normalize_resume_skills(skills)
        query = target_role or "Software Engineer"
        jobs = self._get_market_jobs(query=query, location=location or "United States", results=35, local_only=local_only)

        skill_frequency = Counter()
        for job in jobs:
            job_skills = self._extract_skills_from_text(job.get("description", ""))
            for skill in job_skills:
                skill_frequency[skill] += 1

        trends = {}
        if skill_frequency:
            max_freq = max(skill_frequency.values())
            for skill in sorted(user_skills):
                freq = skill_frequency.get(skill, 0)
                demand_score = int((freq / max_freq) * 100) if max_freq else 0
                trend = "increasing" if demand_score >= 65 else "stable" if demand_score >= 35 else "limited"
                trends[skill] = {
                    "trend": trend,
                    "demand_score": demand_score,
                    "job_openings": int(freq * 100),
                    "growth_rate": int(max(0, min(25, demand_score / 4))),
                    "source": jobs[0].get("source", "live_api"),
                }
            return trends

        for skill in sorted(user_skills):
            role_data = self.job_trends.get(target_role or "", [])
            if role_data:
                demand_score = int(statistics.mean(item["demand"] for item in role_data))
            else:
                demand_score = 55
            trends[skill] = {
                "trend": "stable",
                "demand_score": demand_score,
                "job_openings": 0,
                "growth_rate": int(demand_score / 8),
                "source": "fallback_dataset",
            }
        return trends

    def recommend_career_paths(self, current_skills, target_role=None, experience_level=None):
        user_skills = self._normalize_resume_skills(current_skills)
        career_paths = []

        for role, playbook in ROLE_PLAYBOOKS.items():
            required = set(playbook.get("must_have_skills", []))
            matched = user_skills.intersection(required)
            missing = required - user_skills
            fit = (len(matched) / len(required) * 100.0) if required else 0.0
            salary_ref = self.salary_data.get(role, {}).get(
                "Mid Level", {"min": 85000, "max": 130000, "average": 105000}
            )
            career_paths.append(
                {
                    "title": role,
                    "description": f"{role} path based on market skill expectations.",
                    "required_skills": sorted(required),
                    "growth_potential": "High" if fit >= 50 else "Medium",
                    "avg_salary": f"${salary_ref['average']:,}",
                    "time_to_transition": "3-9 months" if len(missing) <= 3 else "6-12 months",
                    "fit_score": round(fit, 2),
                    "matched_skills": sorted(matched),
                    "missing_skills": sorted(missing),
                    "skills_gap": len(missing),
                }
            )

        career_paths.sort(key=lambda item: item["fit_score"], reverse=True)
        return career_paths[:5]

    def generate_job_search_strategy(self, skills, target_role, experience_level, location="United States"):
        role = target_role or "Software Engineer"
        playbook = ROLE_PLAYBOOKS.get(role, {})
        skill_gaps = [
            skill for skill in playbook.get("must_have_skills", []) if skill not in self._normalize_resume_skills(skills)
        ]
        return {
            "target_companies": ["Large Enterprise", "Growth-stage Startup", "Mid-size Product Company"],
            "networking_suggestions": [
                f"Connect with 10 professionals in {role} roles on LinkedIn.",
                "Request 3 informational conversations with hiring managers or team leads.",
                "Share one portfolio artifact weekly with role-relevant communities.",
            ],
            "application_timeline": {
                "week_1": "Tailor resume versions to 2-3 role variants.",
                "week_2": "Submit 10 high-fit applications and track response rate.",
                "week_3": "Refine interview stories and close top skill gap.",
                "week_4": "Follow up, iterate resume bullets, and expand target list.",
            },
            "skill_priorities": skill_gaps[:5],
            "platform_recommendations": [
                {"platform": "LinkedIn", "focus": "Professional networking and role-based alerts"},
                {"platform": "Indeed", "focus": "Broad role discovery and salary filtering"},
                {"platform": "Glassdoor", "focus": "Compensation and interview signal"},
            ],
            "location": location,
            "experience_level": experience_level,
        }


def find_similar_jobs(
    resume_skills,
    target_role: Optional[str] = None,
    industry: Optional[str] = None,
    experience_level: Optional[str] = None,
    location: str = "United States",
    local_only: bool = False,
):
    matcher = JobMatcher(local_only=local_only)
    return matcher.find_similar_jobs(resume_skills, target_role, industry, experience_level, location, local_only)


def get_salary_insights(
    target_role: str,
    experience_level: str,
    industry: Optional[str] = None,
    location: str = "United States",
    local_only: bool = False,
):
    matcher = JobMatcher(local_only=local_only)
    return matcher.get_salary_insights(target_role, experience_level, industry, location, local_only)


def get_career_recommendations(current_skills, target_role=None, experience_level=None):
    matcher = JobMatcher()
    return matcher.recommend_career_paths(current_skills, target_role, experience_level)


def generate_market_report(
    skills,
    location: Optional[str] = None,
    industry: Optional[str] = None,
    target_role: Optional[str] = None,
    local_only: bool = False,
):
    matcher = JobMatcher(local_only=local_only)
    role = target_role or "Software Engineer"
    salary_snapshot = matcher.get_salary_insights(
        target_role=role,
        experience_level="Mid Level",
        industry=industry,
        location=location or "United States",
        local_only=local_only,
    )
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "skill_demand": matcher.analyze_job_market_trends(skills, location, role, local_only),
        "salary_snapshot": salary_snapshot,
        "data_source": salary_snapshot.get("source", "fallback_dataset"),
        "hiring_summary": {
            "market_role": role,
            "location": location or "United States",
            "source": salary_snapshot.get("source", "fallback_dataset"),
        },
    }
