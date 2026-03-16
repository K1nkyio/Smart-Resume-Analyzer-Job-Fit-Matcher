import re
from collections import Counter
from typing import Dict, List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from config import ATS_KEYWORDS


def _safe_word_tokenize(text: str) -> List[str]:
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"\b\w+\b", text)


def _safe_sent_tokenize(text: str) -> List[str]:
    try:
        return sent_tokenize(text)
    except LookupError:
        return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


class RealisticATSScorer:
    def __init__(self):
        self.action_verbs = set(ATS_KEYWORDS["action_verbs"])
        self.quantifiable_terms = ATS_KEYWORDS["quantifiable_terms"]
        self._ensure_nltk_resources()
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.stop_words = set()

    def _ensure_nltk_resources(self):
        required_resources = [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/stopwords", "stopwords"),
        ]
        for resource_path, package_name in required_resources:
            try:
                nltk.data.find(resource_path)
            except Exception:
                try:
                    nltk.download(package_name, quiet=True, force=True)
                except Exception:
                    pass

    def calculate_realistic_ats_score(
        self,
        resume_text: str,
        job_description: str,
        target_role: Optional[str] = None,
        location: Optional[str] = None,
        return_details: bool = False,
    ):
        if not resume_text or not job_description:
            if return_details:
                return {
                    "score": 0.0,
                    "component_scores": {},
                    "matched_keywords": [],
                    "missing_keywords": [],
                    "suggestions": ["Provide both resume text and job description."],
                    "explanations": {},
                    "before_after_examples": [],
                }
            return 0.0, ["Provide both resume text and job description."]

        keyword_details = self._keyword_match_details(resume_text, job_description)

        scores = {
            "keyword_density": self._score_keyword_density(keyword_details),
            "formatting": self._score_formatting(resume_text),
            "contact_info": self._score_contact_info(resume_text),
            "section_structure": self._score_section_structure(resume_text),
            "quantifiable_achievements": self._score_quantifiable_achievements(resume_text),
            "action_verbs": self._score_action_verbs(resume_text),
            "length_appropriateness": self._score_length(resume_text),
            "readability": self._score_readability(resume_text),
        }

        final_score = (
            scores["keyword_density"] * 0.30
            + scores["section_structure"] * 0.20
            + scores["formatting"] * 0.15
            + scores["contact_info"] * 0.10
            + scores["quantifiable_achievements"] * 0.10
            + scores["action_verbs"] * 0.05
            + scores["length_appropriateness"] * 0.05
            + scores["readability"] * 0.05
        )
        final_score = round(min(final_score, 100.0), 2)

        suggestions = self._generate_suggestions(scores, keyword_details)

        if not return_details:
            return final_score, suggestions

        details = {
            "score": final_score,
            "component_scores": {k: round(v, 2) for k, v in scores.items()},
            "matched_keywords": keyword_details["matched_keywords"],
            "missing_keywords": keyword_details["missing_keywords"],
            "keyword_coverage_percent": round(keyword_details["coverage_percent"], 2),
            "suggestions": suggestions,
            "explanations": self._component_explanations(
                scores=scores,
                keyword_details=keyword_details,
                target_role=target_role,
                location=location,
            ),
            "before_after_examples": self._before_after_examples(keyword_details, target_role),
        }
        return details

    def _extract_job_keywords(self, job_description: str) -> List[str]:
        words = _safe_word_tokenize((job_description or "").lower())
        words = [word for word in words if word.isalnum() and len(word) > 2 and word not in self.stop_words]
        word_freq = Counter(words)

        sentences = _safe_sent_tokenize(job_description)
        phrases = []
        phrase_patterns = [
            r"experience with ([A-Za-z0-9\+#\.\-/ ]+)",
            r"proficiency in ([A-Za-z0-9\+#\.\-/ ]+)",
            r"knowledge of ([A-Za-z0-9\+#\.\-/ ]+)",
            r"familiar with ([A-Za-z0-9\+#\.\-/ ]+)",
        ]
        for sentence in sentences:
            for pattern in phrase_patterns:
                for match in re.findall(pattern, sentence, re.IGNORECASE):
                    phrase = match.strip().lower()
                    if 2 <= len(phrase.split()) <= 4:
                        phrases.append(phrase)

        important_words = [word for word, freq in word_freq.most_common(20) if freq >= 2]
        merged = list(dict.fromkeys(important_words + phrases))
        return merged[:30]

    def _keyword_match_details(self, resume_text: str, job_description: str) -> dict:
        job_keywords = self._extract_job_keywords(job_description)
        resume_lower = (resume_text or "").lower()

        matched_keywords = [kw for kw in job_keywords if kw in resume_lower]
        missing_keywords = [kw for kw in job_keywords if kw not in resume_lower]
        coverage = (len(matched_keywords) / len(job_keywords) * 100.0) if job_keywords else 0.0

        return {
            "job_keywords": job_keywords,
            "matched_keywords": matched_keywords,
            "missing_keywords": missing_keywords,
            "coverage_percent": coverage,
        }

    def _score_keyword_density(self, keyword_details: dict) -> float:
        coverage = keyword_details["coverage_percent"]
        if coverage >= 80:
            return 95.0
        if coverage >= 60:
            return 85.0
        if coverage >= 40:
            return 70.0
        if coverage >= 25:
            return 55.0
        return 30.0

    def _score_formatting(self, resume_text: str) -> float:
        score = 100.0
        text = resume_text or ""
        if "\t" in text:
            score -= 12
        if re.search(r"[^\x00-\x7F]", text):
            score -= 8
        if len(re.findall(r"\n\n", text)) < 2:
            score -= 8
        if re.search(r"\|", text):
            score -= 10
        return max(20.0, score)

    def _score_contact_info(self, resume_text: str) -> float:
        text = resume_text or ""
        score = 0.0
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
            score += 40
        if re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}", text):
            score += 40
        if re.search(r"linkedin\.com/in/[\w\-]+", text, re.IGNORECASE):
            score += 10
        if re.search(r"[A-Z][a-z]+,\s*[A-Z]{2}", text):
            score += 10
        return score

    def _score_section_structure(self, resume_text: str) -> float:
        text = (resume_text or "").lower()
        required_sections = {
            "contact": r"(email|phone|linkedin)",
            "summary": r"(summary|objective|profile)",
            "experience": r"(experience|employment|work)",
            "education": r"(education|degree|university)",
            "skills": r"(skills|competencies|technical)",
        }
        found = sum(1 for _, pattern in required_sections.items() if re.search(pattern, text))
        completeness = found / len(required_sections)
        if completeness >= 0.8:
            return 95.0
        if completeness >= 0.6:
            return 80.0
        if completeness >= 0.4:
            return 60.0
        return 30.0

    def _score_quantifiable_achievements(self, resume_text: str) -> float:
        text = resume_text or ""
        patterns = [
            r"\d+%",
            r"\$[\d,]+",
            r"\d+\+?\s*(million|thousand|k|m)\b",
            r"(increased|improved|reduced|grew|generated|saved)\s+.*?\d+",
            r"\d+\s*(team|people|employees|members)",
        ]
        count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in patterns)
        if count >= 8:
            return 95.0
        if count >= 5:
            return 80.0
        if count >= 3:
            return 65.0
        if count >= 1:
            return 45.0
        return 20.0

    def _score_action_verbs(self, resume_text: str) -> float:
        words = set(_safe_word_tokenize((resume_text or "").lower()))
        verb_count = len(words.intersection(self.action_verbs))
        if verb_count >= 10:
            return 95.0
        if verb_count >= 7:
            return 80.0
        if verb_count >= 4:
            return 65.0
        if verb_count >= 2:
            return 50.0
        return 25.0

    def _score_length(self, resume_text: str) -> float:
        word_count = len((resume_text or "").split())
        if 300 <= word_count <= 700:
            return 100.0
        if 200 <= word_count < 300:
            return 75.0
        if 700 < word_count <= 900:
            return 80.0
        if word_count > 900:
            return 50.0
        return 30.0

    def _score_readability(self, resume_text: str) -> float:
        text = resume_text or ""
        sentences = _safe_sent_tokenize(text)
        if not sentences:
            return 0.0
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        has_bullets = bool(re.search(r"(^[-*]\s)|([•])", text, re.MULTILINE))

        score = 50.0
        if 10 <= avg_sentence_length <= 20:
            score += 25.0
        elif avg_sentence_length < 10:
            score += 15.0
        else:
            score += 5.0
        if has_bullets:
            score += 25.0
        return min(score, 100.0)

    def _generate_suggestions(self, scores: Dict[str, float], keyword_details: dict) -> List[str]:
        suggestions = []
        if scores["keyword_density"] < 60:
            suggestions.append(
                "Add more exact role keywords from the job posting, especially in Experience and Skills sections."
            )
        if scores["contact_info"] < 80:
            suggestions.append("Include email, phone, and LinkedIn in a clear header section.")
        if scores["section_structure"] < 70:
            suggestions.append("Use standard sections: Summary, Experience, Education, Skills.")
        if scores["formatting"] < 70:
            suggestions.append("Use ATS-safe formatting: plain text bullets, no tables, no unusual symbols.")
        if scores["quantifiable_achievements"] < 60:
            suggestions.append("Add measurable outcomes, e.g. percent improvement, savings, or delivery metrics.")
        if scores["action_verbs"] < 60:
            suggestions.append("Start bullets with strong verbs like implemented, optimized, led, improved.")
        if scores["readability"] < 70:
            suggestions.append("Shorten long sentences and keep bullets concise and scannable.")
        if not suggestions:
            suggestions.append("ATS profile is strong. Focus on role-specific tailoring for each application.")

        if keyword_details["missing_keywords"]:
            top_missing = ", ".join(keyword_details["missing_keywords"][:5])
            suggestions.append(f"Priority missing keywords to integrate: {top_missing}.")

        return suggestions

    def _component_explanations(
        self,
        scores: Dict[str, float],
        keyword_details: dict,
        target_role: Optional[str],
        location: Optional[str],
    ) -> Dict[str, dict]:
        role = target_role or "Target role"
        location_context = location or "target market"
        explanations = {
            "keyword_density": {
                "score": round(scores["keyword_density"], 2),
                "what_matched": keyword_details["matched_keywords"][:10],
                "what_is_missing": keyword_details["missing_keywords"][:10],
                "why_it_matters": f"Keyword relevance drives ATS ranking for {role} roles in {location_context}.",
            },
            "section_structure": {
                "score": round(scores["section_structure"], 2),
                "what_matched": ["standard section headings detected" if scores["section_structure"] >= 60 else ""],
                "what_is_missing": ["missing one or more standard sections"] if scores["section_structure"] < 80 else [],
                "why_it_matters": "Structured resumes are parsed more reliably by ATS parsers.",
            },
            "quantifiable_achievements": {
                "score": round(scores["quantifiable_achievements"], 2),
                "what_matched": ["numeric metrics found"] if scores["quantifiable_achievements"] >= 45 else [],
                "what_is_missing": ["limited measurable business impact"] if scores["quantifiable_achievements"] < 60 else [],
                "why_it_matters": "Recruiters prioritize evidence of outcomes, not only responsibilities.",
            },
            "formatting": {
                "score": round(scores["formatting"], 2),
                "what_matched": ["ATS-safe signals"] if scores["formatting"] >= 70 else [],
                "what_is_missing": ["formatting simplification needed"] if scores["formatting"] < 70 else [],
                "why_it_matters": "Complex layout can reduce parser accuracy and keyword detection.",
            },
        }
        return explanations

    def _before_after_examples(self, keyword_details: dict, target_role: Optional[str]) -> List[dict]:
        role = target_role or "target role"
        keyword = keyword_details["missing_keywords"][0] if keyword_details["missing_keywords"] else "relevant keyword"
        return [
            {
                "focus": "Action + metric",
                "before": "Responsible for improving system performance.",
                "after": "Optimized API query paths and reduced p95 latency by 34% across production endpoints.",
            },
            {
                "focus": "Keyword integration",
                "before": f"Worked on projects related to {role}.",
                "after": f"Delivered {keyword}-focused work aligned to {role} requirements with measurable outcomes.",
            },
            {
                "focus": "Impact clarity",
                "before": "Assisted the team with reporting tasks.",
                "after": "Built automated reporting workflow that cut manual reporting time from 6 hours to 1 hour weekly.",
            },
        ]


def calculate_ats_score(
    resume_text: str,
    job_description: str,
    skill_weight: int = 40,
    experience_weight: int = 30,
    education_weight: int = 20,
    keywords_weight: int = 10,
):
    scorer = RealisticATSScorer()
    return scorer.calculate_realistic_ats_score(resume_text, job_description)


def calculate_ats_score_detailed(
    resume_text: str,
    job_description: str,
    target_role: Optional[str] = None,
    location: Optional[str] = None,
) -> dict:
    scorer = RealisticATSScorer()
    return scorer.calculate_realistic_ats_score(
        resume_text=resume_text,
        job_description=job_description,
        target_role=target_role,
        location=location,
        return_details=True,
    )


def get_ats_suggestions(resume_text: str, job_description: str) -> List[str]:
    scorer = RealisticATSScorer()
    _, suggestions = scorer.calculate_realistic_ats_score(resume_text, job_description)
    return suggestions


def debug_ats_scoring(resume_text: str, job_description: str):
    details = calculate_ats_score_detailed(resume_text, job_description)
    print("=== ATS DEBUG ===")
    print(f"Final score: {details['score']}")
    print("Component scores:")
    for component, score in details.get("component_scores", {}).items():
        print(f"  {component}: {score}")
    print(f"Matched keywords: {details.get('matched_keywords', [])[:10]}")
    print(f"Missing keywords: {details.get('missing_keywords', [])[:10]}")
    return details


class ATSOptimizer:
    def __init__(self):
        self.scorer = RealisticATSScorer()

    def get_optimization_report(self, resume_text: str, job_description: str, target_role: Optional[str] = None):
        details = self.scorer.calculate_realistic_ats_score(
            resume_text=resume_text,
            job_description=job_description,
            target_role=target_role,
            return_details=True,
        )
        return {
            "current_score": details["score"],
            "suggestions": details["suggestions"],
            "improvement_areas": details["explanations"],
            "examples": details["before_after_examples"],
            "priority_actions": details["suggestions"][:3],
        }
