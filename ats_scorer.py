import re
from collections import Counter
from typing import Dict, List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from config import ATS_KEYWORDS

ATS_COMPATIBILITY_WEIGHTS = {
    "text_extraction_quality": 0.20,
    "document_structure": 0.20,
    "section_identification": 0.15,
    "formatting_readability": 0.15,
    "keyword_relevance": 0.20,
    "contact_document_signals": 0.10,
}


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
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.stop_words = set()

    def calculate_realistic_ats_score(
        self,
        resume_text: str,
        job_description: str,
        target_role: Optional[str] = None,
        location: Optional[str] = None,
        return_details: bool = False,
        parser_metadata: Optional[dict] = None,
    ):
        if not resume_text:
            if return_details:
                return {
                    "score": None,
                    "component_scores": {},
                    "matched_keywords": [],
                    "missing_keywords": [],
                    "suggestions": ["No readable resume text was extracted; ATS text compatibility cannot be assessed."],
                    "explanations": {},
                    "before_after_examples": [],
                    "score_status": "not assessable: no extracted resume text",
                    "methodology": {
                        "type": "ATS-oriented text heuristic",
                        "weights": {},
                        "configured_weights": ATS_COMPATIBILITY_WEIGHTS,
                        "description": "No ATS compatibility score is produced without extracted resume text.",
                        "limitations": "This is not a prediction of ATS acceptance.",
                    },
                }
            return None, ["No readable resume text was extracted."]

        keyword_details = self._keyword_match_details(resume_text, job_description or "") if job_description else {
            "job_keywords": [], "matched_keywords": [], "missing_keywords": [], "coverage_percent": None,
        }
        lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
        printable = sum(character.isprintable() for character in resume_text) / max(1, len(resume_text))
        char_count = (parser_metadata or {}).get("char_count", len(resume_text))
        scores = {
            "text_extraction_quality": round(min(100.0, printable * 100 * (1.0 if char_count >= 120 else max(.35, char_count / 120))), 2),
            "document_structure": self._score_formatting(resume_text),
            "section_identification": self._score_section_structure(resume_text),
            "formatting_readability": self._score_readability(resume_text),
            "keyword_relevance": keyword_details["coverage_percent"],
            "contact_document_signals": self._score_contact_info(resume_text),
        }
        ats_weights = ATS_COMPATIBILITY_WEIGHTS
        available = [key for key, score in scores.items() if score is not None]
        weight_total = sum(ats_weights[key] for key in available)
        final_score = round(sum(float(scores[key]) * ats_weights[key] for key in available) / weight_total, 2) if weight_total else None
        used_weights = {key: round(ats_weights[key] / weight_total, 4) for key in available} if weight_total else {}

        suggestions = self._generate_suggestions(scores, keyword_details)

        if not return_details:
            return final_score, suggestions

        details = {
            "score": final_score,
            "score_status": "heuristic text compatibility; not a prediction of acceptance by a specific ATS",
            "methodology": {
                "type": "ATS-oriented text heuristic",
                "weights": used_weights,
                "configured_weights": ats_weights,
                "description": "ATS compatibility heuristic: extraction quality 20%, document structure 20%, section identification 15%, text readability 15%, job keyword relevance 20%, contact/document signals 10%. Unavailable job-keyword relevance is omitted and remaining weights are renormalized.",
                "limitations": "This evaluates extracted text signals, not layout fidelity or a specific applicant tracking system. It is not a prediction of ATS acceptance or hiring likelihood.",
            },
            "component_scores": {k: round(v, 2) if v is not None else None for k, v in scores.items()},
            "matched_keywords": keyword_details["matched_keywords"],
            "missing_keywords": keyword_details["missing_keywords"],
            "keyword_coverage_percent": round(keyword_details["coverage_percent"], 2) if keyword_details["coverage_percent"] is not None else None,
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
            score -= 8
        if re.search(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", text):
            score -= 20
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
        text = resume_text or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        section_patterns = [
            r"^(?:professional\s+)?(?:summary|objective|profile)\s*:?[ \t]*$",
            r"^(?:work\s+)?(?:experience|employment|work\s+history)\s*:?[ \t]*$",
            r"^(?:education|academic\s+background)\s*:?[ \t]*$",
            r"^(?:technical\s+)?skills(?:\s+and\s+competencies)?\s*:?[ \t]*$",
        ]
        found = sum(
            1
            for pattern in section_patterns
            if any(re.match(pattern, line, re.IGNORECASE) for line in lines)
        )
        if re.search(r"\b(email|phone|linkedin)\b", text, re.IGNORECASE):
            found += 1
        completeness = found / 5
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
        if scores.get("text_extraction_quality", 0) < 70:
            suggestions.append("Review the uploaded file's extracted text; missing or garbled text can reduce parser access.")
        if scores.get("section_identification", 0) < 60:
            suggestions.append("Use clear, conventional headings for the sections that apply to your background.")
        if scores.get("formatting_readability", 0) < 65:
            suggestions.append("Review extracted text for long, repeated, or difficult-to-scan lines.")
        if scores.get("contact_document_signals", 0) < 50:
            suggestions.append("If appropriate for your privacy and application context, include one clear professional contact route.")
        if scores.get("keyword_relevance") is not None and scores["keyword_relevance"] < 40:
            suggestions.append("Check whether the posting's relevant terms appear where supported by your experience; do not add keywords you cannot substantiate.")
        if not suggestions:
            suggestions.append("No major text-structure issue was detected. Check the original document layout and application instructions manually.")

        if keyword_details.get("missing_keywords"):
            top_missing = ", ".join(keyword_details["missing_keywords"][:5])
            suggestions.append(f"Posting terms not found in extracted resume text (review for relevance): {top_missing}.")

        return suggestions

    def _component_explanations(
        self,
        scores: Dict[str, float],
        keyword_details: dict,
        target_role: Optional[str],
        location: Optional[str],
    ) -> Dict[str, dict]:
        descriptions = {
            "text_extraction_quality": "Text availability and basic character quality from the parser output.",
            "document_structure": "Control-character and broad text-structure signals; visual layout is not inspected.",
            "section_identification": "Recognition of common resume headings in extracted text.",
            "formatting_readability": "Sentence length and list/bullet signals in the extracted text.",
            "keyword_relevance": "Exact posting-term coverage in extracted text, capped at 20% of the total score.",
            "contact_document_signals": "Email, phone, and professional-link signals; absence is not proof the source document omitted them.",
        }
        return {
            key: {
                "score": round(value, 2) if value is not None else None,
                "what_matched": keyword_details.get("matched_keywords", [])[:10] if key == "keyword_relevance" else [],
                "what_is_missing": keyword_details.get("missing_keywords", [])[:10] if key == "keyword_relevance" else [],
                "why_it_matters": descriptions.get(key, "Text compatibility signal."),
            }
            for key, value in scores.items()
        }

    def _before_after_examples(self, keyword_details: dict, target_role: Optional[str]) -> List[dict]:
        # Avoid generic rewrites that could insert achievements or metrics the
        # candidate did not provide. Grounded suggestions can be added here.
        return []


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
    parser_metadata: Optional[dict] = None,
) -> dict:
    scorer = RealisticATSScorer()
    return scorer.calculate_realistic_ats_score(
        resume_text=resume_text,
        job_description=job_description,
        target_role=target_role,
        location=location,
        return_details=True,
        parser_metadata=parser_metadata,
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
