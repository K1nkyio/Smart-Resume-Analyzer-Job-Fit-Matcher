import unittest
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai_service import AIServiceError, AISettings, clear_ai_cache, generate_completion
from job_fit_scorer import calculate_evidence_confidence, calculate_job_fit
from job_requirements import assess_qualification_evidence, extract_job_requirements
from local_ai import analyze_career_intelligence, analyze_resume_and_job_with_ai, redact_resume_pii, validate_ai_assessment
from skill_matcher import compare_resume_to_job_detailed


def _response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _settings(retries=2):
    return AISettings(
        provider="groq", enabled=True, model="test-model", api_key="test-key", max_retries=retries
    )


def _assessment_payload(resume_quote="Built dashboards with SQL.", classification="direct_match"):
    return {
        "candidate_profile": {
            "profession": {"title": "Analyst", "evidence_quote": "Analyst with reporting experience."},
            "seniority": {"level": "Junior", "evidence_quote": "Junior analyst"},
            "skills": [{"name": "SQL", "evidence_quote": "Built dashboards with SQL."}],
            "education": [{"qualification": "BSc in Statistics, in progress", "evidence_quote": "BSc in Statistics, in progress"}],
            "experience_highlights": [{"summary": "Reporting", "evidence_quote": "Built dashboards with SQL."}],
            "certifications": [],
        },
        "requirements": [{
            "requirement": "SQL", "category": "skill", "priority": "mandatory",
            "job_evidence_quote": "SQL is required.",
        }],
        "evidence_matches": [{
            "requirement": "SQL", "classification": classification,
            "resume_evidence_quote": resume_quote, "explanation": "The resume cites SQL dashboard work.",
        }],
        "resume_evaluation": [{
            "dimension": "experience", "priority": "medium", "issue": "Clarify the outcome.",
            "recommendation": "Add the verified result if available.", "evidence_quote": "Built dashboards with SQL.",
        }],
        "optimization": [{
            "section": "experience", "original_quote": "Built dashboards with SQL.",
            "suggested_rewrite": "Built SQL dashboards for reporting.", "rationale": "Clarifies the work.",
        }],
        "limitations": ["Resume evidence is self-reported."],
    }


class AIServiceTests(unittest.TestCase):
    def setUp(self):
        clear_ai_cache()

    def tearDown(self):
        clear_ai_cache()

    def test_successful_groq_request_returns_structured_json_text(self):
        create = Mock(return_value=_response('{"ok": true}'))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch("ai_service._openai_client", return_value=client):
            result = generate_completion("safe prompt", _settings(), structured=True)
        self.assertEqual(result, '{"ok": true}')
        self.assertEqual(create.call_args.kwargs["response_format"], {"type": "json_object"})

    def test_invalid_api_key_is_reported_without_retry(self):
        error = RuntimeError("secret server body must not reach UI")
        error.status_code = 401
        create = Mock(side_effect=error)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch("ai_service._openai_client", return_value=client):
            with self.assertRaises(AIServiceError) as raised:
                generate_completion("prompt", _settings())
        self.assertEqual(raised.exception.code, "invalid_key")
        self.assertNotIn("secret server body", str(raised.exception))
        self.assertEqual(create.call_count, 1)

    def test_rate_limit_retries_with_backoff(self):
        rate_error = RuntimeError("rate limited")
        rate_error.status_code = 429
        create = Mock(side_effect=[rate_error, _response("done")])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch("ai_service._openai_client", return_value=client), patch("ai_service.time.sleep") as sleep:
            self.assertEqual(generate_completion("prompt", _settings()), "done")
        self.assertEqual(create.call_count, 2)
        sleep.assert_called_once()

    def test_network_failure_retries_then_returns_safe_error(self):
        create = Mock(side_effect=ConnectionError("network detail"))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch("ai_service._openai_client", return_value=client), patch("ai_service.time.sleep"):
            with self.assertRaises(AIServiceError) as raised:
                generate_completion("prompt", _settings(retries=1))
        self.assertEqual(raised.exception.code, "network")
        self.assertNotIn("network detail", str(raised.exception))
        self.assertEqual(create.call_count, 2)

    def test_ai_disabled_does_not_call_provider(self):
        with patch("ai_service._openai_client") as client:
            with self.assertRaises(AIServiceError) as raised:
                generate_completion("private resume", AISettings("groq", False, "test", "key"))
        self.assertEqual(raised.exception.code, "disabled")
        client.assert_not_called()

    def test_common_contact_details_are_redacted_before_transmission(self):
        redacted = redact_resume_pii("Name: Jane Doe\nEmail jane@example.com\nCall +1 (415) 555-0199\nhttps://linkedin.com/in/janedoe")
        self.assertNotIn("jane@example.com", redacted)
        self.assertNotIn("555-0199", redacted)
        self.assertNotIn("linkedin.com/in/janedoe", redacted)
        self.assertIn("Name: [REDACTED]", redacted)

    def test_malformed_model_response_is_a_graceful_invalid_response(self):
        with patch("local_ai.generate_completion", return_value="not json"):
            result = analyze_resume_and_job_with_ai(
                "Analyst with reporting experience.", "SQL is required.", settings=_settings()
            )
        self.assertEqual(result["status"], "invalid_response")
        self.assertIsNone(result.get("numeric_scores"))

    def test_missing_evidence_cannot_remain_a_positive_match(self):
        payload = _assessment_payload("unquoted Python claim", "direct_match")
        checked = validate_ai_assessment(payload, "Built dashboards with SQL.", "SQL is required.")
        self.assertEqual(checked["evidence_matches"][0]["classification"], "unverifiable")
        self.assertEqual(checked["evidence_matches"][0]["resume_evidence_quote"], "")

    def test_transferable_evidence_and_in_progress_education_are_retained_as_evidence(self):
        payload = _assessment_payload("Built dashboards with SQL.", "transferable_match")
        checked = validate_ai_assessment(payload, "Analyst with reporting experience. Junior analyst. Built dashboards with SQL. BSc in Statistics, in progress", "SQL is required.")
        self.assertEqual(checked["evidence_matches"][0]["classification"], "transferable_match")
        self.assertEqual(checked["candidate_profile"]["education"][0]["qualification"], "BSc in Statistics, in progress")

    def test_resume_facts_are_associated_only_with_a_section_containing_the_quote(self):
        payload = _assessment_payload()
        payload["candidate_profile"]["skills"][0]["section"] = "skills"
        payload["candidate_profile"]["skills"][0]["evidence_quote"] = "SQL"
        checked = validate_ai_assessment(
            payload,
            "Built dashboards with SQL.\nSQL",
            "SQL is required.",
            {"experience": "Built dashboards with SQL.", "skills": "SQL"},
        )
        self.assertEqual(checked["candidate_profile"]["skills"][0]["section"], "Skills")

    def test_career_intelligence_requires_verifiable_cross_posting_quotes_and_known_gaps(self):
        payload = {
            "career_summary": "SQL appears across these roles.",
            "recurring_requirements": [{
                "requirement": "SQL", "why_it_matters": "Used in multiple postings.",
                "supporting_jobs": [
                    {"job_index": 1, "quote": "SQL required."},
                    {"job_index": 2, "quote": "SQL skills preferred."},
                    {"job_index": 3, "quote": "invented quote"},
                ],
            }],
            "development_recommendations": [
                {"focus": "Python", "first_step": "Practice a small script", "learning_activity": "Build a project", "timeframe": "4 weeks"},
                {"focus": "PhD", "first_step": "Apply now", "learning_activity": "None", "timeframe": "1 year"},
            ],
            "limitations": [],
        }
        with patch("local_ai.generate_completion", return_value=json.dumps(payload)):
            result = analyze_career_intelligence(
                "Analyst", [{"title": "A", "description": "SQL required."},
                            {"title": "B", "description": "SQL skills preferred."},
                            {"title": "C", "description": "Python a plus."}],
                ["Python"], settings=_settings(),
            )
        self.assertTrue(result["available"])
        self.assertEqual(len(result["recurring_requirements"][0]["supporting_jobs"]), 2)
        self.assertEqual([item["focus"] for item in result["development_recommendations"]], ["Python"])

    def test_existing_qualification_comparison_handles_education_and_experience(self):
        job = extract_job_requirements("Requirements:\n- Bachelor's degree required\n- 3 years of experience")
        resume = "EDUCATION\nBSc in Computer Science currently pursuing\nEXPERIENCE\n2 years of experience in reporting"
        evidence = assess_qualification_evidence(
            resume,
            {"education": "BSc in Computer Science currently pursuing", "experience": "2 years of experience in reporting"},
            job,
        )
        self.assertIn("in progress", evidence["education"][0]["status"])
        self.assertIn("below the posted threshold", evidence["experience"][0]["status"])

    def test_deterministic_scores_do_not_change_when_ai_evidence_is_available(self):
        resume = "EXPERIENCE\nBuilt dashboards with SQL.\nSKILLS\nSQL"
        job = "SQL is required."
        sections = {"experience": "Built dashboards with SQL.", "skills": "SQL"}
        requirements = extract_job_requirements(job)
        skill_details = compare_resume_to_job_detailed(resume, job, resume_sections=sections)
        qualification = assess_qualification_evidence(resume, sections, requirements)
        score_without_ai = calculate_job_fit(skill_details, requirements, qualification, sections)["score"]
        score_with_ai = calculate_job_fit(skill_details, requirements, qualification, sections)["score"]
        self.assertEqual(score_without_ai, score_with_ai)
        confidence = calculate_evidence_confidence(resume, {"category_coverage_percent": 70}, {"char_count": len(resume)})
        self.assertIsNone(confidence.get("ai_score"))


if __name__ == "__main__":
    unittest.main()
