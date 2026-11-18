import unittest
import tempfile
import importlib.util
import sqlite3
from contextlib import closing
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from ats_scorer import RealisticATSScorer
from job_matcher import JobMatcher
from job_requirements import assess_qualification_evidence, extract_job_requirements
from local_ai import generate_career_coach_response, generate_tailored_resume_suggestions, semantic_match_score
from resume_parser import extract_text_with_metadata, validate_resume_upload
from resume_assessor import assess_resume_quality
from skill_matcher import compare_resume_to_job_detailed
from workflow_store import WorkflowStore, build_resume_docx, build_resume_pdf
from docx import Document


class UploadedFile(BytesIO):
    def __init__(self, name, content):
        super().__init__(content)
        self.name = name
        self.size = len(content)


class TrustFoundationTests(unittest.TestCase):
    def test_upload_extension_and_size_validation(self):
        validate_resume_upload(UploadedFile("resume.txt", b"experience and skills"))
        with self.assertRaises(ValueError):
            validate_resume_upload(UploadedFile("resume.exe", b"no"))

        too_large = UploadedFile("resume.txt", b"")
        too_large.size = 10 * 1024 * 1024 + 1
        with self.assertRaises(ValueError):
            validate_resume_upload(too_large)

        with self.assertRaisesRegex(ValueError, "valid PDF"):
            validate_resume_upload(UploadedFile("resume.pdf", b"not really a PDF"))

    def test_local_only_market_does_not_create_vacancies(self):
        matcher = JobMatcher(local_only=True)
        self.assertEqual(matcher._get_market_jobs("Data Analyst", "Nairobi"), [])

    def test_configured_salary_reference_is_marked_illustrative(self):
        salary = JobMatcher(local_only=True).get_salary_insights(
            "Data Analyst", "Mid Level", location="Nairobi"
        )
        self.assertEqual(salary["source"], "illustrative_config")

    def test_default_similarity_is_deterministic_and_offline(self):
        score = semantic_match_score("Python SQL data analysis", "Python SQL analytics")
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 100)

    def test_ats_has_no_unverified_before_after_claims(self):
        examples = RealisticATSScorer()._before_after_examples(
            {"missing_keywords": ["Python"]}, "Data Analyst"
        )
        self.assertEqual(examples, [])

    def test_resume_quality_is_distinct_and_gives_fact_safe_actions(self):
        quality = assess_resume_quality(
            "Email: a@example.com\nEXPERIENCE\n- Supported weekly reporting",
            {"experience": "- Supported weekly reporting"},
            {"email": "a@example.com"},
        )
        self.assertEqual(quality["score_type"], "heuristic_document_quality")
        self.assertIn("section_completeness", quality["dimensions"])
        self.assertTrue(any("Do not estimate or invent metrics." in item["how"] for item in quality["recommendations"]))

    def test_docx_export_contains_only_supplied_resume_content(self):
        payload = build_resume_docx(
            {"email": "person@example.com"},
            {"experience": "- Built internal reporting tools", "skills": "Python, SQL"},
        )
        doc = Document(BytesIO(payload))
        extracted = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        self.assertIn("person@example.com", extracted)
        self.assertIn("Built internal reporting tools", extracted)
        self.assertIn("Python, SQL", extracted)
        self.assertNotIn("34%", extracted)

    @unittest.skipUnless(importlib.util.find_spec("reportlab"), "reportlab is installed with core requirements")
    def test_pdf_export_contains_supplied_resume_content(self):
        import PyPDF2

        payload = build_resume_pdf(
            {"name": "Zoë Example", "email": "ada@example.com"},
            {"experience": "Built reporting tools", "skills": "Python, SQL"},
        )
        self.assertTrue(payload.startswith(b"%PDF-"))
        reader = PyPDF2.PdfReader(BytesIO(payload))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.assertIn("Zoë Example", text)
        self.assertIn("Built reporting tools", text)

    def test_career_coach_uses_analysis_summary_not_raw_resume(self):
        analysis = {
            "resume_text": "RAW SENSITIVE RESUME BODY",
            "overall_score": 72,
            "skill_details": {
                "target_role_context": "Data Analyst",
                "matched_details": [{"skill": "SQL"}],
                "missing_details": [{"skill": "Tableau"}],
            },
            "resume_quality": {"score": 80, "recommendations": []},
            "ats_details": {"suggestions": []},
            "qualification_evidence": {
                "experience": [{"requirement": "5 years", "status": "cannot determine"}]
            },
        }
        with patch("local_ai.ollama_generate", return_value="Focus on SQL evidence.") as generate:
            answer = generate_career_coach_response("What should I improve?", analysis)
        self.assertEqual(answer, "Focus on SQL evidence.")
        prompt = generate.call_args.args[0]
        self.assertIn("Tableau", prompt)
        self.assertIn("cannot determine", prompt)
        self.assertNotIn("RAW SENSITIVE RESUME BODY", prompt)

    def test_job_requirements_distinguish_explicit_required_preferred_and_evidence(self):
        details = compare_resume_to_job_detailed(
            "EXPERIENCE\nBuilt dashboards using Python.\nSKILLS\nSQL",
            "Python is required. SQL is preferred. Tableau experience.",
            resume_sections={"experience": "Built dashboards using Python.", "skills": "SQL"},
        )
        matched = {item["skill"]: item for item in details["matched_details"]}
        self.assertEqual(matched["Python"]["requirement_type"], "required")
        self.assertEqual(matched["Python"]["evidence_level"], "demonstrated")
        self.assertEqual(matched["SQL"]["requirement_type"], "preferred")
        self.assertEqual(matched["SQL"]["evidence_level"], "mentioned without experience or project evidence")
        missing = {item["skill"]: item for item in details["missing_details"]}
        self.assertEqual(missing["Tableau"]["requirement_type"], "unspecified")

    def test_missing_required_skill_has_more_weight_than_preferred_skill(self):
        details = compare_resume_to_job_detailed(
            "SKILLS\nSQL",
            "Python is required. SQL is preferred.",
            resume_sections={"skills": "SQL"},
        )
        self.assertEqual(details["score_components"]["direct_match"], 33.33)
        self.assertEqual(details["score_components"]["required_skill_coverage"], 0.0)

    def test_related_technology_is_reported_without_direct_match_credit(self):
        details = compare_resume_to_job_detailed(
            "EXPERIENCE\nBuilt reporting pipelines with PostgreSQL.",
            "SQL is required.",
            resume_sections={"experience": "Built reporting pipelines with PostgreSQL."},
        )
        self.assertNotIn("SQL", details["matched_skills"])
        missing = next(item for item in details["missing_details"] if item["skill"] == "SQL")
        self.assertEqual(missing["related_resume_skills"], ["PostgreSQL"])
        self.assertIn("direct requirement not demonstrated", missing["evidence_level"])

    def test_job_description_extractor_preserves_only_explicit_fields(self):
        extracted = extract_job_requirements(
            "Job Title: Data Analyst\nCompany: Example Co\nLocation: Nairobi\n"
            "Responsibilities:\n- Build SQL dashboards\n"
            "Requirements:\n- Bachelor's degree required\n- 3+ years of experience\n"
            "Preferred Qualifications:\n- Python preferred\n"
        )
        self.assertEqual(extracted["title"], "Data Analyst")
        self.assertEqual(extracted["company"], "Example Co")
        self.assertEqual(extracted["location"], "Nairobi")
        self.assertTrue(any("Build SQL dashboards" in item for item in extracted["responsibilities"]))
        self.assertTrue(extracted["education_requirements"])
        self.assertTrue(extracted["experience_requirements"])
        skill = {item["name"]: item for item in extracted["skills"]}["Python"]
        self.assertEqual(skill["requirement_type"], "preferred")

    def test_qualification_evidence_is_report_only_and_keeps_unknown_distinct(self):
        requirements = extract_job_requirements(
            "Requirements:\n- Master's degree required\n- 5+ years of experience\n"
            "- AWS Certified Solutions Architect certification required"
        )
        assessment = assess_qualification_evidence(
            "EDUCATION\nBachelor's degree in Information Systems\n"
            "EXPERIENCE\n3 years of experience in analytics",
            {
                "education": "Bachelor's degree in Information Systems",
                "experience": "3 years of experience in analytics",
            },
            requirements,
        )
        self.assertIn("different degree level", assessment["education"][0]["status"])
        self.assertIn("below the posted threshold", assessment["experience"][0]["status"])
        self.assertIn("not found in resume text", assessment["certifications"][0]["status"])
        self.assertNotIn("score", assessment)
        self.assertIn("not included", assessment["score_inclusion"])

    def test_qualification_evidence_reports_unknown_without_inventing_credentials(self):
        requirements = extract_job_requirements(
            "Requirements:\n- Bachelor's degree\n- 3 years of experience\n"
            "- CPA certification preferred"
        )
        assessment = assess_qualification_evidence("", {}, requirements)
        self.assertIn("cannot determine", assessment["education"][0]["status"])
        self.assertIn("cannot determine", assessment["experience"][0]["status"])
        self.assertIn("not found in resume text", assessment["certifications"][0]["status"])

    def test_uploaded_docx_job_description_extracts_readable_text(self):
        document = Document()
        document.add_paragraph("Job Title: Data Analyst")
        document.add_paragraph("Python is required for this role.")
        buffer = BytesIO()
        document.save(buffer)
        upload = UploadedFile("job.docx", buffer.getvalue())
        validate_resume_upload(upload)
        parsed = extract_text_with_metadata(upload, enable_ocr=False)
        self.assertIn("Python is required", parsed["text"])
        self.assertEqual(parsed["parser"], "docx_parser")

    def test_tailored_suggestions_parse_json_lists(self):
        payload = '[{"issue":"Clarity","recommendation":"Use a direct verb.","section":"experience","original_text":"Helped with reports","example_rewrite":"Prepared reports"}]'
        with patch("local_ai.ollama_generate", return_value=f"```json\n{payload}\n```"):
            suggestions = generate_tailored_resume_suggestions("EXPERIENCE\nHelped with reports", "Analyst role")
        self.assertEqual(suggestions[0]["original_text"], "Helped with reports")

    def test_workflow_history_tracks_application_progress_locally(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkflowStore(str(Path(tmp) / "workflow.db"))
            resume_id = store.save_resume_version("resume.txt", "candidate text")
            analysis_id = store.save_analysis(
                resume_id, "Analyst", "Nairobi", "Technology", "Entry Level", 60, 70, 60
            )
            self.assertEqual(store.list_analysis_history()[0]["application_status"], "Not tracked")
            store.update_application_status(analysis_id, "Interview")
            self.assertEqual(store.list_analysis_history()[0]["application_status"], "Interview")
            with self.assertRaises(ValueError):
                store.update_application_status(analysis_id, "Hired")

    def test_workflow_store_adds_progress_column_to_existing_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "legacy.db")
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("""CREATE TABLE analyzer_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, resume_id INTEGER, target_role TEXT,
                    location TEXT, industry TEXT, experience_level TEXT, skill_score REAL,
                    ats_score REAL, overall_score REAL, details_json TEXT,
                    callback_outcome INTEGER, created_at TEXT)""")
                conn.commit()
            store = WorkflowStore(db_path)
            self.assertEqual(store.list_analysis_history(), [])
            with closing(sqlite3.connect(db_path)) as conn:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(analyzer_analyses)")}
            self.assertIn("application_status", columns)


if __name__ == "__main__":
    unittest.main()
