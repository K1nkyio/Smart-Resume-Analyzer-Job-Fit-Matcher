import unittest

from job_fit_scorer import calculate_job_fit
from job_requirements import assess_qualification_evidence, extract_job_requirements
from resume_assessor import assess_resume_quality
from skill_matcher import compare_resume_to_job_detailed


class AssessmentEngineTests(unittest.TestCase):
    def _score(self, resume, job, sections):
        requirements = extract_job_requirements(job)
        skills = compare_resume_to_job_detailed(
            resume, job, resume_sections=sections, target_role="Junior Software Developer"
        )
        qualifications = assess_qualification_evidence(resume, sections, requirements)
        return calculate_job_fit(skills, requirements, qualifications, sections)

    def test_partial_software_candidate_is_not_scored_as_zero(self):
        resume = (
            "EXPERIENCE\nBuilt REST APIs with Node.js and Express.js. Used PostgreSQL and Git.\n"
            "PROJECTS\nDeveloped a React application with JavaScript.\n"
            "EDUCATION\nDiploma in Software Engineering completed.\n"
            "BSc in Computer Science currently pursuing."
        )
        sections = {
            "experience": "Built REST APIs with Node.js and Express.js. Used PostgreSQL and Git.",
            "projects": "Developed a React application with JavaScript.",
            "education": "Diploma in Software Engineering completed.\nBSc in Computer Science currently pursuing.",
        }
        job = (
            "Job Title: Junior Software Developer\nRequirements:\n"
            "- Laravel, React, Next.js, C#, MySQL, MongoDB, API development, Git, testing, and debugging.\n"
            "- Bachelor's degree required."
        )
        result = self._score(resume, job, sections)
        self.assertIsNotNone(result["score"])
        self.assertGreater(result["score"], 0)
        evidence = {row["requirement"]: row for row in result["requirements"]}
        self.assertEqual(evidence["React"]["classification"], "direct match")
        self.assertNotEqual(evidence["MySQL"]["classification"], "direct match")
        requirements = extract_job_requirements(job)
        qualifications = assess_qualification_evidence(resume, sections, requirements)
        self.assertIn("in progress; not completed", qualifications["education"][0]["status"])

    def test_unrelated_degree_status_does_not_change_requested_degree_status(self):
        requirements = extract_job_requirements("Requirements:\n- Bachelor's degree required")
        evidence = assess_qualification_evidence(
            "EDUCATION\nBachelor's degree in Finance, completed.\nMaster's degree in Data Science, currently pursuing.",
            {"education": "Bachelor's degree in Finance, completed.\nMaster's degree in Data Science, currently pursuing."},
            requirements,
        )
        self.assertIn("completed bachelor", evidence["education"][0]["status"])

    def test_missing_evidence_is_distinct_from_zero_or_negative_ability_claim(self):
        result = self._score("EXPERIENCE\nSupported office operations.", "Python required", {"experience": "Supported office operations."})
        python = next(row for row in result["requirements"] if row["requirement"] == "Python")
        self.assertEqual(python["classification"], "not demonstrated")
        self.assertIn("does not establish", python["interpretation"])

    def test_job_requirement_cues_remain_local_to_their_source_lines(self):
        details = compare_resume_to_job_detailed(
            "EXPERIENCE\nBuilt with React.",
            "Laravel is required.\nReact experience is useful.",
            resume_sections={"experience": "Built with React."},
        )
        types = {item["skill"]: item["requirement_type"] for item in details["matched_details"] + details["missing_details"]}
        self.assertEqual(types["Laravel"], "required")
        self.assertEqual(types["React"], "unspecified")

    def test_empty_resume_quality_is_not_assessable(self):
        quality = assess_resume_quality("", {}, {})
        self.assertIsNone(quality["score"])
        self.assertEqual(quality["evidence_coverage"]["available_categories"], 0)

    def test_quality_weights_omit_optional_sections(self):
        quality = assess_resume_quality(
            "EXPERIENCE\nDeveloped a reporting process.\nSKILLS\nExcel",
            {"experience": "Developed a reporting process.", "skills": "Excel"},
            {},
        )
        self.assertNotIn("education_certifications", quality["weights_used"])
        self.assertAlmostEqual(sum(quality["weights_used"].values()), 1.0)

    def test_project_work_counts_as_practical_job_evidence(self):
        result = self._score(
            "PROJECTS\nBuilt a Python data analysis dashboard.",
            "Python required",
            {"projects": "Built a Python data analysis dashboard."},
        )
        row = next(row for row in result["requirements"] if row["requirement"] == "Python")
        self.assertEqual(row["classification"], "direct match")
        self.assertIn("projects", row["resume_sources"])

    def test_presentations_and_job_fit_are_independent(self):
        clear = assess_resume_quality(
            "SUMMARY\nSoftware developer with project experience and practical skills.\n"
            "PROJECTS\nBuilt a web application.\nSKILLS\nReact, JavaScript",
            {"summary": "Software developer with project experience and practical skills.", "projects": "Built a web application.", "skills": "React, JavaScript"},
            {},
        )
        self.assertIsNotNone(clear["score"])
        weak_fit = self._score("PROJECTS\nBuilt a web application.", "C# required", {"projects": "Built a web application."})
        self.assertLess(weak_fit["score"], 50)


if __name__ == "__main__":
    unittest.main()
