# accuracy_tester.py - Test the accuracy of resume analysis
from skill_matcher import compare_resume_to_job, debug_skill_extraction
from ats_scorer import calculate_ats_score, debug_ats_scoring

def test_skill_matching_accuracy():
    """Test skill matching with known examples."""
    print("=== TESTING SKILL MATCHING ACCURACY ===\n")
    
    # Test Case 1: Python Developer
    test_resume_1 = """
    John Doe
    Software Developer
    john.doe@email.com | (555) 123-4567
    
    SUMMARY
    Experienced Python developer with 5 years in web development
    
    EXPERIENCE
    Software Engineer at TechCorp (2020-2023)
    - Developed web applications using Python and Django
    - Built REST APIs using Flask and FastAPI
    - Worked with PostgreSQL and MongoDB databases
    - Used Git for version control and deployed on AWS
    - Led a team of 3 developers
    
    SKILLS
    Python, Django, Flask, JavaScript, React, PostgreSQL, MongoDB, Git, AWS, HTML, CSS
    """
    
    test_job_1 = """
    Python Developer Position
    
    Requirements:
    - 3+ years of Python experience
    - Experience with Django or Flask
    - Knowledge of databases (PostgreSQL preferred)
    - Familiarity with AWS cloud services
    - Git version control
    - JavaScript and React experience is a plus
    """
    
    print("TEST 1: Python Developer")
    print("Expected skills: Python, Django, Flask, PostgreSQL, AWS, Git, JavaScript, React")
    
    matched, missing, score = compare_resume_to_job(test_resume_1, test_job_1)
    
    print(f"Matched skills: {matched}")
    print(f"Missing skills: {missing}")
    print(f"Match score: {score:.1f}%")
    print(f"Expected score range: 80-95%")
    print(f"✅ PASS" if 80 <= score <= 95 else f"❌ FAIL")
    print()
    
    # Test Case 2: Data Scientist (should have lower match)
    test_resume_2 = """
    Jane Smith
    Marketing Specialist
    jane@email.com | (555) 987-6543
    
    EXPERIENCE  
    Marketing Coordinator (2021-2023)
    - Managed social media campaigns
    - Used Google Analytics for reporting
    - Created content using Adobe Photoshop
    - Analyzed campaign performance with Excel
    
    SKILLS
    Social Media Marketing, Google Analytics, Adobe Photoshop, Excel, Content Creation
    """
    
    test_job_2 = """
    Senior Data Scientist Position
    
    Requirements:
    - PhD in Computer Science or related field
    - 5+ years of machine learning experience
    - Proficiency in Python, R, and TensorFlow
    - Experience with big data tools (Spark, Hadoop)
    - Strong statistical analysis skills
    - SQL and database experience
    """
    
    print("TEST 2: Marketing Specialist vs Data Scientist Job (should have low match)")
    print("Expected skills: Python, R, TensorFlow, Machine Learning, SQL, Statistics")
    
    matched_2, missing_2, score_2 = compare_resume_to_job(test_resume_2, test_job_2)
    
    print(f"Matched skills: {matched_2}")
    print(f"Missing skills: {missing_2}")
    print(f"Match score: {score_2:.1f}%")
    print(f"Expected score range: 0-20%")
    print(f"✅ PASS" if 0 <= score_2 <= 20 else f"❌ FAIL")
    print()
    
    return (80 <= score <= 95) and (0 <= score_2 <= 20)

def test_ats_scoring_accuracy():
    """Test ATS scoring with known examples."""
    print("=== TESTING ATS SCORING ACCURACY ===\n")
    
    # Good resume example
    good_resume = """
    John Doe
    Software Engineer
    john.doe@email.com | (555) 123-4567 | LinkedIn: linkedin.com/in/johndoe
    San Francisco, CA
    
    SUMMARY
    Experienced software engineer with 5 years developing scalable web applications
    
    EXPERIENCE
    Senior Software Engineer | TechCorp | 2020-2023
    • Led development of customer portal, increasing user engagement by 40%
    • Implemented microservices architecture, reducing system downtime by 60%  
    • Managed team of 5 developers and delivered 12 major releases
    • Optimized database queries, improving application response time by 35%
    
    Software Developer | StartupXYZ | 2018-2020
    • Built REST APIs serving 10,000+ daily users
    • Reduced bug reports by 50% through comprehensive testing
    
    EDUCATION
    Bachelor of Computer Science | University of California | 2018
    
    SKILLS
    Python, JavaScript, React, Node.js, PostgreSQL, AWS, Docker, Git
    """
    
    job_description = """
    Senior Software Engineer Position
    
    We are seeking a Senior Software Engineer with strong Python and JavaScript experience.
    
    Requirements:
    - 5+ years software development experience
    - Proficiency in Python, JavaScript, React
    - Experience with databases (PostgreSQL preferred)
    - AWS cloud experience
    - Strong problem-solving skills
    - Team leadership experience
    """
    
    print("TEST 1: Well-formatted resume with good content")
    
    ats_score, suggestions = calculate_ats_score(good_resume, job_description)
    
    print(f"ATS Score: {ats_score:.1f}%")
    print(f"Number of suggestions: {len(suggestions)}")
    print(f"Expected score range: 75-90%")
    print(f"✅ PASS" if 75 <= ats_score <= 90 else f"❌ FAIL")
    print()
    
    # Poor resume example
    poor_resume = """
    John
    
    I am looking for a job in technology. I have worked at different companies 
    and know some programming. I am responsible for various duties and helped 
    with projects. I have good communication skills and am a team player.
    
    Skills: programming, computers, teamwork
    """
    
    print("TEST 2: Poorly formatted resume with vague content")
    
    ats_score_2, suggestions_2 = calculate_ats_score(poor_resume, job_description)
    
    print(f"ATS Score: {ats_score_2:.1f}%")
    print(f"Number of suggestions: {len(suggestions_2)}")
    print(f"Expected score range: 15-40%")
    print(f"✅ PASS" if 15 <= ats_score_2 <= 40 else f"❌ FAIL")
    print()
    
    return (75 <= ats_score <= 90) and (15 <= ats_score_2 <= 40)

def run_accuracy_tests():
    """Run all accuracy tests."""
    print("🧪 RESUME ANALYZER ACCURACY TESTS")
    print("=" * 50)
    
    skill_test_passed = test_skill_matching_accuracy()
    ats_test_passed = test_ats_scoring_accuracy()
    
    print("OVERALL RESULTS:")
    print(f"Skill Matching: {'✅ PASS' if skill_test_passed else '❌ FAIL'}")
    print(f"ATS Scoring: {'✅ PASS' if ats_test_passed else '❌ FAIL'}")
    
    if skill_test_passed and ats_test_passed:
        print("\n🎉 ALL TESTS PASSED! The system is working accurately.")
    else:
        print("\n⚠️ SOME TESTS FAILED. Check the implementation.")
    
    return skill_test_passed and ats_test_passed

if __name__ == "__main__":
    run_accuracy_tests()