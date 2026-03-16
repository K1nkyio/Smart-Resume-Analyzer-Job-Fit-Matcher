# config.py

# Comprehensive skills database with weights and categories
SKILLS = {
    "technical": {
        # Programming Languages
        "Python": 4, "Java": 4, "C++": 4, "C#": 4, "JavaScript": 4, "TypeScript": 3,
        "Go": 3, "Rust": 3, "Swift": 3, "Kotlin": 3, "Ruby": 3, "PHP": 3,
        "Scala": 3, "R": 3, "MATLAB": 3, "Julia": 2, "Dart": 2, "Perl": 2,
        
        # Web Development
        "React": 4, "Vue.js": 3, "Angular": 3, "Node.js": 4, "Express.js": 3,
        "Next.js": 3, "Nuxt.js": 2, "Svelte": 2, "HTML5": 3, "CSS3": 3,
        "SASS": 2, "LESS": 2, "Bootstrap": 2, "Tailwind CSS": 2, "Material UI": 2,
        "jQuery": 2, "Redux": 3, "Vuex": 2, "MobX": 2,
        
        # Backend & APIs
        "Django": 4, "Flask": 3, "FastAPI": 3, "Spring Boot": 4, "ASP.NET": 3,
        "Laravel": 3, "Ruby on Rails": 3, "Express": 3, "GraphQL": 3, "REST API": 4,
        "WebSocket": 2, "gRPC": 2, "Microservices": 4, "API Gateway": 3,
        
        # Databases
        "SQL": 4, "MySQL": 3, "PostgreSQL": 4, "MongoDB": 3, "Redis": 3,
        "Cassandra": 2, "Neo4j": 2, "InfluxDB": 2, "ElasticSearch": 3,
        "DynamoDB": 2, "Firebase": 3, "SQLite": 2, "Oracle": 3, "SQL Server": 3,
        
        # Cloud & DevOps
        "AWS": 4, "Azure": 4, "Google Cloud": 4, "Docker": 4, "Kubernetes": 4,
        "Jenkins": 3, "GitLab CI": 3, "GitHub Actions": 3, "Terraform": 3,
        "Ansible": 3, "Chef": 2, "Puppet": 2, "Vagrant": 2, "Helm": 2,
        
        # Data Science & AI/ML
        "TensorFlow": 4, "PyTorch": 4, "Keras": 3, "Scikit-learn": 4,
        "Pandas": 4, "NumPy": 4, "Matplotlib": 3, "Seaborn": 3, "Plotly": 3,
        "Apache Spark": 3, "Hadoop": 3, "Kafka": 3, "Airflow": 3,
        "MLflow": 2, "Kubeflow": 2, "OpenCV": 3, "NLTK": 3, "spaCy": 3,
        
        # Mobile Development
        "React Native": 3, "Flutter": 3, "iOS Development": 3, "Android Development": 3,
        "Xamarin": 2, "Ionic": 2, "Cordova": 2,
        
        # Game Development
        "Unity": 3, "Unreal Engine": 3, "Godot": 2, "Cocos2d": 2,
        
        # Blockchain
        "Solidity": 3, "Web3": 2, "Ethereum": 3, "Smart Contracts": 3,
        
        # Security
        "Cybersecurity": 4, "Penetration Testing": 3, "OWASP": 3, "SSL/TLS": 3,
        "OAuth": 3, "JWT": 3, "Encryption": 3, "Firewall": 3,
    },
    
    "tools": {
        # Version Control
        "Git": 4, "GitHub": 4, "GitLab": 3, "Bitbucket": 3, "SVN": 2,
        
        # IDEs & Editors
        "VS Code": 3, "IntelliJ IDEA": 3, "PyCharm": 3, "Eclipse": 2,
        "Visual Studio": 3, "Atom": 2, "Sublime Text": 2, "Vim": 2,
        
        # Project Management
        "JIRA": 3, "Trello": 2, "Asana": 2, "Monday.com": 2, "Notion": 2,
        "Slack": 2, "Microsoft Teams": 2, "Discord": 1,
        
        # Analytics & BI
        "Tableau": 4, "Power BI": 4, "Google Analytics": 3, "Looker": 3,
        "QlikView": 2, "D3.js": 3, "Grafana": 3, "Kibana": 2,
        
        # Design
        "Figma": 3, "Adobe Photoshop": 3, "Adobe Illustrator": 3,
        "Sketch": 3, "InVision": 2, "Adobe XD": 3, "Canva": 2,
        
        # Testing
        "Selenium": 3, "Jest": 3, "Mocha": 2, "Cypress": 3, "Postman": 3,
        "JUnit": 3, "pytest": 3, "TestNG": 2, "Cucumber": 2,
        
        # Operating Systems
        "Linux": 4, "Windows": 3, "macOS": 3, "Ubuntu": 3, "CentOS": 2,
        
        # Others
        "Postman": 3, "Insomnia": 2, "Wireshark": 2, "VMware": 2,
        "VirtualBox": 2, "SSH": 3, "FTP": 2, "SFTP": 2,
    },
    
    "soft": {
        # Communication
        "Communication": 4, "Public Speaking": 3, "Presentation": 3,
        "Technical Writing": 4, "Documentation": 3, "Storytelling": 2,
        
        # Leadership
        "Leadership": 4, "Team Management": 4, "Mentoring": 3,
        "Delegation": 3, "Conflict Resolution": 3, "Coaching": 3,
        
        # Collaboration
        "Teamwork": 4, "Collaboration": 4, "Cross-functional": 3,
        "Stakeholder Management": 3, "Client Relations": 3,
        
        # Problem Solving
        "Problem Solving": 4, "Critical Thinking": 4, "Analytical": 4,
        "Decision Making": 3, "Creative Thinking": 3, "Innovation": 3,
        
        # Personal Traits
        "Adaptability": 4, "Flexibility": 3, "Resilience": 3,
        "Self-motivated": 3, "Proactive": 3, "Initiative": 3,
        "Attention to Detail": 4, "Organizational": 3, "Multitasking": 3,
        
        # Time Management
        "Time Management": 4, "Prioritization": 3, "Planning": 3,
        "Goal Setting": 3, "Deadline Management": 3,
        
        # Learning
        "Continuous Learning": 4, "Curiosity": 3, "Research": 3,
        "Self-directed Learning": 3, "Knowledge Sharing": 3,
    },
    
    "industry_specific": {
        # Healthcare
        "HIPAA": 3, "HL7": 3, "FHIR": 3, "Medical Devices": 3,
        "Clinical Research": 3, "FDA Regulations": 3, "EMR/EHR": 3,
        
        # Finance
        "Financial Modeling": 4, "Risk Management": 4, "Compliance": 4,
        "Banking": 3, "Trading": 3, "Portfolio Management": 3,
        "Basel III": 2, "SOX": 3, "PCI DSS": 3, "KYC": 3, "AML": 3,
        
        # Education
        "Curriculum Development": 3, "Learning Management Systems": 3,
        "Educational Technology": 3, "Assessment": 3, "Pedagogy": 3,
        
        # Marketing
        "SEO": 3, "SEM": 3, "Social Media Marketing": 3, "Content Marketing": 3,
        "Email Marketing": 3, "PPC": 3, "Google Ads": 3, "Facebook Ads": 3,
        "Marketing Automation": 3, "CRM": 3, "Salesforce": 3, "HubSpot": 3,
        
        # Sales
        "Sales": 4, "CRM": 3, "Lead Generation": 3, "Customer Acquisition": 3,
        "Account Management": 3, "Negotiation": 4, "Pipeline Management": 3,
        
        # HR
        "Recruitment": 3, "Talent Acquisition": 3, "Performance Management": 3,
        "Employee Relations": 3, "HR Information Systems": 3, "Payroll": 3,
        "Benefits Administration": 3, "Training & Development": 3,
        
        # Manufacturing
        "Lean Manufacturing": 3, "Six Sigma": 3, "Quality Control": 3,
        "Supply Chain": 3, "Inventory Management": 3, "ERP": 3, "SAP": 3,
        
        # Legal
        "Contract Law": 3, "Intellectual Property": 3, "Corporate Law": 3,
        "Litigation": 3, "Legal Research": 3, "Compliance": 4,
    }
}

# Industry-specific skill mappings
INDUSTRY_SKILLS = {
    "Technology": {
        "must_have": ["Python", "Java", "JavaScript", "SQL", "Git", "Problem Solving"],
        "good_to_have": ["AWS", "Docker", "React", "Node.js", "MongoDB", "Machine Learning"],
        "emerging": ["Kubernetes", "GraphQL", "TypeScript", "Serverless", "AI/ML", "Blockchain"]
    },
    
    "Healthcare": {
        "must_have": ["HIPAA", "Medical Terminology", "Patient Care", "Documentation"],
        "good_to_have": ["EMR/EHR", "HL7", "Clinical Research", "Quality Assurance"],
        "emerging": ["Telemedicine", "Health Informatics", "AI in Healthcare", "Genomics"]
    },
    
    "Finance": {
        "must_have": ["Financial Analysis", "Risk Management", "Excel", "Compliance", "SQL"],
        "good_to_have": ["Python", "R", "Bloomberg Terminal", "Financial Modeling", "Trading"],
        "emerging": ["Fintech", "Cryptocurrency", "Robo-advisors", "RegTech", "AI in Finance"]
    },
    
    "Marketing": {
        "must_have": ["Digital Marketing", "SEO", "Content Marketing", "Analytics", "CRM"],
        "good_to_have": ["Google Ads", "Social Media", "Email Marketing", "Adobe Creative Suite"],
        "emerging": ["Marketing Automation", "Influencer Marketing", "Voice Search", "AI Marketing"]
    },
    
    "Education": {
        "must_have": ["Curriculum Development", "Assessment", "Classroom Management", "Communication"],
        "good_to_have": ["Educational Technology", "LMS", "Online Teaching", "Data Analysis"],
        "emerging": ["VR/AR in Education", "Personalized Learning", "AI Tutoring", "Microlearning"]
    },
    
    "Engineering": {
        "must_have": ["Problem Solving", "Technical Drawing", "Project Management", "Safety"],
        "good_to_have": ["CAD", "MATLAB", "Simulation", "Quality Control", "Six Sigma"],
        "emerging": ["IoT", "3D Printing", "Robotics", "Sustainable Design", "Digital Twins"]
    }
}

# Experience level skill expectations
EXPERIENCE_LEVELS = {
    "Entry Level": {
        "years": "0-2",
        "key_skills": ["Basic Programming", "Communication", "Learning Agility", "Teamwork"],
        "skill_depth": "basic",
        "expected_skills_count": 5
    },
    
    "Mid Level": {
        "years": "3-7",
        "key_skills": ["Advanced Technical", "Project Management", "Mentoring", "Problem Solving"],
        "skill_depth": "intermediate",
        "expected_skills_count": 10
    },
    
    "Senior Level": {
        "years": "8-15",
        "key_skills": ["Architecture", "Leadership", "Strategy", "System Design", "Team Management"],
        "skill_depth": "advanced",
        "expected_skills_count": 15
    },
    
    "Executive": {
        "years": "15+",
        "key_skills": ["Strategic Planning", "Business Development", "P&L Management", "Vision"],
        "skill_depth": "expert",
        "expected_skills_count": 20
    }
}

# ATS Keywords and phrases
ATS_KEYWORDS = {
    "action_verbs": [
        "achieved", "administered", "analyzed", "architected", "built", "collaborated",
        "created", "designed", "developed", "enhanced", "established", "executed",
        "facilitated", "generated", "implemented", "improved", "increased", "initiated",
        "launched", "led", "managed", "optimized", "organized", "produced", "reduced",
        "streamlined", "supervised", "transformed", "upgraded"
    ],
    
    "quantifiable_terms": [
        "increased by", "decreased by", "improved by", "reduced by", "managed",
        "led team of", "budget of", "revenue of", "saved", "generated",
        "within", "under budget", "ahead of schedule", "exceeded targets"
    ],
    
    "technical_certifications": [
        "AWS Certified", "Microsoft Certified", "Google Certified", "Cisco Certified",
        "PMP", "Agile", "Scrum Master", "Six Sigma", "CISSP", "CompTIA",
        "Oracle Certified", "Salesforce Certified", "VMware Certified"
    ],
    
    "education_keywords": [
        "Bachelor", "Master", "PhD", "Degree", "University", "College",
        "Certification", "Training", "Course", "Diploma", "GPA"
    ],
    
    "common_sections": [
        "summary", "objective", "experience", "education", "skills",
        "projects", "achievements", "certifications", "awards", "references"
    ]
}

# Salary data by role and experience (sample data - in real app, connect to salary APIs)
SALARY_DATA = {
    "Data Scientist": {
        "Entry Level": {"min": 70000, "max": 95000, "average": 82500, "growth": 15},
        "Mid Level": {"min": 95000, "max": 140000, "average": 117500, "growth": 12},
        "Senior Level": {"min": 140000, "max": 200000, "average": 170000, "growth": 8},
        "Executive": {"min": 200000, "max": 350000, "average": 275000, "growth": 5}
    },
    
    "Software Engineer": {
        "Entry Level": {"min": 65000, "max": 90000, "average": 77500, "growth": 18},
        "Mid Level": {"min": 90000, "max": 130000, "average": 110000, "growth": 14},
        "Senior Level": {"min": 130000, "max": 180000, "average": 155000, "growth": 10},
        "Executive": {"min": 180000, "max": 300000, "average": 240000, "growth": 6}
    },
    
    "Product Manager": {
        "Entry Level": {"min": 75000, "max": 100000, "average": 87500, "growth": 12},
        "Mid Level": {"min": 100000, "max": 150000, "average": 125000, "growth": 10},
        "Senior Level": {"min": 150000, "max": 220000, "average": 185000, "growth": 7},
        "Executive": {"min": 220000, "max": 400000, "average": 310000, "growth": 4}
    },
    
    "Marketing Manager": {
        "Entry Level": {"min": 50000, "max": 70000, "average": 60000, "growth": 8},
        "Mid Level": {"min": 70000, "max": 100000, "average": 85000, "growth": 6},
        "Senior Level": {"min": 100000, "max": 150000, "average": 125000, "growth": 4},
        "Executive": {"min": 150000, "max": 250000, "average": 200000, "growth": 2}
    }
}

# Job market trends (sample data)
JOB_TRENDS = {
    "Data Scientist": [
        {"month": "Jan 2024", "demand": 85},
        {"month": "Feb 2024", "demand": 87},
        {"month": "Mar 2024", "demand": 90},
        {"month": "Apr 2024", "demand": 92},
        {"month": "May 2024", "demand": 95},
        {"month": "Jun 2024", "demand": 98}
    ],
    
    "Software Engineer": [
        {"month": "Jan 2024", "demand": 90},
        {"month": "Feb 2024", "demand": 92},
        {"month": "Mar 2024", "demand": 89},
        {"month": "Apr 2024", "demand": 94},
        {"month": "May 2024", "demand": 96},
        {"month": "Jun 2024", "demand": 98}
    ]
}

# Learning resources and recommendations
LEARNING_RESOURCES = {
    "technical": {
        "Python": [
            "Complete Python Bootcamp on Udemy",
            "Python.org official tutorial",
            "Automate the Boring Stuff with Python",
            "Python for Data Science - Coursera"
        ],
        "JavaScript": [
            "freeCodeCamp JavaScript course",
            "JavaScript.info tutorial",
            "You Don't Know JS book series",
            "Modern JavaScript Course - Udemy"
        ],
        "React": [
            "Official React documentation",
            "React - The Complete Guide (Udemy)",
            "freeCodeCamp React course",
            "React Native for mobile development"
        ],
        "AWS": [
            "AWS Certified Solutions Architect course",
            "AWS Free Tier hands-on labs",
            "A Cloud Guru AWS courses",
            "AWS official training and certification"
        ],
        "Machine Learning": [
            "Andrew Ng's Machine Learning Course (Coursera)",
            "Fast.ai practical deep learning course",
            "Kaggle Learn micro-courses",
            "Machine Learning Yearning by Andrew Ng"
        ]
    },
    
    "soft": {
        "Leadership": [
            "Dale Carnegie Leadership Training",
            "Harvard Business Review Leadership articles",
            "The 7 Habits of Highly Effective People",
            "Leadership in Technology - LinkedIn Learning"
        ],
        "Communication": [
            "Toastmasters International",
            "Coursera Business Communication courses",
            "TED Talks on effective communication",
            "Public speaking workshops"
        ],
        "Project Management": [
            "PMP Certification training",
            "Agile and Scrum certification",
            "Project Management Professional courses",
            "Trello/Asana project management tools"
        ]
    }
}

# Resume section templates and best practices
RESUME_TEMPLATES = {
    "summary": {
        "template": "[X] years of experience in [FIELD] with expertise in [KEY_SKILLS]. Proven track record of [ACHIEVEMENT] and [IMPACT]. Seeking to leverage [STRENGTH] to contribute to [COMPANY/ROLE].",
        "tips": [
            "Keep it 2-3 sentences maximum",
            "Include years of experience",
            "Mention key skills relevant to the job",
            "Quantify achievements where possible",
            "Tailor to the specific role and company"
        ]
    },
    
    "experience": {
        "template": "[ACTION_VERB] [WHAT] [HOW/TOOLS] resulting in [QUANTIFIABLE_RESULT]",
        "tips": [
            "Start each bullet with a strong action verb",
            "Include quantifiable results (numbers, percentages)",
            "Use past tense for previous roles, present for current",
            "Focus on achievements, not just responsibilities",
            "Keep bullets concise but impactful"
        ]
    },
    
    "skills": {
        "categories": ["Technical Skills", "Tools & Technologies", "Soft Skills", "Languages"],
        "tips": [
            "Organize skills by category",
            "List most relevant skills first",
            "Include proficiency levels if helpful",
            "Don't oversell your abilities",
            "Update regularly as you learn new skills"
        ]
    }
}

# Common resume mistakes and fixes
COMMON_MISTAKES = {
    "formatting": [
        "Inconsistent font sizes and styles",
        "Poor spacing and alignment",
        "Too much or too little white space",
        "Unprofessional email address",
        "Missing contact information"
    ],
    
    "content": [
        "Generic objective statements",
        "Listing duties instead of achievements",
        "Including irrelevant information",
        "Spelling and grammatical errors",
        "Not tailoring to the job description"
    ],
    
    "ats_issues": [
        "Using images or graphics for text",
        "Complex formatting and tables",
        "Non-standard section headers",
        "Missing keywords from job description",
        "Inconsistent job titles and company names"
    ]
}

# Industry-specific resume tips
INDUSTRY_TIPS = {
    "Technology": [
        "Include links to GitHub, portfolio, or live projects",
        "Mention specific technologies and frameworks",
        "Quantify your impact (performance improvements, user growth)",
        "Include open source contributions",
        "Highlight problem-solving and innovation"
    ],
    
    "Healthcare": [
        "Emphasize patient care and safety",
        "Include relevant certifications and licenses",
        "Mention compliance with regulations (HIPAA, etc.)",
        "Highlight teamwork and collaboration",
        "Include continuing education and training"
    ],
    
    "Finance": [
        "Quantify financial impact and results",
        "Include relevant certifications (CFA, CPA, etc.)",
        "Mention compliance and risk management",
        "Highlight analytical and problem-solving skills",
        "Include software proficiency (Excel, Bloomberg, etc.)"
    ],
    
    "Marketing": [
        "Include metrics and ROI from campaigns",
        "Mention specific platforms and tools used",
        "Highlight creativity and innovation",
        "Include portfolio or campaign examples",
        "Show understanding of target audiences"
    ]
}

# Configuration for different analysis modes
ANALYSIS_MODES = {
    "quick": {
        "description": "Fast skill matching and basic ATS check",
        "features": ["skill_matching", "basic_ats"],
        "time_estimate": "30 seconds"
    },
    
    "standard": {
        "description": "Comprehensive analysis with recommendations",
        "features": ["skill_matching", "ats_analysis", "recommendations", "structure_check"],
        "time_estimate": "2 minutes"
    },
    
    "premium": {
        "description": "Full analysis with career insights and market data",
        "features": ["skill_matching", "ats_analysis", "recommendations", "structure_check", 
                   "career_insights", "salary_data", "learning_path", "cover_letter_tips"],
        "time_estimate": "5 minutes"
    }
}

# Email templates for follow-ups
EMAIL_TEMPLATES = {
    "follow_up": {
        "subject": "Following up on my application for [POSITION] at [COMPANY]",
        "body": "Dear [HIRING_MANAGER],\n\nI hope this email finds you well. I wanted to follow up on my application for the [POSITION] role at [COMPANY] that I submitted on [DATE].\n\nI'm very excited about the opportunity to contribute to [COMPANY] with my [RELEVANT_SKILLS] experience. I believe my background in [SPECIFIC_AREA] would be valuable for [SPECIFIC_PROJECT/GOAL].\n\nI would welcome the opportunity to discuss how I can contribute to your team. Please let me know if you need any additional information.\n\nBest regards,\n[YOUR_NAME]"
    },
    
    "thank_you": {
        "subject": "Thank you for the interview - [POSITION] at [COMPANY]",
        "body": "Dear [INTERVIEWER_NAME],\n\nThank you for taking the time to meet with me yesterday to discuss the [POSITION] role at [COMPANY]. I enjoyed our conversation about [SPECIFIC_TOPIC_DISCUSSED].\n\nOur discussion reinforced my enthusiasm for this opportunity and confirmed that my experience in [RELEVANT_AREA] would be valuable for [SPECIFIC_CHALLENGE/PROJECT].\n\nI'm excited about the possibility of joining your team and contributing to [COMPANY_GOAL]. Please don't hesitate to reach out if you need any additional information.\n\nBest regards,\n[YOUR_NAME]"
    }
}

# Role-specific optimization playbooks for practical guidance
ROLE_PLAYBOOKS = {
    "Data Analyst": {
        "must_have_skills": ["SQL", "Python", "Tableau", "Excel", "Statistics"],
        "nice_to_have_skills": ["Power BI", "A/B Testing", "Data Modeling", "ETL"],
        "keywords": ["dashboard", "insights", "stakeholder", "KPI", "analysis"],
        "certifications": ["Google Data Analytics Certificate", "Microsoft Power BI Data Analyst"],
        "portfolio_projects": [
            "Build an end-to-end KPI dashboard for a public dataset",
            "Run an A/B test analysis and present business recommendations",
            "Publish a SQL + Python data quality pipeline project"
        ],
        "skill_impact": {
            "SQL": "SQL is a baseline requirement for most analyst roles.",
            "Python": "Python increases automation and advanced analytics capability.",
            "Tableau": "Tableau demonstrates business storytelling and visualization ability."
        }
    },
    "Backend Engineer": {
        "must_have_skills": ["Python", "Java", "Node.js", "SQL", "REST API", "Git"],
        "nice_to_have_skills": ["Docker", "Kubernetes", "AWS", "Microservices", "Redis"],
        "keywords": ["scalable", "API", "latency", "reliability", "distributed systems"],
        "certifications": ["AWS Certified Developer - Associate", "CKAD"],
        "portfolio_projects": [
            "Build a production-style REST API with auth, caching, and tests",
            "Deploy a containerized service with CI/CD and health checks",
            "Design a high-throughput event processing service"
        ],
        "skill_impact": {
            "REST API": "API design is central to backend delivery and integration.",
            "Docker": "Containerization is expected for modern deployment workflows.",
            "AWS": "Cloud execution skills reduce onboarding time in most teams."
        }
    },
    "Frontend Engineer": {
        "must_have_skills": ["JavaScript", "React", "HTML5", "CSS3", "Git"],
        "nice_to_have_skills": ["TypeScript", "Next.js", "Testing", "Accessibility"],
        "keywords": ["responsive", "accessibility", "performance", "component design"],
        "certifications": ["Meta Front-End Developer", "W3C Front-End Web Developer"],
        "portfolio_projects": [
            "Build an accessible multi-page React app with performance budget",
            "Create a design-system component library with documentation",
            "Implement a production-ready frontend testing workflow"
        ],
        "skill_impact": {
            "React": "React proficiency is frequently used as a screening filter.",
            "TypeScript": "Type safety improves maintainability in large codebases.",
            "Accessibility": "Accessibility capability is increasingly required in enterprise environments."
        }
    },
    "Data Scientist": {
        "must_have_skills": ["Python", "Machine Learning", "Statistics", "SQL", "Pandas"],
        "nice_to_have_skills": ["PyTorch", "TensorFlow", "MLflow", "AWS", "Airflow"],
        "keywords": ["modeling", "experiment", "feature engineering", "MLOps"],
        "certifications": ["TensorFlow Developer Certificate", "AWS Machine Learning Specialty"],
        "portfolio_projects": [
            "Train and deploy an end-to-end classification model with monitoring",
            "Build a feature engineering and model retraining pipeline",
            "Publish experiment tracking and model comparison notebook"
        ],
        "skill_impact": {
            "Machine Learning": "ML depth determines readiness for model-centric responsibilities.",
            "Statistics": "Statistical rigor improves experiment quality and model validation.",
            "SQL": "Data extraction and validation are mandatory for model development."
        }
    },
    "Product Manager": {
        "must_have_skills": ["Leadership", "Communication", "Analytics", "Agile", "SQL"],
        "nice_to_have_skills": ["Product Strategy", "A/B Testing", "Roadmapping", "Stakeholder Management"],
        "keywords": ["roadmap", "user outcomes", "prioritization", "cross-functional"],
        "certifications": ["PSPO", "Pragmatic Institute Product Management"],
        "portfolio_projects": [
            "Create a product requirements document with measurable KPIs",
            "Run a prioritization framework analysis for a feature backlog",
            "Develop a product experiment plan and success metrics"
        ],
        "skill_impact": {
            "Product Strategy": "Strategy evidence signals ownership beyond delivery.",
            "Analytics": "Data-driven decision making is critical for product prioritization.",
            "Stakeholder Management": "Effective cross-functional alignment drives execution success."
        }
    }
}

# Cross-role skill synonyms for more robust extraction
SKILL_SYNONYMS = {
    "JavaScript": ["js", "java script", "ecmascript"],
    "TypeScript": ["ts"],
    "Node.js": ["node", "nodejs", "node js"],
    "React": ["reactjs", "react.js"],
    "Vue.js": ["vue", "vuejs", "vue js"],
    "Machine Learning": ["ml", "machine learning model", "predictive modeling"],
    "Artificial Intelligence": ["ai"],
    "Deep Learning": ["dl", "neural networks"],
    "Scikit-learn": ["sklearn", "scikit learn"],
    "PostgreSQL": ["postgres", "postgresql"],
    "SQL Server": ["mssql", "ms sql server"],
    "Google Cloud": ["gcp", "google cloud platform"],
    "REST API": ["restful api", "rest apis", "restful services"],
    "GitHub Actions": ["github action", "gha"],
    "Kubernetes": ["k8s"],
    "Power BI": ["powerbi"],
    "A/B Testing": ["ab testing", "a b testing", "split testing"],
    "TensorFlow": ["tensorflow", "tf"],
    "PyTorch": ["pytorch"]
}

PRIVACY_POLICY_TEXT = """
Privacy Policy (Local Application)

1. Resume and job description data is processed only for analysis features in this application.
2. If "No Data Retained" mode is enabled, resume content and analysis outputs are not stored in the local database.
3. Uploaded files are processed in-memory and are not persisted as files by default.
4. If external market APIs are enabled, only role/location queries are sent; resume body content is not transmitted.
5. Users can clear locally stored history at any time from the workflow controls.
"""
