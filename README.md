# Smart Resume Analyzer + Job Fit Matcher

A local Streamlit app for resume parsing, job fit analysis, ATS-oriented checks, career recommendations and optional market lookups.

## Run locally

1. Create and activate a Python virtual environment.
2. Install the core dependencies: `python -m pip install -r requirements.txt`.
3. Start the app: `streamlit run app.py`.

OCR is optional: install `requirements-ocr.txt`, the Tesseract executable, and Poppler for image-based PDFs. Optional spaCy support is listed in `requirements-nlp.txt`; it also needs a compatible model installed separately. Local AI suggestions use Ollama only when enabled in the sidebar; install and run Ollama separately and configure `OLLAMA_BASE_URL` and `LOCAL_AI_OLLAMA_MODEL` if needed. The default text similarity is local TF-IDF and does not download an embedding model. Embedding support is optional in `requirements-embeddings.txt`; analysis only loads a locally cached or local-path model and does not download models automatically.

## Privacy and storage

- No-retention mode is enabled by default. It prevents analysis results from being written to the local SQLite workflow tables.
- Streamlit keeps the current analysis in session memory so the result screens can work. The session should be closed to clear that in-memory result.
- Turn off no-retention only when you want local workflow history saved in `users.db`. The sidebar action clears analyzer history; it does not delete the database file or unrelated tables.
- Ollama is off by default. Enabling it sends resume and job text to the configured Ollama service, which may be on the same machine or another host.
- Enabling external market lookup sends role and location terms to Adzuna. Resume text is not part of that request.
- Resume uploads are limited to 10 MB and are processed in memory. Do not expose a local deployment publicly without adding authentication and deployment-specific access controls.

## Market data

Adzuna listings are shown only when a live API request returns results and credentials are configured with `ADZUNA_APP_ID` and `ADZUNA_APP_KEY`. The app no longer substitutes generated benchmark roles for vacancies. Configured salary and skill trend references are illustrative, may be stale, and are not live market estimates.

## Current analysis workspaces

- **Analyze Resume + Job** evaluates one resume against one job description.
- **Compare Two Resumes** compares two resumes against the same job description.
- **Compare Job Opportunities** evaluates one resume against three to five supplied job descriptions and compares detected skill coverage, resume quality and ATS text heuristics.
- Single-job workflows accept pasted job descriptions or uploaded PDF, DOCX, and TXT files. Extracted fields and requirement lines are heuristic and should be checked against the source.
- The **Optimize Resume** workspace lets you edit extracted sections, accept or reject anchored local AI suggestions, and export the edited content to DOCX or PDF. AI rewrites require human review; only suggestions anchored to an exact extracted quote can be applied directly.
- The **Skills Intelligence** tab groups detected resume skills by catalog category and shows the section where each appeared, related technologies, and job-specific gaps. It does not assign proficiency levels.
- When No Data Retained is turned off, local history can track each opportunity as saved, applied, interview, offer, or rejected. Clear workflow history from the sidebar when it is no longer needed.

## Assessment methodology and limits

The four displayed assessments are separate signals:

- **Job fit** is a configurable weighted mean of assessable categories: required skills (30%), relevant experience (25%), responsibilities (15%), education and certifications (10%), preferred skills (10%), and project evidence (10%). Categories without extracted evidence are omitted and remaining weights are renormalized. Exact skills in work, project, or education context earn more evidence credit than a skills-list mention. Related technologies receive limited credit and remain labeled as related; they do not prove direct proficiency. Requirement classifications and the category scores are shown for review. This is a heuristic and not a hiring probability.
- **Resume quality** uses a weighted mean of applicable document-presentation dimensions: structure (15%), profile (10%), experience presentation (25%), skills presentation (15%), education and certifications (10%), projects and achievements (15%), and readability (10%). Missing optional sections are omitted and weights renormalized. Contact completeness is reported separately and is not scored.
- **ATS compatibility** uses extracted-text heuristics for extraction quality (20%), structure (20%), section identification (15%), readability (15%), job keyword relevance (20%), and contact/document signals (10%). Unavailable keyword relevance is omitted and the remaining weights renormalized. It does not assess visual layout or predict acceptance by a specific ATS.
- **TF-IDF lexical similarity** measures vocabulary overlap only. It is not semantic understanding, requirement coverage, or competence. A low value is not evidence that a candidate is unqualified.

Requirement and resume evidence extraction is deterministic and catalog-based. It can miss unfamiliar or nonstandard terminology, and all extracted sections, requirements, and evidence classifications should be reviewed against the source documents. "Not demonstrated" describes the supplied resume text; it does not mean the candidate lacks the ability. Education and experience checks are self-reported and credential checks are not independently verified. The scoring model is a heuristic, not scientifically calibrated; no benchmark accuracy is claimed.

The skill matcher must receive the original text with line breaks. Flattening text before requirement classification can lose headings and make a cue such as "required" appear to apply to every skill in the posting. The app keeps raw text for requirement and evidence extraction; cleaned text is retained for similarity and display tasks.

Education, certification, and numeric experience-duration requirements are compared conservatively against extracted resume text and shown as separate evidence checks. Missing sections and ambiguous requirements are reported as unknown; stated years are self-reported, and credentials are not independently verified. These checks do not affect the numeric job-fit score. The current requirement extractor does not determine transferable skills beyond the curated technology relationships above.

The Career Coach is available only when Ollama was explicitly enabled for the analysis. It receives a compact summary of detected scores, matches, gaps and recommendations rather than the raw resume text; local model output can still be wrong and should be checked against the evidence.

## Validation

Run the focused unit checks with `python -m unittest discover -s tests`. `accuracy_tester.py` is a legacy manual scoring smoke check and prints sample candidate data; it is not a calibrated benchmark. Scoring is heuristic guidance, not a hiring probability or scientifically validated hiring prediction. Browser visual review, authentication, and deployment hardening are not currently provided.
