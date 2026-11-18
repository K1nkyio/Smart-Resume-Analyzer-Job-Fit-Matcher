import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from ats_scorer import calculate_ats_score_detailed
from config import PRIVACY_POLICY_TEXT, ROLE_PLAYBOOKS
from job_matcher import find_similar_jobs, generate_market_report, get_salary_insights
from job_fit_scorer import calculate_job_fit
from job_requirements import assess_qualification_evidence, extract_job_requirements
from local_ai import (
    LocalAISettings,
    extract_structured_profile,
    generate_career_coach_response,
    generate_cover_letter,
    generate_gap_analysis,
    generate_tailored_resume_suggestions,
    semantic_match_score,
)
from resume_parser import (
    clean_text,
    extract_contact_info,
    extract_sections,
    extract_text_with_metadata,
    validate_resume_upload,
)
from resume_assessor import assess_resume_quality
from skill_matcher import (
    compare_resume_to_job_detailed,
    generate_job_fit_plan,
    get_skill_recommendations,
)
from validation import evaluate_from_dataframe
from workflow_store import WorkflowStore, build_ats_safe_export, build_resume_docx, build_resume_pdf


st.set_page_config(
    page_title="Smart Resume Analyzer + Job Fit Matcher",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_store() -> WorkflowStore:
    if "workflow_store" not in st.session_state:
        st.session_state.workflow_store = WorkflowStore("users.db")
    return st.session_state.workflow_store


def run_single_analysis(
    uploaded_file,
    job_description: str,
    target_role: str,
    industry: str,
    experience_level: str,
    location: str,
    enable_ocr: bool,
    local_only_mode: bool,
    enable_local_ai: bool,
    ollama_model: str,
) -> Dict[str, Any]:
    validate_resume_upload(uploaded_file)
    parse_meta = extract_text_with_metadata(uploaded_file, enable_ocr=enable_ocr)
    resume_text = parse_meta.get("text", "")
    if not resume_text.strip():
        raise ValueError("No readable resume text was extracted. Check the file and, for scanned documents, enable OCR and install its system tools.")
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_description)
    job_requirements = extract_job_requirements(job_description)
    contact_info = extract_contact_info(resume_text)
    sections = extract_sections(resume_text)
    qualification_evidence = assess_qualification_evidence(
        resume_text, sections, job_requirements
    )
    resume_quality = assess_resume_quality(resume_text, sections, contact_info)

    ai_settings = LocalAISettings(ollama_model=ollama_model)
    # Keep similarity deterministic and offline by default. Embedding models can
    # be enabled explicitly after adding a model-consent control.
    semantic_score = semantic_match_score(resume_clean, job_clean, use_embeddings=False)
    structured_profile = (
        extract_structured_profile(resume_text=resume_text, job_text=job_description, settings=ai_settings)
        if enable_local_ai
        else {}
    )
    gap_analysis = generate_gap_analysis(structured_profile) if enable_local_ai else {}
    tailored_suggestions = (
        generate_tailored_resume_suggestions(resume_text=resume_text, job_text=job_description, settings=ai_settings)
        if enable_local_ai
        else []
    )
    cover_letter = (
        generate_cover_letter(
            resume_text=resume_text,
            job_text=job_description,
            company="",
            settings=ai_settings,
        )
        if enable_local_ai
        else ""
    )

    skill_details = compare_resume_to_job_detailed(
        # Keep source line breaks and headings for section-aware evidence and
        # requirement classification; clean_text intentionally flattens them.
        resume_text=resume_text,
        job_text=job_description,
        industry=industry,
        experience_level=experience_level,
        target_role=target_role,
        location=location,
        resume_sections=sections,
    )
    ats_details = calculate_ats_score_detailed(
        resume_text=resume_text,
        job_description=job_description,
        target_role=target_role,
        location=location,
        parser_metadata=parse_meta,
    )
    job_fit = calculate_job_fit(
        skill_details=skill_details,
        job_requirements=job_requirements,
        qualification_evidence=qualification_evidence,
        resume_sections=sections,
    )

    recommendations = get_skill_recommendations(
        missing_skills=set(skill_details["missing_skills"]),
        industry=industry,
        target_role=target_role,
        experience_level=experience_level,
        location=location,
    )
    fit_plan = generate_job_fit_plan(
        missing_skills=set(skill_details["missing_skills"]),
        target_role=target_role,
        experience_level=experience_level,
        location=location,
        current_skill_score=skill_details["skill_score"],
    )

    salary_data = get_salary_insights(
        target_role=target_role,
        experience_level=experience_level,
        industry=industry,
        location=location,
        local_only=local_only_mode,
    )
    similar_jobs = find_similar_jobs(
        resume_skills=skill_details.get("resume_skills", {}),
        target_role=target_role,
        industry=industry,
        experience_level=experience_level,
        location=location,
        local_only=local_only_mode,
    )
    market_report = generate_market_report(
        skills=skill_details.get("resume_skills", {}),
        location=location,
        industry=industry,
        target_role=target_role,
        local_only=local_only_mode,
    )

    # Job fit and resume/ATS quality are separate dimensions; do not blend them.
    overall_score = job_fit["score"]

    return {
        "analysis_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "file_name": getattr(uploaded_file, "name", "resume"),
        "parser_metadata": parse_meta,
        "resume_text": resume_text,
        "resume_clean": resume_clean,
        "contact_info": contact_info,
        "sections": sections,
        "resume_quality": resume_quality,
        "skill_details": skill_details,
        "job_requirements": job_requirements,
        "qualification_evidence": qualification_evidence,
        "job_fit": job_fit,
        "ats_details": ats_details,
        "semantic_match_score": semantic_score,
        "local_ai": {
            "enabled": enable_local_ai,
            "ollama_model": ollama_model,
            "structured_profile": structured_profile,
            "gap_analysis": gap_analysis,
            "tailored_suggestions": tailored_suggestions,
            "cover_letter": cover_letter,
        },
        "recommendations": recommendations,
        "fit_plan": fit_plan,
        "salary_data": salary_data,
        "similar_jobs": similar_jobs,
        "market_report": market_report,
        "overall_score": overall_score,
    }


def run_single_analysis_safely(**kwargs) -> Dict[str, Any]:
    try:
        return run_single_analysis(**kwargs)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception:
        # Avoid exposing parser, model, service, or document details in the UI.
        st.error("Analysis could not be completed. Check the uploaded document and local processing setup, then try again.")
        st.stop()


def render_score_cards(result: Dict[str, Any], label: str):
    st.subheader(f"{label} Assessment")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Job Fit", f"{result['overall_score']:.1f}%" if result.get("overall_score") is not None else "Not assessable")
    with col2:
        st.metric("Resume Quality", f"{result['resume_quality']['score']:.1f}%")
    with col3:
        st.metric("ATS Text Heuristic", f"{result['ats_details']['score']:.1f}%")
    with col4:
        st.metric("Lexical Similarity (TF-IDF)", f"{result.get('semantic_match_score', 0.0):.1f}%")
    st.caption("Resume quality evaluates document presentation. Job fit evaluates assessable job requirements. ATS compatibility and TF-IDF lexical similarity are separate signals; none predicts hiring outcomes or competence.")


def render_resume_quality(result: Dict[str, Any]):
    assessment = result["resume_quality"]
    st.subheader("Resume document quality")
    st.caption(assessment["limitations"])
    with st.expander("How this score is calculated"):
        st.write(assessment["method"])
    dimensions = pd.DataFrame(
        [{"Dimension": key.replace("_", " ").title(), "Score": value} for key, value in assessment["dimensions"].items()]
    )
    st.dataframe(dimensions, hide_index=True, use_container_width=True)
    if assessment["recommendations"]:
        st.subheader("Recommended improvements")
        for item in assessment["recommendations"]:
            with st.expander(item["action"]):
                st.write("**Why it matters:** " + item["why"])
                st.write("**How to apply it:** " + item["how"])
    else:
        st.success("The extracted document has the expected sections and contact details. Review the evidence before making edits.")


def render_skills_intelligence(result: Dict[str, Any]):
    st.subheader("Skills intelligence")
    st.caption("Detected mentions and their resume locations. These are not proficiency ratings.")
    details = result.get("skill_details", {})
    matched = {item["skill"]: item for item in details.get("matched_details", [])}
    missing = {item["skill"]: item for item in details.get("missing_details", [])}
    rows = []
    for category, items in details.get("resume_skills", {}).items():
        for item in items:
            skill = item.get("name", "")
            match = matched.get(skill)
            rows.append({
                "Skill": skill,
                "Category": category,
                "Resume evidence location": ", ".join(item.get("sections_found", [])),
                "Evidence": match.get("evidence_level", "Detected in resume; not a direct job requirement match") if match else "Detected in resume; not a direct job requirement match",
                "Confidence": item.get("confidence"),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values(["Category", "Skill"]), hide_index=True, use_container_width=True)
        category_counts = pd.DataFrame(rows).groupby("Category", as_index=False).size().rename(columns={"size": "Detected skills"})
        st.plotly_chart(px.bar(category_counts, x="Category", y="Detected skills", title="Detected skills by category"), use_container_width=True)
    else:
        st.info("No catalog skills were detected. The built-in list is limited, so this does not mean the resume has no useful skills.")

    st.markdown("#### Job requirements and gaps")
    if missing:
        st.dataframe(pd.DataFrame([
            {
                "Requirement": skill,
                "Type": item.get("requirement_type", "unspecified"),
                "Evidence": item.get("evidence_level", ""),
                "Related resume skills": ", ".join(item.get("related_resume_skills", [])),
            }
            for skill, item in missing.items()
        ]), hide_index=True, use_container_width=True)
    elif matched:
        st.success("All cataloged job skills were detected directly in the resume. Review each evidence location for context.")
    else:
        st.info("No cataloged job skills were extracted from this posting.")


def render_career_coach(result: Dict[str, Any], key_suffix: str = "main"):
    st.subheader("Career coach")
    if not result.get("local_ai", {}).get("enabled"):
        st.info("Enable the configured Ollama service before analysis to use the evidence-based chat coach. The score and recommendation panels remain available without AI.")
        return

    state_key = f"career_coach_messages_{key_suffix}"
    messages = st.session_state.setdefault(state_key, [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask about your fit, gaps, or next steps", key=f"career_coach_input_{key_suffix}")
    if question:
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Reviewing the analysis evidence..."):
                try:
                    answer = generate_career_coach_response(
                        question,
                        result,
                        settings=LocalAISettings(ollama_model=result["local_ai"].get("ollama_model", "llama3.1")),
                    )
                except Exception:
                    answer = ""
            if not answer:
                answer = "The configured Ollama service did not return an answer. Check that it is running and the selected model is available."
            st.markdown(answer)
        messages.append({"role": "assistant", "content": answer})


def render_explainability(result: Dict[str, Any], label: str):
    st.subheader(f"{label} Explainability")
    with st.expander("How the scores are calculated"):
        job_fit = result.get("job_fit", {})
        st.markdown("**Job-specific fit**")
        st.write(job_fit.get("method", result["skill_details"].get("score_method", "Job-fit scoring method unavailable.")))
        if job_fit.get("category_scores"):
            st.dataframe(pd.DataFrame([
                {
                    "Category": key.replace("_", " ").title(),
                    "Score": value,
                    "Weight used": job_fit.get("weights_used", {}).get(key, 0),
                    "Evidence": job_fit.get("category_evidence", {}).get(key, {}).get("assessable", "n/a"),
                }
                for key, value in job_fit["category_scores"].items()
            ]), hide_index=True, use_container_width=True)
            st.caption(f"Evidence coverage across configured categories: {job_fit.get('evidence_coverage_percent', 0):.1f}%")
        if job_fit.get("requirements"):
            st.markdown("**Requirement-level evidence used for job fit**")
            st.dataframe(pd.DataFrame([
                {
                    "Requirement": item["requirement"],
                    "Classification": item["classification"],
                    "Posting type": item["requirement_type"],
                    "Resume sources": ", ".join(item["resume_sources"]),
                    "Evidence contribution": item["evidence_weight"],
                    "Interpretation": item["interpretation"],
                }
                for item in job_fit["requirements"]
            ]), hide_index=True, use_container_width=True)
        st.markdown("**ATS compatibility**")
        st.write(result["ats_details"].get("methodology", {}).get("description", "ATS text heuristics; not a prediction of acceptance by any particular ATS."))
        st.markdown("**Resume quality**")
        st.write(result.get("resume_quality", {}).get("method", "Resume quality methodology unavailable."))
        st.markdown("**Text similarity**")
        st.write("TF-IDF lexical similarity measures shared vocabulary only. It is not semantic understanding, qualification coverage, or proof of competence.")
        ats_method = result["ats_details"].get("methodology", {})
        st.write(ats_method.get("limitations", "ATS score limitations unavailable."))
        st.json(ats_method.get("weights", {}))

    matched = result["skill_details"]["matched_details"]
    missing = result["skill_details"]["missing_details"]

    requirements = result.get("job_requirements", {})
    if requirements:
        evidence = result.get("qualification_evidence", {})
        evidence_rows = [
            {
                "Category": category.title(),
                "Job requirement": item["requirement"],
                "Evidence status": item["status"],
                "Resume evidence": item["resume_evidence"] or "—",
            }
            for category in ("education", "experience", "certifications")
            for item in evidence.get(category, [])
        ]
        if evidence_rows:
            st.markdown("**Other qualification evidence**")
            st.caption(
                "Evidence is based only on extracted resume text. Years are self-reported, "
                "credentials are not verified, and these checks do not affect the numeric job-fit score."
            )
            st.dataframe(pd.DataFrame(evidence_rows), hide_index=True, use_container_width=True)

        with st.expander("Extracted job description details", expanded=False):
            st.caption(requirements.get("extraction_method", "Rule-based extraction; review the source."))
            st.caption("Education, experience-duration, and certification lines are surfaced for review but are not included in the job-fit score.")
            summary = {key.replace("_", " ").title(): requirements.get(key) for key in ("title", "company", "location") if requirements.get(key)}
            if summary:
                st.json(summary)
            skill_rows = requirements.get("skills", [])
            if skill_rows:
                st.markdown("Detected skills and requirement labels")
                st.dataframe(pd.DataFrame(skill_rows), hide_index=True, use_container_width=True)
            for heading, field in (("Responsibilities", "responsibilities"), ("Required or general qualification lines", "required_items"), ("Preferred qualification lines", "preferred_items"), ("Education", "education_requirements"), ("Experience", "experience_requirements"), ("Certifications", "certifications")):
                values = requirements.get(field, [])
                if values:
                    st.markdown(f"**{heading}**")
                    for value in values:
                        st.write(f"- {value}")

    coverage = result["skill_details"].get("requirement_coverage", {})
    if coverage:
        st.markdown("Job requirement coverage")
        st.caption("Required/preferred labels appear only when nearby wording in the posting signals them. Other skills remain unspecified.")
        st.dataframe(pd.DataFrame([
            {"Requirement type": label.title(), "Detected": values["detected"], "Matched": values["matched"]}
            for label, values in coverage.items()
        ]), hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Matched skills")
        if matched:
            st.dataframe(pd.DataFrame(matched), use_container_width=True)
        else:
            st.info("No matched skills were detected.")
    with col2:
        st.markdown("Missing skills and fixes")
        if missing:
            missing_table = []
            for item in missing:
                missing_table.append(
                    {
                        "skill": item["skill"],
                        "why_it_matters": item["why_it_matters"],
                        "evidence_level": item.get("evidence_level", ""),
                        "related_resume_skills": ", ".join(item.get("related_resume_skills", [])),
                        "related_skill_note": item.get("related_skill_note", ""),
                        "before": item["fix_example"]["before"],
                        "after": item["fix_example"]["after"],
                    }
                )
            st.dataframe(pd.DataFrame(missing_table), use_container_width=True)
        else:
            st.success("No missing skills detected for this job description.")

    st.markdown("ATS component explainability")
    component_scores = result["ats_details"].get("component_scores", {})
    component_explanations = result["ats_details"].get("explanations", {})
    rows = []
    for component, score in component_scores.items():
        exp = component_explanations.get(component, {})
        rows.append(
            {
                "component": component,
                "score": score,
                "matched": ", ".join(exp.get("what_matched", [])[:3]),
                "missing": ", ".join(exp.get("what_is_missing", [])[:3]),
                "why_it_matters": exp.get("why_it_matters", ""),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("Before and after bullet examples")
    examples = result["ats_details"].get("before_after_examples", [])
    if examples:
        st.dataframe(pd.DataFrame(examples), use_container_width=True)

    local_ai = result.get("local_ai", {})
    if local_ai.get("enabled"):
        st.markdown("---")
        st.subheader("Local AI insights")

        profile = local_ai.get("structured_profile", {})
        if profile:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Detected profession", profile.get("profession", "") or "-")
            with col2:
                st.metric("Detected target role", profile.get("target_role", "") or "-")
            with col3:
                st.metric("Detected seniority", profile.get("seniority", "") or "-")

            with st.expander("Structured profile (JSON)"):
                st.json(profile)

        gap = local_ai.get("gap_analysis", {})
        missing_keywords = gap.get("missing_keywords", []) if isinstance(gap, dict) else []
        if missing_keywords:
            st.markdown("Missing keywords (AI-derived)")
            st.dataframe(pd.DataFrame([{"missing_keyword": kw} for kw in missing_keywords]), use_container_width=True)

        suggestions = local_ai.get("tailored_suggestions", [])
        if suggestions:
            st.markdown("Tailored resume suggestions")
            try:
                st.dataframe(pd.DataFrame(suggestions), use_container_width=True)
            except Exception:
                st.json(suggestions)


def render_role_plan(result: Dict[str, Any]):
    st.subheader("Role-specific optimization and 30/60/90 plan")

    rec = result["recommendations"]
    st.markdown("Priority missing skills")
    if rec["priority_skills"]:
        st.dataframe(pd.DataFrame(rec["priority_skills"]), use_container_width=True)
    else:
        st.info("No missing skills to prioritize.")

    fit_plan = result["fit_plan"]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detected skill gaps", len(fit_plan["top_missing_skills"]))
        st.caption("Learning suggestions do not guarantee a change in job fit.")
    with col2:
        st.markdown("Certification suggestions")
        for cert in fit_plan["certification_suggestions"]:
            st.write(f"- {cert}")

    plan_df = pd.DataFrame(
        {
            "30 Days": fit_plan["plan_30_days"],
            "60 Days": fit_plan["plan_60_days"],
            "90 Days": fit_plan["plan_90_days"],
        }
    )
    st.dataframe(plan_df, use_container_width=True)

    local_ai = result.get("local_ai", {})
    if local_ai.get("enabled"):
        cover_letter = local_ai.get("cover_letter", "")
        if cover_letter:
            st.markdown("---")
            st.subheader("AI-generated cover letter")
            st.text_area("Cover letter", value=cover_letter, height=320)


def render_market_data(result: Dict[str, Any]):
    st.subheader("Market data and real-world job fit")
    salary = result["salary_data"]
    source = salary.get("source", "unknown")
    if source != "live_api_adzuna":
        st.info("Live salary data is unavailable. The figures below are illustrative reference data, not current market estimates.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min Salary", f"${salary['min']:,}")
    with col2:
        st.metric("Max Salary", f"${salary['max']:,}")
    with col3:
        st.metric("Average Salary", f"${salary['average']:,}")
    with col4:
        st.metric("Data Source", salary.get("source", "unknown"))

    jobs = result["similar_jobs"]
    st.markdown("Similar jobs")
    live_jobs = [job for job in jobs if job.get("source") == "live_api_adzuna"]
    if live_jobs:
        st.dataframe(
            pd.DataFrame(live_jobs)[["title", "company", "location", "match_score", "source"]].head(10),
            use_container_width=True,
        )
    else:
        st.info("No live job listings are available for this search. Role benchmarks are not vacancies.")

    demand = result["market_report"].get("skill_demand", {})
    if demand:
        demand_df = pd.DataFrame(
            [
                {"skill": skill, "demand_score": details["demand_score"]}
                for skill, details in demand.items()
            ]
        ).sort_values("demand_score", ascending=False)
        source = next(iter(demand.values())).get("source", "unknown")
        chart_title = "Configured illustrative skill reference" if source != "live_api_adzuna" else "Relative skill mentions in retrieved listings"
        st.caption("Illustrative reference values are not current market measurements." if source != "live_api_adzuna" else "Relative counts within retrieved listings; not a labor-market trend.")
        fig = px.bar(demand_df.head(10), x="skill", y="demand_score", title=chart_title)
        st.plotly_chart(fig, use_container_width=True)


def render_application_progress(result: Dict[str, Any], key_suffix: str):
    analysis_id = result.get("persisted_analysis_id")
    if not analysis_id:
        return
    store = get_store()
    history = store.list_analysis_history(limit=100)
    saved_status = next((item["application_status"] for item in history if item["id"] == analysis_id), "Not tracked")
    status_options = ["Not tracked", "Saved", "Applied", "Interview", "Offer", "Rejected"]
    status = st.selectbox(
        "Application progress", status_options, index=status_options.index(saved_status),
        key=f"application_status_{key_suffix}_{analysis_id}",
    )
    if st.button("Update application progress", key=f"save_application_status_{key_suffix}_{analysis_id}"):
        store.update_application_status(analysis_id, status)
        st.success("Application progress updated in local workflow history.")


def render_workflow_and_export(result: Dict[str, Any], no_data_retained: bool):
    st.subheader("Workflow, versioning, and export")

    if no_data_retained:
        st.info("No Data Retained mode is active. Results are not saved to the local database.")
    else:
        store = get_store()
        history = store.list_analysis_history(limit=50)
        versions = store.list_resume_versions(limit=20)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Saved resume versions")
            st.dataframe(pd.DataFrame(versions), use_container_width=True)
        with col2:
            st.markdown("Analysis history")
            st.dataframe(pd.DataFrame(history), use_container_width=True)
        render_application_progress(result, "single")

    export_text = build_ats_safe_export(
        contact_info=result["contact_info"],
        sections=result["sections"],
        target_role=result["fit_plan"]["target_role"],
        missing_skills=result["skill_details"]["missing_skills"],
        suggestions=result["ats_details"]["suggestions"],
    )
    st.download_button(
        "Download optimization notes (.txt)",
        data=export_text,
        file_name=f"resume_optimization_notes_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )


def render_optimization_studio(result: Dict[str, Any]):
    st.subheader("Resume optimization studio")
    st.caption("Edit your extracted resume content directly. Original text is preserved in this session; export contains only the content shown below.")
    analysis_key = hashlib.sha256(
        f"{result.get('file_name')}:{result.get('analysis_time')}".encode("utf-8")
    ).hexdigest()[:12]
    editable_sections = {}
    suggestions = result.get("local_ai", {}).get("tailored_suggestions", [])
    if suggestions:
        st.markdown("AI suggestions for review")
        st.warning("Suggested wording must be checked for factual accuracy. A suggestion can only be applied when its quoted original text exactly matches an extracted section.")
    for section_name, content in result.get("sections", {}).items():
        key = f"resume_edit_{analysis_key}_{section_name}"
        if key not in st.session_state:
            st.session_state[key] = str(content)
        with st.expander(section_name.replace("_", " ").title(), expanded=section_name in {"summary", "experience"}):
            for index, suggestion in enumerate(suggestions):
                if str(suggestion.get("section", "")).strip().lower() != section_name.lower():
                    continue
                original = str(suggestion.get("original_text", ""))
                proposed = str(suggestion.get("example_rewrite", ""))
                if not original or not proposed or original not in str(content):
                    st.info(f"Suggestion {index + 1} could not be safely anchored to extracted {section_name} text; review it manually below.")
                    continue
                choice_key = f"ai_suggestion_choice_{analysis_key}_{index}"
                choice = st.selectbox(
                    f"{suggestion.get('issue', f'Suggestion {index + 1}')} — review action",
                    ["Keep current wording", "Use suggested wording", "Reject suggestion"],
                    key=choice_key,
                )
                st.caption(suggestion.get("recommendation", ""))
                st.markdown("**Original quote**")
                st.write(original)
                st.markdown("**Proposed rewrite**")
                st.write(proposed)
                marker = f"{analysis_key}:{index}"
                applied = set(st.session_state.get("applied_ai_suggestions", []))
                if choice == "Use suggested wording" and marker not in applied:
                    st.session_state[key] = st.session_state[key].replace(original, proposed, 1)
                    applied.add(marker)
                elif choice in {"Keep current wording", "Reject suggestion"} and marker in applied:
                    st.session_state[key] = st.session_state[key].replace(proposed, original, 1)
                    applied.discard(marker)
                st.session_state["applied_ai_suggestions"] = sorted(applied)
            left, right = st.columns(2)
            with left:
                st.caption("Extracted original")
                st.text_area("Original content", value=str(content), height=170, disabled=True, key=f"original_{analysis_key}_{section_name}")
            with right:
                st.caption("Your editable version")
                st.text_area("Edit this section", key=key, height=170, label_visibility="collapsed")
        editable_sections[section_name] = st.session_state[key]

    if not editable_sections:
        st.info("No labeled resume sections were extracted. Use the extracted resume export and review its contents before editing.")
    else:
        docx_bytes = build_resume_docx(result.get("contact_info", {}), editable_sections)
        st.download_button(
            "Download edited resume (.docx)",
            data=docx_bytes,
            file_name=f"edited_resume_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key=f"download_edited_resume_{analysis_key}",
        )
        st.download_button(
            "Download edited resume (.pdf)",
            data=build_resume_pdf(result.get("contact_info", {}), editable_sections),
            file_name=f"edited_resume_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key=f"download_edited_resume_pdf_{analysis_key}",
        )
        if st.button("Reset edits to extracted text", key=f"reset_resume_edits_{analysis_key}"):
            for section_name, content in result.get("sections", {}).items():
                st.session_state[f"resume_edit_{analysis_key}_{section_name}"] = str(content)
            active_markers = {f"{analysis_key}:{index}" for index in range(len(suggestions))}
            st.session_state["applied_ai_suggestions"] = [marker for marker in st.session_state.get("applied_ai_suggestions", []) if marker not in active_markers]
            st.rerun()
    resume_docx = build_resume_docx(result["contact_info"], result["sections"])
    st.download_button(
        "Download extracted resume (.docx)",
        data=resume_docx,
        file_name=f"resume_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        help="This reformats extracted content only. Review it for parsing errors before using it.",
    )

    report_json = json.dumps(result, indent=2, default=str)
    st.download_button(
        "Download full analysis report (.json)",
        data=report_json,
        file_name=f"analysis_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )


def persist_result(
    result: Dict[str, Any],
    target_role: str,
    location: str,
    industry: str,
    experience_level: str,
    no_data_retained: bool,
    resume_id: Optional[int] = None,
) -> Optional[int]:
    if no_data_retained:
        return None

    store = get_store()
    if resume_id is None:
        resume_id = store.save_resume_version(
            file_name=result["file_name"],
            resume_text=result["resume_text"],
            contact_info=result["contact_info"],
            sections=result["sections"],
            version_label=f"{target_role} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        )
    analysis_id = store.save_analysis(
        resume_id=resume_id,
        target_role=target_role,
        location=location,
        industry=industry,
        experience_level=experience_level,
        skill_score=result["overall_score"] if result["overall_score"] is not None else result["skill_details"]["skill_score"],
        ats_score=result["ats_details"]["score"],
        overall_score=result["overall_score"],
        details=result,
    )
    result["persisted_analysis_id"] = analysis_id
    return resume_id


st.title("Smart Resume Analyzer + Job Fit Matcher")
st.caption("Understand your resume, compare opportunities, and make evidence-backed improvements.")
if "analysis_mode" not in st.session_state:
    st.session_state["analysis_mode"] = "Analyze Resume + Job"
action_cols = st.columns(3)
quick_actions = [
    ("Analyze a resume", "Analyze Resume + Job"),
    ("Compare resumes", "Compare Two Resumes"),
    ("Compare job opportunities", "Compare Job Opportunities"),
]
for column, (label, mode) in zip(action_cols, quick_actions):
    with column:
        if st.button(label, use_container_width=True, key=f"quick_{mode}"):
            st.session_state["analysis_mode"] = mode
            st.rerun()
if "result_a" not in st.session_state and "job_comparison_results" not in st.session_state:
    st.info("Start by choosing a workspace, uploading your resume, and adding a job description. You can run local analysis without saving your data.")

with st.sidebar:
    st.header("Analysis Setup")
    analysis_mode = st.radio(
        "Workspace",
        ["Analyze Resume + Job", "Compare Two Resumes", "Compare Job Opportunities"],
        key="analysis_mode",
    )

    role_options = sorted(ROLE_PLAYBOOKS.keys())
    selected_role = st.selectbox("Target Role", role_options, index=0)
    custom_role = st.text_input("Custom Role (optional)")
    target_role = custom_role.strip() if custom_role.strip() else selected_role

    industry = st.selectbox(
        "Industry",
        ["Technology", "Healthcare", "Finance", "Marketing", "Education", "Engineering", "General"],
    )
    experience_level = st.selectbox("Experience Level", ["Entry Level", "Mid Level", "Senior Level", "Executive"])
    location = st.text_input("Target Location", value="United States")

    local_only_mode = st.toggle("Local-only processing", value=True)
    no_data_retained = st.toggle("No data retained (recommended)", value=True,
        help="Analysis is not written to SQLite. Results remain in this Streamlit session until it ends.")
    enable_ocr = st.toggle("Enable OCR for image PDFs", value=True)

    enable_local_ai = st.toggle("Send resume to configured Ollama service for AI suggestions", value=False,
        help="When enabled, resume and job text are sent to the Ollama service configured below.")
    ollama_model = st.text_input("Ollama model", value="llama3.1")

    if st.button("Clear stored workflow history"):
        get_store().clear_all_history()
        st.success("Stored history cleared.")

uploaded_resume_a = st.file_uploader(
    "Upload Resume A",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
    help="PDF, DOCX, TXT, and image resume formats are supported.",
)

uploaded_resume_b = None
if analysis_mode == "Compare Two Resumes":
    uploaded_resume_b = st.file_uploader(
        "Upload Resume B (for side-by-side comparison)",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
    )

uploaded_job_file = st.file_uploader(
    "Upload job description (optional)",
    type=["pdf", "docx", "txt"],
    help="Upload a PDF, DOCX, or TXT posting, or paste it below.",
    disabled=analysis_mode == "Compare Job Opportunities",
    key="job_description_upload",
)
if uploaded_job_file:
    upload_identity = f"{uploaded_job_file.name}:{uploaded_job_file.size}"
    if st.session_state.get("loaded_job_description_upload") != upload_identity:
        try:
            validate_resume_upload(uploaded_job_file)
            extracted_job = extract_text_with_metadata(uploaded_job_file, enable_ocr=False).get("text", "")
            if not extracted_job.strip():
                st.warning("No text was extracted from that job description. Paste the posting below instead.")
            else:
                st.session_state["job_description_input"] = extracted_job
                st.session_state["loaded_job_description_upload"] = upload_identity
                st.success("Job description text extracted. Review it before running analysis.")
        except ValueError as exc:
            st.error(str(exc))

job_comparison_inputs = []
if analysis_mode == "Compare Job Opportunities":
    opportunity_count = st.selectbox("Number of opportunities", [3, 4, 5], index=0)
    for index in range(opportunity_count):
        with st.expander(f"Opportunity {index + 1}", expanded=index < 3):
            role_title = st.text_input(f"Job title {index + 1}", key=f"compare_job_title_{index}")
            description = st.text_area(
                f"Job description {index + 1}", height=170,
                placeholder="Paste the full responsibilities and qualifications.",
                key=f"compare_job_description_{index}",
            )
            job_comparison_inputs.append((role_title.strip(), description.strip()))
    job_description = job_comparison_inputs[0][1] if job_comparison_inputs else ""
else:
    job_description = st.text_area(
        "Paste Job Description",
        height=220,
        placeholder="Paste the complete role requirements and responsibilities.",
        key="job_description_input",
    )

if st.button("Run Analysis", type="primary", use_container_width=True):
    if analysis_mode == "Compare Job Opportunities":
        completed_jobs = [(title, description) for title, description in job_comparison_inputs if description]
        if not uploaded_resume_a:
            st.error("Upload a resume before comparing job opportunities.")
        elif len(completed_jobs) < 3:
            st.error("Provide at least three job descriptions to compare opportunities.")
        elif uploaded_resume_a.size > 10 * 1024 * 1024:
            st.error("Resume files must be 10 MB or smaller.")
        else:
            comparison_results = {}
            shared_resume_id = None
            for index, (role_title, description) in enumerate(completed_jobs, start=1):
                role = role_title or target_role
                with st.spinner(f"Analyzing opportunity {index} of {len(completed_jobs)}..."):
                    comparison_results[f"{role} · Opportunity {index}"] = run_single_analysis_safely(
                        uploaded_file=uploaded_resume_a,
                        job_description=description,
                        target_role=role,
                        industry=industry,
                        experience_level=experience_level,
                        location=location,
                        enable_ocr=enable_ocr,
                        local_only_mode=local_only_mode,
                        enable_local_ai=enable_local_ai,
                        ollama_model=ollama_model,
                    )
                shared_resume_id = persist_result(
                    comparison_results[f"{role} · Opportunity {index}"],
                    role,
                    location,
                    industry,
                    experience_level,
                    no_data_retained,
                    resume_id=shared_resume_id,
                )
            st.session_state["job_comparison_results"] = comparison_results
            st.session_state.pop("result_a", None)
            st.session_state.pop("result_b", None)
    elif not uploaded_resume_a or not job_description.strip():
        st.error("Upload at least Resume A and provide a job description.")
    elif analysis_mode == "Compare Two Resumes" and not uploaded_resume_b:
        st.error("Upload Resume B to compare two resumes.")
    elif uploaded_resume_b and analysis_mode == "Compare Two Resumes" and uploaded_resume_b.size > 10 * 1024 * 1024:
        st.error("Resume files must be 10 MB or smaller.")
    elif uploaded_resume_a.size > 10 * 1024 * 1024:
        st.error("Resume files must be 10 MB or smaller.")
    else:
        with st.spinner("Analyzing Resume A..."):
            result_a = run_single_analysis_safely(
                uploaded_file=uploaded_resume_a,
                job_description=job_description,
                target_role=target_role,
                industry=industry,
                experience_level=experience_level,
                location=location,
                enable_ocr=enable_ocr,
                local_only_mode=local_only_mode,
                enable_local_ai=enable_local_ai,
                ollama_model=ollama_model,
            )
        persist_result(result_a, target_role, location, industry, experience_level, no_data_retained)
        st.session_state["result_a"] = result_a
        st.session_state.pop("job_comparison_results", None)
        st.session_state.pop("career_coach_messages_main", None)

        if analysis_mode == "Compare Two Resumes" and uploaded_resume_b:
            with st.spinner("Analyzing Resume B..."):
                result_b = run_single_analysis_safely(
                    uploaded_file=uploaded_resume_b,
                    job_description=job_description,
                    target_role=target_role,
                    industry=industry,
                    experience_level=experience_level,
                    location=location,
                    enable_ocr=enable_ocr,
                    local_only_mode=local_only_mode,
                    enable_local_ai=enable_local_ai,
                    ollama_model=ollama_model,
                )
            persist_result(result_b, target_role, location, industry, experience_level, no_data_retained)
            st.session_state["result_b"] = result_b
        else:
            st.session_state["result_b"] = None

if "job_comparison_results" in st.session_state:
    comparison_results = st.session_state["job_comparison_results"]
    st.header("Job opportunity comparison")
    st.caption("Compare detected qualification coverage and evidence. Job fit is a heuristic score, not a hiring probability.")
    comparison_rows = []
    for label, result in comparison_results.items():
        matched_count = len(result["skill_details"].get("matched_details", []))
        missing_count = len(result["skill_details"].get("missing_details", []))
        requirement_coverage = result["skill_details"].get("requirement_coverage", {})
        required = requirement_coverage.get("required", {"detected": 0, "matched": 0})
        preferred = requirement_coverage.get("preferred", {"detected": 0, "matched": 0})
        qualification_evidence = result.get("qualification_evidence", {})

        def evidence_summary(category):
            rows = qualification_evidence.get(category, [])
            return "; ".join(item["status"] for item in rows) if rows else "Not specified"

        comparison_rows.append({
            "Opportunity": label,
            "Detected job fit": result["overall_score"],
            "Required skills": f"{required['matched']}/{required['detected']} matched" if required["detected"] else "Not detected",
            "Preferred skills": f"{preferred['matched']}/{preferred['detected']} matched" if preferred["detected"] else "Not detected",
            "Education evidence": evidence_summary("education"),
            "Experience evidence": evidence_summary("experience"),
            "Resume quality": result["resume_quality"]["score"],
            "ATS text heuristic": result["ats_details"]["score"],
            "Matched skills": matched_count,
            "Unmatched skills": missing_count,
        })
    comparison_df = pd.DataFrame(comparison_rows).sort_values("Detected job fit", ascending=False, na_position="last")
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
    assessable_comparisons = comparison_df[comparison_df["Detected job fit"].notna()]
    if not assessable_comparisons.empty:
        strongest = assessable_comparisons.iloc[0]
        st.info(f"Highest detected job fit: {strongest['Opportunity']} ({strongest['Detected job fit']:.1f}%). Review its missing requirements and evidence before deciding.")
    else:
        st.info("The provided postings did not contain enough extractable requirements to calculate a comparable job-fit score.")

    report_tabs = st.tabs(list(comparison_results.keys()))
    for tab, (label, result) in zip(report_tabs, comparison_results.items()):
        with tab:
            render_score_cards(result, label)
            render_explainability(result, label)
            render_resume_quality(result)
            if not no_data_retained:
                render_application_progress(result, f"comparison_{list(comparison_results).index(label)}")
    coach_label = st.selectbox("Coach context", list(comparison_results.keys()))
    coach_index = list(comparison_results).index(coach_label)
    render_career_coach(comparison_results[coach_label], key_suffix=f"job_{coach_index}")

if "result_a" in st.session_state:
    result_a = st.session_state["result_a"]
    result_b = st.session_state.get("result_b")

    tab_overview, tab_resume, tab_skills, tab_explain, tab_role, tab_market, tab_coach, tab_optimize, tab_workflow, tab_privacy = st.tabs(
        ["Career Dashboard", "Resume Quality", "Skills Intelligence", "Explainability", "Role Plan", "Market Data", "Career Coach", "Optimize Resume", "Workflow + Export", "Privacy + Validation"]
    )

    with tab_overview:
        st.caption("Your current resume and opportunity snapshot. Scores are heuristic guidance, not hiring predictions.")
        render_score_cards(result_a, "Resume A")
        top_actions = result_a.get("resume_quality", {}).get("recommendations", [])[:3]
        if top_actions:
            st.markdown("#### Recommended next actions")
            for item in top_actions:
                st.write(f"- **{item['action']}** — {item['how']}")
        skill_names = [item.get("skill") for item in result_a["skill_details"].get("matched_details", [])]
        if skill_names:
            st.markdown("#### Evidence-backed skills detected")
            st.write(", ".join(skill_names[:12]))
        if result_b:
            render_score_cards(result_b, "Resume B")
            if result_a["overall_score"] is not None and result_b["overall_score"] is not None:
                delta = round(result_a["overall_score"] - result_b["overall_score"], 2)
                winner = "Resume A" if delta >= 0 else "Resume B"
                st.info(f"Comparison winner: {winner} ({abs(delta):.2f}% job-fit score difference).")
            else:
                st.info("Job fit could not be scored for both resumes because one or more assessments lacked extractable evidence.")

            compare_df = pd.DataFrame(
                [
                    {
                        "Resume": "A",
                        "Skill Score": result_a["skill_details"]["skill_score"],
                        "ATS Score": result_a["ats_details"]["score"],
                        "Overall": result_a["overall_score"],
                    },
                    {
                        "Resume": "B",
                        "Skill Score": result_b["skill_details"]["skill_score"],
                        "ATS Score": result_b["ats_details"]["score"],
                        "Overall": result_b["overall_score"],
                    },
                ]
            )
            st.dataframe(compare_df, use_container_width=True)
            fig = px.bar(compare_df, x="Resume", y=["Skill Score", "ATS Score", "Overall"], barmode="group")
            st.plotly_chart(fig, use_container_width=True)

    with tab_resume:
        render_resume_quality(result_a)
        if result_b:
            st.markdown("---")
            render_resume_quality(result_b)

    with tab_skills:
        render_skills_intelligence(result_a)
        if result_b:
            st.markdown("---")
            st.subheader("Resume B skills")
            render_skills_intelligence(result_b)

    with tab_explain:
        render_explainability(result_a, "Resume A")
        if result_b:
            st.markdown("---")
            render_explainability(result_b, "Resume B")

    with tab_role:
        render_role_plan(result_a)
        if result_b:
            st.markdown("---")
            st.subheader("Resume B role plan")
            render_role_plan(result_b)

    with tab_market:
        render_market_data(result_a)
        if result_b:
            st.markdown("---")
            st.subheader("Resume B market profile")
            render_market_data(result_b)

    with tab_coach:
        render_career_coach(result_a)

    with tab_optimize:
        render_optimization_studio(result_a)

    with tab_workflow:
        render_workflow_and_export(result_a, no_data_retained)

    with tab_privacy:
        st.subheader("Privacy and trust controls")
        st.text(PRIVACY_POLICY_TEXT.strip())
        st.markdown(
            "Current runtime mode: "
            + ("local-only processing enabled" if local_only_mode else "external market API calls allowed")
            + ", "
            + ("no data retained enabled." if no_data_retained else "local workflow history enabled.")
        )

        st.markdown("---")
        st.subheader("Validation toolkit")
        st.write(
            "Upload a CSV with columns such as `predicted_skills`, `ground_truth_skills`, "
            "`overall_score`, and `callback_outcome`."
        )
        validation_file = st.file_uploader("Upload validation CSV", type=["csv"], key="validation_csv")
        if validation_file:
            df = pd.read_csv(validation_file)
            metrics = evaluate_from_dataframe(df)
            st.json(metrics)
