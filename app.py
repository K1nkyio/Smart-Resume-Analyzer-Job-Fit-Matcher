import json
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from ats_scorer import calculate_ats_score_detailed
from config import PRIVACY_POLICY_TEXT, ROLE_PLAYBOOKS
from job_matcher import find_similar_jobs, generate_market_report, get_salary_insights
from local_ai import (
    LocalAISettings,
    extract_structured_profile,
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
)
from skill_matcher import (
    compare_resume_to_job_detailed,
    generate_job_fit_plan,
    get_skill_recommendations,
)
from validation import evaluate_from_dataframe
from workflow_store import WorkflowStore, build_ats_safe_export


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
    parse_meta = extract_text_with_metadata(uploaded_file, enable_ocr=enable_ocr)
    resume_text = parse_meta.get("text", "")
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_description)
    contact_info = extract_contact_info(resume_text)
    sections = extract_sections(resume_text)

    ai_settings = LocalAISettings(ollama_model=ollama_model)
    semantic_score = semantic_match_score(resume_clean, job_clean)
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
        resume_text=resume_clean,
        job_text=job_clean,
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

    overall_score = round(
        (skill_details["skill_score"] + ats_details["score"] + semantic_score) / 3.0,
        2,
    )

    return {
        "analysis_time": datetime.utcnow().isoformat() + "Z",
        "file_name": getattr(uploaded_file, "name", "resume"),
        "parser_metadata": parse_meta,
        "resume_text": resume_text,
        "resume_clean": resume_clean,
        "contact_info": contact_info,
        "sections": sections,
        "skill_details": skill_details,
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


def render_score_cards(result: Dict[str, Any], label: str):
    st.subheader(f"{label} Score Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Score", f"{result['overall_score']:.1f}%")
    with col2:
        st.metric("Skill Match Score", f"{result['skill_details']['skill_score']:.1f}%")
    with col3:
        st.metric("ATS Score", f"{result['ats_details']['score']:.1f}%")

    st.metric("Semantic Match Score (AI)", f"{result.get('semantic_match_score', 0.0):.1f}%")


def render_explainability(result: Dict[str, Any], label: str):
    st.subheader(f"{label} Explainability")

    matched = result["skill_details"]["matched_details"]
    missing = result["skill_details"]["missing_details"]

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
        st.metric("Estimated Fit-Score Gain", f"{fit_plan['estimated_fit_score_gain']:.1f}%")
        st.metric("Projected Skill Score", f"{fit_plan['projected_skill_score']:.1f}%")
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
    if jobs:
        st.dataframe(
            pd.DataFrame(jobs)[["title", "company", "location", "match_score", "source"]].head(10),
            use_container_width=True,
        )
    else:
        st.info("No similar jobs found for the current input.")

    demand = result["market_report"].get("skill_demand", {})
    if demand:
        demand_df = pd.DataFrame(
            [
                {"skill": skill, "demand_score": details["demand_score"]}
                for skill, details in demand.items()
            ]
        ).sort_values("demand_score", ascending=False)
        fig = px.bar(demand_df.head(10), x="skill", y="demand_score", title="Skill demand score in market data")
        st.plotly_chart(fig, use_container_width=True)


def render_workflow_and_export(result: Dict[str, Any], no_data_retained: bool):
    st.subheader("Workflow, versioning, and export")
    store = get_store()

    if no_data_retained:
        st.info("No Data Retained mode is active. Results are not saved to the local database.")
    else:
        history = store.list_analysis_history(limit=50)
        versions = store.list_resume_versions(limit=20)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Saved resume versions")
            st.dataframe(pd.DataFrame(versions), use_container_width=True)
        with col2:
            st.markdown("Analysis history")
            st.dataframe(pd.DataFrame(history), use_container_width=True)

    export_text = build_ats_safe_export(
        contact_info=result["contact_info"],
        sections=result["sections"],
        target_role=result["fit_plan"]["target_role"],
        missing_skills=result["skill_details"]["missing_skills"],
        suggestions=result["ats_details"]["suggestions"],
    )
    st.download_button(
        "Download ATS-safe export (.txt)",
        data=export_text,
        file_name=f"ats_safe_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )

    report_json = json.dumps(result, indent=2, default=str)
    st.download_button(
        "Download full analysis report (.json)",
        data=report_json,
        file_name=f"analysis_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )


def persist_result(
    result: Dict[str, Any],
    target_role: str,
    location: str,
    industry: str,
    experience_level: str,
    no_data_retained: bool,
):
    if no_data_retained:
        return

    store = get_store()
    resume_id = store.save_resume_version(
        file_name=result["file_name"],
        resume_text=result["resume_text"],
        contact_info=result["contact_info"],
        sections=result["sections"],
        version_label=f"{target_role} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
    )
    store.save_analysis(
        resume_id=resume_id,
        target_role=target_role,
        location=location,
        industry=industry,
        experience_level=experience_level,
        skill_score=result["skill_details"]["skill_score"],
        ats_score=result["ats_details"]["score"],
        overall_score=result["overall_score"],
        details=result,
    )


st.title("Smart Resume Analyzer + Job Fit Matcher")
st.caption("Production-focused analysis with explainability, role-tailoring, and market context.")

with st.sidebar:
    st.header("Analysis Setup")
    analysis_mode = st.radio("Mode", ["Single Resume", "Compare Two Resumes"], index=0)

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
    no_data_retained = st.toggle("No data retained mode", value=False)
    enable_ocr = st.toggle("Enable OCR for image PDFs", value=True)

    enable_local_ai = st.toggle("Enable Local AI (Ollama)", value=True)
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

job_description = st.text_area(
    "Paste Job Description",
    height=220,
    placeholder="Paste the complete role requirements and responsibilities.",
)

if st.button("Run Analysis", type="primary", use_container_width=True):
    if not uploaded_resume_a or not job_description.strip():
        st.error("Upload at least Resume A and provide a job description.")
    else:
        with st.spinner("Analyzing Resume A..."):
            result_a = run_single_analysis(
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

        if analysis_mode == "Compare Two Resumes" and uploaded_resume_b:
            with st.spinner("Analyzing Resume B..."):
                result_b = run_single_analysis(
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

if "result_a" in st.session_state:
    result_a = st.session_state["result_a"]
    result_b = st.session_state.get("result_b")

    tab_overview, tab_explain, tab_role, tab_market, tab_workflow, tab_privacy = st.tabs(
        ["Overview", "Explainability", "Role Plan", "Market Data", "Workflow + Export", "Privacy + Validation"]
    )

    with tab_overview:
        render_score_cards(result_a, "Resume A")
        if result_b:
            render_score_cards(result_b, "Resume B")
            delta = round(result_a["overall_score"] - result_b["overall_score"], 2)
            winner = "Resume A" if delta >= 0 else "Resume B"
            st.info(f"Comparison winner: {winner} ({abs(delta):.2f}% overall score difference).")

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
