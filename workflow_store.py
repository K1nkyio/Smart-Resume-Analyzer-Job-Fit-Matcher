import hashlib
import json
import sqlite3
from contextlib import contextmanager
from io import BytesIO
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from docx import Document
from docx.shared import Inches, Pt


class WorkflowStore:
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyzer_resumes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_label TEXT,
                    file_name TEXT,
                    content_hash TEXT,
                    resume_text TEXT,
                    contact_json TEXT,
                    sections_json TEXT,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyzer_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER,
                    target_role TEXT,
                    location TEXT,
                    industry TEXT,
                    experience_level TEXT,
                    skill_score REAL,
                    ats_score REAL,
                    overall_score REAL,
                    details_json TEXT,
                    callback_outcome INTEGER,
                    application_status TEXT NOT NULL DEFAULT 'Not tracked',
                    created_at TEXT,
                    FOREIGN KEY (resume_id) REFERENCES analyzer_resumes(id)
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(analyzer_analyses)").fetchall()}
            if "application_status" not in columns:
                conn.execute(
                    "ALTER TABLE analyzer_analyses ADD COLUMN application_status TEXT NOT NULL DEFAULT 'Not tracked'"
                )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_analyzer_analyses_created_at
                ON analyzer_analyses(created_at)
                """
            )
            conn.commit()

    def save_resume_version(
        self,
        file_name: str,
        resume_text: str,
        contact_info: Optional[Dict[str, Any]] = None,
        sections: Optional[Dict[str, Any]] = None,
        version_label: Optional[str] = None,
    ) -> int:
        content_hash = hashlib.sha256((resume_text or "").encode("utf-8", errors="ignore")).hexdigest()
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analyzer_resumes
                (version_label, file_name, content_hash, resume_text, contact_json, sections_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_label or now,
                    file_name,
                    content_hash,
                    resume_text,
                    json.dumps(contact_info or {}),
                    json.dumps(sections or {}),
                    now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def list_resume_versions(self, limit: int = 25) -> List[dict]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, version_label, file_name, content_hash, created_at
                FROM analyzer_resumes
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "version_label": row[1],
                "file_name": row[2],
                "content_hash": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    def save_analysis(
        self,
        resume_id: Optional[int],
        target_role: str,
        location: str,
        industry: str,
        experience_level: str,
        skill_score: float,
        ats_score: float,
        overall_score: float,
        details: Optional[Dict[str, Any]] = None,
        callback_outcome: Optional[int] = None,
        application_status: str = "Not tracked",
    ) -> int:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analyzer_analyses
                (resume_id, target_role, location, industry, experience_level,
                 skill_score, ats_score, overall_score, details_json, callback_outcome, application_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resume_id,
                    target_role,
                    location,
                    industry,
                    experience_level,
                    float(skill_score) if skill_score is not None else None,
                    float(ats_score) if ats_score is not None else None,
                    float(overall_score) if overall_score is not None else None,
                    json.dumps(details or {}),
                    callback_outcome,
                    application_status,
                    now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def list_analysis_history(self, limit: int = 100) -> List[dict]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, resume_id, target_role, location, industry, experience_level,
                       skill_score, ats_score, overall_score, callback_outcome, created_at, application_status
                FROM analyzer_analyses
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "resume_id": row[1],
                "target_role": row[2],
                "location": row[3],
                "industry": row[4],
                "experience_level": row[5],
                "skill_score": row[6],
                "ats_score": row[7],
                "overall_score": row[8],
                "callback_outcome": row[9],
                "created_at": row[10],
                "application_status": row[11],
            }
            for row in rows
        ]

    def update_application_status(self, analysis_id: int, status: str) -> None:
        allowed = {"Not tracked", "Saved", "Applied", "Interview", "Offer", "Rejected"}
        if status not in allowed:
            raise ValueError("Unsupported application status.")
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE analyzer_analyses SET application_status = ? WHERE id = ?",
                (status, int(analysis_id)),
            )
            if cursor.rowcount == 0:
                raise ValueError("The selected analysis record was not found.")
            conn.commit()

    def clear_all_history(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM analyzer_analyses")
            conn.execute("DELETE FROM analyzer_resumes")
            conn.commit()


def build_ats_safe_export(
    contact_info: Dict[str, Any],
    sections: Dict[str, Any],
    target_role: str,
    missing_skills: List[str],
    suggestions: List[str],
) -> str:
    lines = []
    lines.append("ATS-SAFE RESUME OPTIMIZATION OUTPUT")
    lines.append(f"Target role: {target_role}")
    lines.append("")

    if contact_info:
        lines.append("CONTACT")
        for key in ["email", "phone", "linkedin", "github", "location", "website"]:
            value = contact_info.get(key)
            if value:
                lines.append(f"{key.title()}: {value}")
        lines.append("")

    ordered_sections = ["summary", "experience", "projects", "education", "skills", "certifications"]
    for section_name in ordered_sections:
        value = sections.get(section_name)
        if value:
            lines.append(section_name.upper())
            lines.append(str(value).strip())
            lines.append("")

    if missing_skills:
        lines.append("TOP MISSING SKILLS TO ADD EVIDENCE FOR")
        for skill in missing_skills[:10]:
            lines.append(f"- {skill}")
        lines.append("")

    if suggestions:
        lines.append("ATS OPTIMIZATION CHECKLIST")
        for suggestion in suggestions[:10]:
            lines.append(f"- {suggestion}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_resume_docx(contact_info: Dict[str, Any], sections: Dict[str, Any]) -> bytes:
    """Build a plain, ATS-readable DOCX from extracted candidate content only."""
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    normal = document.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10)

    contact_info = contact_info or {}
    sections = sections or {}
    contact_values = [
        contact_info.get(key)
        for key in ("email", "phone", "location", "linkedin", "website", "github")
        if contact_info.get(key)
    ]
    name = contact_info.get("name")
    if name:
        title = document.add_paragraph()
        title.style = document.styles["Title"]
        title.add_run(str(name))
    if contact_values:
        document.add_paragraph(" | ".join(str(value) for value in contact_values))

    section_titles = {
        "summary": "Professional Summary",
        "experience": "Experience",
        "projects": "Projects",
        "education": "Education",
        "skills": "Skills",
        "certifications": "Certifications",
        "awards": "Awards and Achievements",
        "languages": "Languages",
    }
    for key, heading in section_titles.items():
        content = sections.get(key)
        if not content:
            continue
        document.add_heading(heading, level=1)
        for raw_line in str(content).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("- ", "* ", "• ")):
                document.add_paragraph(line[2:], style="List Bullet")
            else:
                document.add_paragraph(line)

    output = BytesIO()
    document.save(output)
    return output.getvalue()


def build_resume_pdf(contact_info: Dict[str, Any], sections: Dict[str, Any]) -> bytes:
    """Create a simple, text-based PDF from user-provided resume content."""
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate
    from pathlib import Path
    import reportlab

    font_dir = Path(reportlab.__file__).parent / "fonts"
    if "ResumeVera" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("ResumeVera", str(font_dir / "Vera.ttf")))
    if "ResumeVera-Bold" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("ResumeVera-Bold", str(font_dir / "VeraBd.ttf")))

    output = BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=letter, rightMargin=0.7 * inch, leftMargin=0.7 * inch,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch,
        title="Resume", author="Smart Resume Analyzer",
    )
    base = getSampleStyleSheet()
    name_style = ParagraphStyle("ResumeName", parent=base["Title"], fontName="ResumeVera-Bold", alignment=TA_LEFT, textColor=colors.HexColor("#14233b"), spaceAfter=6)
    heading_style = ParagraphStyle("ResumeHeading", parent=base["Heading2"], fontName="ResumeVera-Bold", textColor=colors.HexColor("#14233b"), spaceBefore=10, spaceAfter=4, keepWithNext=True)
    body_style = ParagraphStyle("ResumeBody", parent=base["BodyText"], fontName="ResumeVera", fontSize=9.5, leading=13, spaceAfter=3)
    contact_info = contact_info or {}
    sections = sections or {}
    story = []
    if contact_info.get("name"):
        story.append(Paragraph(escape(str(contact_info["name"])), name_style))
    contact = [str(contact_info[key]) for key in ("email", "phone", "location", "linkedin", "website", "github") if contact_info.get(key)]
    if contact:
        story.append(Paragraph(escape(" | ".join(contact)), body_style))

    headings = {
        "summary": "Professional Summary", "experience": "Experience", "projects": "Projects",
        "education": "Education", "skills": "Skills", "certifications": "Certifications",
        "awards": "Awards and Achievements", "languages": "Languages",
    }
    for key, heading in headings.items():
        content = sections.get(key)
        if not content:
            continue
        story.append(Paragraph(escape(heading), heading_style))
        for raw_line in str(content).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("- ", "* ", "• ")):
                line = "- " + line[2:]
            story.append(Paragraph(escape(line), body_style))
    doc.build(story)
    return output.getvalue()
