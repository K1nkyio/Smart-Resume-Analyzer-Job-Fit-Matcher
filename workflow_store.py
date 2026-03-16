import hashlib
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


class WorkflowStore:
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

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
                    created_at TEXT,
                    FOREIGN KEY (resume_id) REFERENCES analyzer_resumes(id)
                )
                """
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
        now = datetime.utcnow().isoformat() + "Z"

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
    ) -> int:
        now = datetime.utcnow().isoformat() + "Z"
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analyzer_analyses
                (resume_id, target_role, location, industry, experience_level,
                 skill_score, ats_score, overall_score, details_json, callback_outcome, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resume_id,
                    target_role,
                    location,
                    industry,
                    experience_level,
                    float(skill_score),
                    float(ats_score),
                    float(overall_score),
                    json.dumps(details or {}),
                    callback_outcome,
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
                       skill_score, ats_score, overall_score, callback_outcome, created_at
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
            }
            for row in rows
        ]

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
