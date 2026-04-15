from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "classroom_monitor.db"
SESSION_DAYS = 14


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return None if row is None else dict(row)


def _create_user(conn: sqlite3.Connection, *, user_id: str, name: str, email: str, role: str, department: str, password: str) -> None:
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    conn.execute(
        """
        INSERT OR IGNORE INTO users
        (id, name, email, role, department, password_salt, password_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, name, email.lower(), role, department, salt, password_hash, _utc_now()),
    )


def _seed_demo_data(conn: sqlite3.Connection) -> None:
    _create_user(
        conn,
        user_id="usr_instructor_demo",
        name="Demo Instructor",
        email="instructor@classroom.local",
        role="instructor",
        department="Computer Science",
        password="classroom123",
    )
    _create_user(
        conn,
        user_id="usr_student_demo",
        name="Demo Student",
        email="student@classroom.local",
        role="student",
        department="Computer Science",
        password="classroom123",
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO subjects
        (id, name, code, description, icon_index, total_students, instructor_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sub_demo_dl",
            "Deep Learning",
            "DL26",
            "Local demo subject wired to classroom-monitor.",
            1,
            41,
            "usr_instructor_demo",
            _utc_now(),
        ),
    )


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                role TEXT NOT NULL CHECK(role IN ('instructor', 'student')),
                department TEXT NOT NULL DEFAULT 'Computer Science',
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS subjects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                code TEXT NOT NULL,
                description TEXT,
                icon_index INTEGER NOT NULL DEFAULT 0,
                total_students INTEGER NOT NULL DEFAULT 0,
                instructor_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analysis_runs (
                id TEXT PRIMARY KEY,
                subject_name TEXT NOT NULL,
                subject_id TEXT,
                class_date TEXT NOT NULL,
                topic TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                current_step TEXT NOT NULL DEFAULT '',
                error_message TEXT,
                run_dir TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS run_artifacts (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
                label TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        _seed_demo_data(conn)
        conn.commit()


def public_user(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    return {
        "id": data["id"],
        "name": data["name"],
        "email": data["email"],
        "role": data["role"],
        "department": data["department"],
    }


def login(email: str, password: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE lower(email) = lower(?)", (email.strip(),)).fetchone()
        if row is None:
            return None
        if _hash_password(password, row["password_salt"]) != row["password_hash"]:
            return None
        token = secrets.token_urlsafe(32)
        expires = datetime.utcnow() + timedelta(days=SESSION_DAYS)
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, row["id"], _utc_now(), expires.isoformat(timespec="seconds") + "Z"),
        )
        conn.commit()
        return {"token": token, "user": public_user(row)}


def logout(token: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()


def get_user_for_token(token: str) -> dict[str, Any] | None:
    if not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT users.*
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ? AND sessions.expires_at > ?
            """,
            (token, _utc_now()),
        ).fetchone()
        return public_user(row) if row else None


def list_subjects(user: dict[str, Any]) -> list[dict[str, Any]]:
    with _connect() as conn:
        if user["role"] == "instructor":
            rows = conn.execute(
                "SELECT * FROM subjects WHERE instructor_id = ? ORDER BY created_at DESC",
                (user["id"],),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM subjects ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]


def create_subject(user: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    subject_id = secrets.token_hex(8)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO subjects
            (id, name, code, description, icon_index, total_students, instructor_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject_id,
                str(payload.get("name", "")).strip(),
                str(payload.get("code", "")).strip().upper(),
                str(payload.get("description") or "").strip() or None,
                int(payload.get("icon_index", payload.get("iconIndex", 0)) or 0),
                int(payload.get("total_students", payload.get("totalStudents", 0)) or 0),
                user["id"],
                _utc_now(),
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
        return dict(row)


def delete_subject(user: dict[str, Any], subject_id: str) -> bool:
    with _connect() as conn:
        if user["role"] == "instructor":
            cur = conn.execute("DELETE FROM subjects WHERE id = ? AND instructor_id = ?", (subject_id, user["id"]))
        else:
            cur = conn.execute("DELETE FROM subjects WHERE id = ? AND 1 = 0", (subject_id,))
        conn.commit()
        return cur.rowcount > 0


def upsert_run(
    run_id: str,
    *,
    subject_name: str,
    subject_id: str | None,
    class_date: str,
    topic: str,
    status: str,
    progress: float,
    current_step: str,
    run_dir: str,
    error_message: str | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    now = _utc_now()
    result_json = json.dumps(result) if result is not None else None
    completed_at = now if status in {"completed", "failed"} else None
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO analysis_runs
            (id, subject_name, subject_id, class_date, topic, status, progress, current_step, error_message, run_dir, result_json, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                progress=excluded.progress,
                current_step=excluded.current_step,
                error_message=excluded.error_message,
                result_json=COALESCE(excluded.result_json, analysis_runs.result_json),
                completed_at=COALESCE(excluded.completed_at, analysis_runs.completed_at)
            """,
            (
                run_id,
                subject_name,
                subject_id,
                class_date,
                topic,
                status,
                float(progress),
                current_step,
                error_message,
                run_dir,
                result_json,
                now,
                completed_at,
            ),
        )
        conn.commit()


def replace_artifacts(run_id: str, artifacts: list[dict[str, str]]) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM run_artifacts WHERE run_id = ?", (run_id,))
        for artifact in artifacts:
            conn.execute(
                "INSERT INTO run_artifacts (id, run_id, label, artifact_type, path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    secrets.token_hex(8),
                    run_id,
                    artifact["label"],
                    artifact["type"],
                    artifact["path"],
                    _utc_now(),
                ),
            )
        conn.commit()


def list_run_summaries(subject: str | None = None, date: str | None = None) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if subject:
        clauses.append("lower(subject_name) = lower(?)")
        params.append(subject)
    if date:
        clauses.append("class_date = ?")
        params.append(date)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM analysis_runs {where} ORDER BY created_at DESC",
            params,
        ).fetchall()
        summaries: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            if data.get("result_json"):
                try:
                    payload = json.loads(data["result_json"])
                except Exception:
                    payload = {}
            else:
                payload = {}
            payload.setdefault("run_id", data["id"])
            payload.setdefault("class_date", data["class_date"])
            payload.setdefault("subject_name", data["subject_name"])
            payload.setdefault("topic", data["topic"])
            payload.setdefault("timestamp", data["completed_at"] or data["created_at"])
            payload.setdefault("status", data["status"])
            summaries.append(payload)
        return summaries
