"""
Session Manager for PT Physio Coach
Handles session persistence (SQLite), rep counting state machine, and progress tracking.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from .exercise_library import get_exercise

logger = logging.getLogger(__name__)

DB_FILENAME = "pt_sessions.db"


def _get_db_path() -> str:
    if os.name == "posix":
        if "darwin" in os.sys.platform.lower():
            config_dir = Path.home() / "Library" / "Application Support" / "live-vlm-webui"
        else:
            config_dir = (
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "live-vlm-webui"
            )
    else:
        config_dir = Path(os.environ.get("APPDATA", Path.home())) / "live-vlm-webui"
    config_dir.mkdir(parents=True, exist_ok=True)
    return str(config_dir / DB_FILENAME)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exercise_id TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    total_reps INTEGER DEFAULT 0,
    avg_form_score REAL DEFAULT 0.0,
    corrections_summary TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS rep_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    phase TEXT,
    form_score REAL DEFAULT 0.0,
    corrections TEXT DEFAULT '[]',
    feedback TEXT DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
"""


class RepCounter:
    """State machine that tracks VLM-reported phases to count reps."""

    def __init__(self, exercise_id: str):
        self.exercise = get_exercise(exercise_id)
        self.reps = 0
        self.last_phase: Optional[str] = None
        self.phase_history: list[str] = []
        self._seen_end_phase = False

    def update(self, phase: str, rep_boundary: bool) -> bool:
        """Process a new phase observation. Returns True if a rep was just completed."""
        if not self.exercise:
            return False

        if phase == self.last_phase:
            return False

        self.last_phase = phase
        self.phase_history.append(phase)
        if len(self.phase_history) > 20:
            self.phase_history = self.phase_history[-20:]

        if rep_boundary:
            self.reps += 1
            self._seen_end_phase = False
            return True

        if phase == self.exercise.rep_end_phase:
            self._seen_end_phase = True
        elif phase == self.exercise.rep_start_phase and self._seen_end_phase:
            self.reps += 1
            self._seen_end_phase = False
            return True

        return False


class SessionManager:
    """Manages exercise sessions, persists to SQLite, tracks reps and form."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _get_db_path()
        self._db: Optional[aiosqlite.Connection] = None
        self._active_session_id: Optional[int] = None
        self._rep_counter: Optional[RepCounter] = None
        self._form_scores: list[float] = []
        self._all_corrections: list[str] = []
        self._paused = False

    async def initialize(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        logger.info(f"Session database initialized at {self.db_path}")

    async def close(self):
        if self._active_session_id:
            await self.end_session()
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def active(self) -> bool:
        return self._active_session_id is not None and not self._paused

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def active_session_id(self) -> Optional[int]:
        return self._active_session_id

    @property
    def current_reps(self) -> int:
        return self._rep_counter.reps if self._rep_counter else 0

    async def start_session(self, exercise_id: str) -> int:
        if self._active_session_id:
            await self.end_session()

        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._db.execute(
            "INSERT INTO sessions (exercise_id, start_time, status) VALUES (?, ?, 'active')",
            (exercise_id, now),
        )
        await self._db.commit()
        self._active_session_id = cursor.lastrowid
        self._rep_counter = RepCounter(exercise_id)
        self._form_scores = []
        self._all_corrections = []
        self._paused = False
        logger.info(f"Session {self._active_session_id} started for exercise {exercise_id}")
        return self._active_session_id

    def pause_session(self):
        self._paused = True

    def resume_session(self):
        self._paused = False

    async def record_frame(self, parsed: dict) -> dict:
        """Record a VLM analysis frame. Returns update dict with rep info."""
        if not self._active_session_id or self._paused or not self._rep_counter:
            return {}

        phase = parsed.get("phase", "")
        form_score = parsed.get("form_score", 0)
        corrections = parsed.get("corrections", [])
        feedback = parsed.get("feedback", "")
        rep_boundary = parsed.get("rep_boundary", False)

        rep_completed = self._rep_counter.update(phase, rep_boundary)

        if isinstance(form_score, (int, float)) and form_score > 0:
            self._form_scores.append(float(form_score))
        for c in corrections:
            if c and c not in self._all_corrections:
                self._all_corrections.append(c)

        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT INTO rep_events (session_id, timestamp, phase, form_score, corrections, feedback) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self._active_session_id, now, phase, form_score, json.dumps(corrections), feedback),
        )
        await self._db.commit()

        result = {
            "phase": phase,
            "form_score": form_score,
            "corrections": corrections,
            "feedback": feedback,
            "total_reps": self._rep_counter.reps,
            "rep_completed": rep_completed,
        }
        return result

    async def end_session(self) -> Optional[dict]:
        if not self._active_session_id:
            return None

        now = datetime.now(timezone.utc).isoformat()
        avg_score = sum(self._form_scores) / len(self._form_scores) if self._form_scores else 0.0
        total_reps = self._rep_counter.reps if self._rep_counter else 0

        await self._db.execute(
            "UPDATE sessions SET end_time=?, total_reps=?, avg_form_score=?, corrections_summary=?, status='completed' WHERE id=?",
            (now, total_reps, round(avg_score, 1), json.dumps(self._all_corrections), self._active_session_id),
        )
        await self._db.commit()

        session_id = self._active_session_id

        cursor = await self._db.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
        row = await cursor.fetchone()
        summary = dict(row) if row else {}

        self._active_session_id = None
        self._rep_counter = None
        self._form_scores = []
        self._all_corrections = []
        self._paused = False

        logger.info(f"Session {session_id} ended: {total_reps} reps, avg score {avg_score:.1f}")
        return summary

    async def get_sessions(self, limit: int = 50, offset: int = 0) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_session_detail(self, session_id: int) -> Optional[dict]:
        cursor = await self._db.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
        session_row = await cursor.fetchone()
        if not session_row:
            return None

        cursor = await self._db.execute(
            "SELECT * FROM rep_events WHERE session_id=? ORDER BY timestamp",
            (session_id,),
        )
        events = await cursor.fetchall()
        result = dict(session_row)
        result["events"] = [dict(e) for e in events]
        return result

    async def get_progress(self, exercise_id: Optional[str] = None) -> dict:
        if exercise_id:
            cursor = await self._db.execute(
                "SELECT * FROM sessions WHERE exercise_id=? AND status='completed' ORDER BY start_time",
                (exercise_id,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM sessions WHERE status='completed' ORDER BY start_time"
            )
        rows = await cursor.fetchall()
        sessions = [dict(r) for r in rows]

        by_exercise: dict[str, list] = {}
        for s in sessions:
            eid = s["exercise_id"]
            by_exercise.setdefault(eid, []).append(s)

        return {
            "total_sessions": len(sessions),
            "by_exercise": {
                eid: {
                    "sessions": sess_list,
                    "total_reps": sum(s["total_reps"] for s in sess_list),
                    "avg_form_score": round(
                        sum(s["avg_form_score"] for s in sess_list) / len(sess_list), 1
                    ) if sess_list else 0,
                }
                for eid, sess_list in by_exercise.items()
            },
        }
