"""
evaluation/session_log.py — HCI Session Logger
Logs each question-answer interaction to CSV for research analysis.
"""

import csv
import time
from datetime import datetime
from pathlib import Path


LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class SessionLogger:
    """
    Logs student interactions with VoiceSense for HCI evaluation.
    Each row corresponds to one question-answer exchange.
    """

    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = LOG_DIR / f"session_{timestamp}.csv"
        self.session_start = time.time()
        self.rows: list[dict] = []
        self._write_header()

    def _write_header(self) -> None:
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames())
            writer.writeheader()

    @staticmethod
    def _fieldnames() -> list[str]:
        return [
            "timestamp",
            "question",
            "subject",
            "level",
            "transcript",
            "correct",
            "matched_answer",
            "similarity_score",
            "response_time_s",
        ]

    def log(
        self,
        question: str,
        subject: str,
        level: str,
        transcript: str,
        correct: bool,
        matched_answer: str | None,
        similarity_score: float,
        response_time_s: float,
    ) -> None:
        """Append one interaction row to the session log."""
        row = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "subject": subject,
            "level": level,
            "transcript": transcript,
            "correct": correct,
            "matched_answer": matched_answer or "",
            "similarity_score": round(similarity_score, 3),
            "response_time_s": round(response_time_s, 2),
        }
        self.rows.append(row)

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames())
            writer.writerow(row)

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame(self.rows)

    def summary(self) -> dict:
        """Generate a session summary report."""
        if not self.rows:
            return {"total": 0, "correct": 0, "score_pct": 0, "avg_response_time": 0}

        total = len(self.rows)
        correct = sum(1 for r in self.rows if r["correct"])
        avg_time = sum(r["response_time_s"] for r in self.rows) / total
        missed = [r["question"] for r in self.rows if not r["correct"]]

        return {
            "total": total,
            "correct": correct,
            "score_pct": round(correct / total * 100, 1),
            "avg_response_time": round(avg_time, 2),
            "missed_questions": missed,
            "log_path": str(self.log_path),
            "session_duration_s": round(time.time() - self.session_start, 1),
        }
