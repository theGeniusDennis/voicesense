"""
classifier.py — Answer Matching Module
Uses TF-IDF cosine similarity for flexible answer evaluation.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


SIMILARITY_THRESHOLD = 0.6  # Minimum cosine similarity to count as correct


def load_questions(csv_path: str) -> pd.DataFrame:
    """Load quiz questions from CSV."""
    return pd.read_csv(csv_path)


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for consistent comparison."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _get_accepted_answers(row: pd.Series) -> list[str]:
    """Return all accepted answers as a flat list (pipe-separated in CSV)."""
    return [_normalize(a) for a in str(row["correct_answer"]).split("|") if a.strip()]


def evaluate_answer(transcript: str, row: pd.Series) -> dict:
    """
    Evaluate a student's spoken answer against the correct answers.

    Uses two strategies:
    1. Exact / substring match — handles short numeric answers well
    2. TF-IDF cosine similarity — handles paraphrasing and longer answers

    Returns:
        dict with keys:
            'correct' (bool)
            'matched_answer' (str | None)
            'score' (float) — similarity score (1.0 for exact match)
    """
    normalized_transcript = _normalize(transcript)
    accepted = _get_accepted_answers(row)

    # Strategy 1: exact / substring match
    for answer in accepted:
        if answer == normalized_transcript or answer in normalized_transcript:
            return {"correct": True, "matched_answer": answer, "score": 1.0}

    # Strategy 2: TF-IDF cosine similarity
    if not accepted:
        return {"correct": False, "matched_answer": None, "score": 0.0}

    corpus = [normalized_transcript] + accepted
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        best_idx = scores.argmax()
        best_score = float(scores[best_idx])

        if best_score >= SIMILARITY_THRESHOLD:
            return {"correct": True, "matched_answer": accepted[best_idx], "score": best_score}
    except ValueError:
        pass  # TF-IDF can fail on very short single-token inputs — handled by exact match above

    return {"correct": False, "matched_answer": None, "score": 0.0}
