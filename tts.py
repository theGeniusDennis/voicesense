"""
tts.py — Text-to-Speech module using gTTS
Returns audio as bytes — playback handled by st.audio() in the browser.
No server-side audio playback, works on any cloud platform.
"""

from gtts import gTTS
import io


def _get_bytes(text: str, lang: str = "en", slow: bool = False) -> bytes:
    """Generate TTS audio and return as MP3 bytes."""
    tts = gTTS(text=text, lang=lang, slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def get_question_audio(question: str) -> bytes:
    return _get_bytes(f"Question. {question}")


def get_correct_audio() -> bytes:
    return _get_bytes("That's correct! Well done.")


def get_incorrect_audio(correct_answer: str) -> bytes:
    return _get_bytes(f"Not quite. The correct answer is: {correct_answer}")


def get_session_start_audio() -> bytes:
    return _get_bytes("Welcome to VoiceSense. Let's begin your quiz.")


def get_session_end_audio(score: int, total: int) -> bytes:
    return _get_bytes(f"Quiz complete! You got {score} out of {total} correct. Great effort!")
