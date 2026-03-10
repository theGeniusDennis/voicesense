"""
tts.py — Text-to-Speech module using gTTS
Converts text to speech and plays it via winsound (built-in, no external window).
Uses ffmpeg to convert MP3 to WAV since winsound only supports WAV.
"""

from gtts import gTTS
from pathlib import Path
import subprocess
import winsound

_MP3_PATH = Path(__file__).parent / "tts_output.mp3"
_WAV_PATH = Path(__file__).parent / "tts_output.wav"


def speak(text: str, lang: str = "en", slow: bool = False) -> None:
    """Convert text to speech and play it silently."""
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(str(_MP3_PATH))

    # Convert MP3 to WAV using ffmpeg (already installed)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(_MP3_PATH), str(_WAV_PATH)],
        capture_output=True
    )

    # Play WAV using built-in winsound — no external window
    winsound.PlaySound(str(_WAV_PATH), winsound.SND_FILENAME)


def speak_question(question: str) -> None:
    speak(f"Question: {question}")


def speak_correct() -> None:
    speak("That's correct! Well done.")


def speak_incorrect(correct_answer: str) -> None:
    speak(f"Not quite. The correct answer is: {correct_answer}")


def speak_session_start() -> None:
    speak("Welcome to VoiceSense. Let's begin your quiz.")


def speak_session_end(score: int, total: int) -> None:
    speak(f"Quiz complete! You got {score} out of {total} correct. Great effort!")
