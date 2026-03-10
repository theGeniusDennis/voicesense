"""
stt.py — Speech-to-Text module using OpenAI Whisper
Accepts audio from the browser (via st.audio_input) and transcribes it.
No microphone access on the server — all recording happens in the browser.
"""

import whisper
import tempfile
import os
import time
import numpy as np


def load_model(model_size: str = "base") -> whisper.Whisper:
    """
    Load the Whisper ASR model.
    model_size options: 'tiny', 'base', 'small', 'medium', 'large'
    Use 'base' for a good speed/accuracy balance.
    """
    return whisper.load_model(model_size)


def transcribe_upload(model: whisper.Whisper, audio_file) -> dict:
    """
    Transcribe audio from a Streamlit UploadedFile (returned by st.audio_input).
    The browser sends audio in WebM/OGG format — ffmpeg handles the conversion.

    Returns dict with 'text' and 'duration' keys.
    """
    start = time.time()

    # Save browser audio to a temp file for Whisper to read
    suffix = ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, fp16=False)
        return {
            "text": result["text"].strip(),
            "duration": round(time.time() - start, 2),
        }
    finally:
        os.unlink(tmp_path)


def transcribe(model: whisper.Whisper, audio: np.ndarray, sample_rate: int = 16000) -> dict:
    """
    Transcribe a numpy audio array using Whisper.
    Kept for local/testing use.
    """
    from scipy.io.wavfile import write

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        write(tmp_path, sample_rate, (audio * 32767).astype(np.int16))

    try:
        result = model.transcribe(tmp_path, fp16=False)
        return {
            "text": result["text"].strip(),
            "duration": 0.0,
        }
    finally:
        os.unlink(tmp_path)
