"""
stt.py — Speech-to-Text module using OpenAI Whisper
Records audio from the microphone and transcribes it to text.
"""

import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
import os
import time

SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
RECORD_SECONDS = 5   # Default recording duration


def load_model(model_size: str = "base") -> whisper.Whisper:
    """
    Load the Whisper ASR model.
    model_size options: 'tiny', 'base', 'small', 'medium', 'large'
    Use 'tiny' for fastest inference on low-end hardware.
    Use 'base' for a good speed/accuracy balance.
    """
    return whisper.load_model(model_size)


def record_audio(duration: int = RECORD_SECONDS, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Record audio from the default microphone.
    Returns a numpy array of audio samples.
    """
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    sd.wait()  # Block until recording is complete
    return audio.flatten()


def start_recording(duration: int = RECORD_SECONDS, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Start a non-blocking recording. Returns the audio array (filled asynchronously).
    Call finish_recording() after your countdown to wait for completion.
    """
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    return audio


def finish_recording(audio: np.ndarray) -> np.ndarray:
    """Wait for the recording started by start_recording() to complete."""
    sd.wait()
    return audio.flatten()


def transcribe(model: whisper.Whisper, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """
    Transcribe a numpy audio array using Whisper.
    Returns a dict with 'text' and 'language' keys.
    Whisper auto-detects language — useful for bilingual English/Twi responses.
    """
    # Save audio to a temp WAV file (Whisper reads from file)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        write(tmp_path, sample_rate, (audio * 32767).astype(np.int16))

    try:
        result = model.transcribe(tmp_path, fp16=False)
        return {
            "text": result["text"].strip(),
            "language": result.get("language", "en")
        }
    finally:
        os.unlink(tmp_path)


def record_and_transcribe(model: whisper.Whisper, duration: int = RECORD_SECONDS) -> dict:
    """
    Convenience function: record audio then immediately transcribe it.
    Returns dict with 'text', 'language', and 'duration' keys.
    """
    start = time.time()
    audio = record_audio(duration=duration)
    result = transcribe(model, audio)
    result["duration"] = round(time.time() - start, 2)
    return result
