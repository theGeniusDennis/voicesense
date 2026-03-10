# VoiceSense

**A voice-driven quiz assistant for Ghanaian primary and JHS students.**

Built as a research prototype for the University of Ghana HCI Lab — ASR research track.

---

## Research Question

> How usable and effective is a voice-based quiz assistant for students in a Ghanaian classroom setting, and what interaction challenges arise from local speech patterns and bilingual (English/Twi) responses?

---

## What It Does

1. **Speaks a quiz question** aloud (gTTS text-to-speech)
2. **Records the student's spoken answer** via microphone
3. **Transcribes the answer** using OpenAI Whisper ASR (runs locally, no API key needed)
4. **Evaluates the answer** — accepts English or Twi responses using TF-IDF cosine similarity
5. **Gives spoken feedback** — "Correct!" or "The answer was X"
6. **Logs the interaction** for HCI research analysis (transcript, response time, language used)

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `openai-whisper` also requires [ffmpeg](https://ffmpeg.org/download.html) to be installed on your system.
>
> Windows: `choco install ffmpeg` or download from the website.
> macOS: `brew install ffmpeg`
> Linux: `sudo apt install ffmpeg`

### 2. Run the app

```bash
streamlit run app.py
```

---

## Project Structure

```
voiceSense/
├── app.py                  # Streamlit main application
├── stt.py                  # Whisper ASR module (speech-to-text)
├── tts.py                  # gTTS module (text-to-speech)
├── classifier.py           # Bilingual answer matcher (TF-IDF + cosine)
├── data/
│   └── questions.csv       # Quiz dataset (30 Ghanaian curriculum questions)
├── evaluation/
│   ├── session_log.py      # HCI session logger
│   └── logs/               # Session CSV logs (auto-generated)
├── requirements.txt
└── README.md
```

---

## Dataset

`data/questions.csv` contains 30 questions across Math, English, Science, and Social Studies (Primary 2–JHS 2). Each question has:
- `correct_answer` — accepted English answers (pipe-separated for multiple options)
- `twi_answers` — accepted Twi answers where applicable

To add questions, simply add rows to the CSV.

---

## HCI Evaluation

After each session, a CSV log is saved to `evaluation/logs/session_YYYYMMDD_HHMMSS.csv`.

Each row captures:
| Field | Description |
|---|---|
| `question` | The quiz question |
| `transcript` | What Whisper heard |
| `correct` | Whether the answer was accepted |
| `matched_answer` | Which accepted answer was matched |
| `similarity_score` | TF-IDF cosine similarity (0–1) |
| `language_used` | `english` or `twi` |
| `response_time_s` | Seconds from question to response |

The session summary screen shows score, accuracy, language breakdown, and missed questions.

---

## Whisper Model Selection

In `stt.py`, change the `model_size` to trade off speed vs. accuracy:

| Model | Speed | Accuracy | Notes |
|-------|-------|----------|-------|
| `tiny` | Fastest | Lower | Good for demos on low-end hardware |
| `base` | Fast | Good | **Default — recommended** |
| `small` | Moderate | Better | Good for research accuracy |

---

## Alignment with UG HCI Lab Research

This project directly addresses the lab's **ASR research track**:

- **Full ASR + TTS pipeline** — not a simulation; real speech in, speech out
- **Ghanaian educational context** — locally relevant curriculum content
- **Bilingual interaction** — studies code-switching between English and Twi
- **Interaction data collection** — mirrors real HCI usability study methodology
- **Extensibility** — architecture ready to swap Whisper for a low-resource Twi ASR model

---

## Future Research Directions

- Integrate a **Twi-specific ASR model** (e.g., fine-tuned Whisper on Twi data)
- Conduct a **formal usability study** with primary school students
- Expand to **Ga and Ewe** languages
- Analyse **ASR failure modes** on Ghanaian-accented English
- Deploy via **Streamlit Cloud** for broader access
