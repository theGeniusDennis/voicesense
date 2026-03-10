# VoiceSense — Product Requirements Document

## 1. Project Overview

**VoiceSense** is a voice-driven quiz assistant designed for Ghanaian primary and JHS students. It uses Automatic Speech Recognition (ASR) to capture spoken answers, NLP to evaluate correctness, and Text-to-Speech (TTS) to deliver feedback — creating a full speech-in, speech-out interaction loop.

**Research Question:**
> How usable and effective is a voice-based quiz assistant for students in a Ghanaian classroom setting, and what interaction challenges arise from local speech patterns and bilingual (English/Twi) responses?

**Purpose:**
- Demonstrate a working ASR + NLP + TTS pipeline relevant to the UG HCI Lab's research track
- Provide a platform for studying how students interact with voice AI in a local educational context
- Contribute to the underexplored area of voice interfaces for Ghanaian learners

---

## 2. Objectives

1. Build a real **ASR pipeline** using OpenAI Whisper for speech-to-text transcription
2. Implement a **flexible answer matcher** using TF-IDF cosine similarity (handles paraphrasing, partial answers)
3. Support **bilingual responses** — accept correct answers in both English and Twi
4. Deliver **spoken feedback** via gTTS text-to-speech
5. Log **HCI interaction data** (transcripts, response times, accuracy) for research evaluation
6. Provide a **Streamlit UI** for live demo and testing

---

## 3. Scope

### In-Scope
- Voice input via microphone (Whisper STT)
- Ghanaian primary/JHS curriculum quiz content (Math, English, Science, Social Studies)
- Bilingual answer matching: English + Twi
- Spoken + visual feedback
- Session logging for HCI evaluation (CSV output)
- Streamlit web interface

### Out-of-Scope (Future Work)
- Full Twi/Ga/Ewe speech recognition (requires low-resource ASR models — active research area)
- Deep learning answer evaluation (BERT, transformers)
- Multi-user classroom management
- Production deployment

---

## 4. Target Users

- **Primary audience:** Primary 4–JHS 2 students in Ghana
- **Secondary audience:** HCI researchers studying voice interaction in African educational contexts
- **Tertiary audience:** Teachers evaluating AI-assisted quiz tools

---

## 5. Functional Requirements

| ID  | Requirement            | Description                                                             |
|-----|------------------------|-------------------------------------------------------------------------|
| FR1 | Voice Input            | Capture student speech via microphone; transcribe using Whisper ASR     |
| FR2 | Question Display       | Show current question on screen and speak it aloud via TTS              |
| FR3 | Answer Matching        | Compare transcript to correct answer using TF-IDF cosine similarity     |
| FR4 | Bilingual Support      | Accept correct answers in English or Twi for applicable questions       |
| FR5 | Spoken Feedback        | Speak "Correct!" or "Not quite — the answer is X" after each response  |
| FR6 | Session Logging        | Log question, transcript, correctness, response time per session        |
| FR7 | Dataset Management     | Load questions from CSV; easily extendable with new questions           |
| FR8 | Accuracy Report        | Display session summary: score, avg response time, common errors        |

---

## 6. Non-Functional Requirements

| ID   | Requirement     | Description                                                        |
|------|-----------------|--------------------------------------------------------------------|
| NFR1 | Latency         | Transcription + matching response within 3 seconds                 |
| NFR2 | Usability       | Interface operable by a Primary 5 student with minimal instruction |
| NFR3 | Portability     | Runs on any system with Python 3.8+ and internet (for gTTS)       |
| NFR4 | Modularity      | STT, TTS, classifier, and logger are independent swappable modules |
| NFR5 | Reproducibility | All experiments reproducible from README instructions              |

---

## 7. Technical Stack

| Component      | Technology                          |
|----------------|-------------------------------------|
| ASR (STT)      | `openai-whisper` (local inference)  |
| TTS            | `gTTS` + `playsound`                |
| Answer Matching| `scikit-learn` (TF-IDF + cosine)    |
| Audio Capture  | `sounddevice` + `scipy`             |
| UI             | `Streamlit`                         |
| Data           | `pandas` + CSV                      |
| Version Control| Git / GitHub                        |

---

## 8. Dataset

Stored in `data/questions.csv`. 20–30 questions covering Ghanaian primary/JHS curriculum.

| Field           | Description                                         |
|-----------------|-----------------------------------------------------|
| `question`      | The quiz question (displayed and spoken)            |
| `correct_answer`| Accepted English answers, separated by `\|`          |
| `subject`       | Math / English / Science / Social Studies           |
| `level`         | Primary 3 – JHS 2                                   |
| `twi_answers`   | Accepted Twi answers where applicable               |

**Sample rows:**

| question | correct_answer | subject | level | twi_answers |
|---|---|---|---|---|
| What is 7 multiplied by 8? | 56 | Math | Primary 5 | aduoson |
| Name the capital of Ghana. | Accra | Social Studies | Primary 4 | |
| How many sides does a triangle have? | 3\|three | Math | Primary 3 | mmiensa |
| What gas do plants need for photosynthesis? | carbon dioxide\|CO2 | Science | JHS 1 | |

---

## 9. System Architecture

```
[Microphone] → [stt.py: Whisper ASR] → transcript
                                            ↓
                              [classifier.py: TF-IDF matcher]
                              checks English + Twi answers
                                            ↓
                           correct / incorrect + right answer
                                            ↓
                              [tts.py: gTTS spoken feedback]
                                            ↓
                    [app.py: Streamlit UI — question, transcript, result]
                                            ↓
                     [evaluation/session_log.py: logs row to CSV]
```

---

## 10. HCI Evaluation Design

The session logger is the research backbone of this project. Each interaction logs:

| Field            | Purpose                                   |
|------------------|-------------------------------------------|
| `question`       | What was asked                            |
| `transcript`     | What Whisper heard                        |
| `correct`        | Boolean — was the answer right?           |
| `response_time`  | Seconds from question display to answer   |
| `language_used`  | "english" or "twi" (detected from match) |

After a session, a summary report shows:
- Total score (X/N correct)
- Average response time
- Questions most frequently answered incorrectly
- Language breakdown (English vs Twi responses)

This mirrors real HCI usability study methodology.

---

## 11. Milestones & Timeline

| # | Milestone | Description | Est. Time |
|---|-----------|-------------|-----------|
| M1 | Environment setup | Python env, dependencies installed | 0.5 day |
| M2 | Dataset | 20–30 questions CSV with Twi answers | 1 day |
| M3 | STT module | Whisper integration + audio recording | 1 day |
| M4 | TTS module | gTTS spoken feedback | 0.5 day |
| M5 | Classifier | TF-IDF answer matcher, bilingual | 1 day |
| M6 | Session logger | CSV logging + session report | 0.5 day |
| M7 | Streamlit UI | Full app integration | 1 day |
| M8 | README + docs | Research framing, setup guide | 0.5 day |

**Total:** ~6 days for working research prototype

---

## 12. Deliverables

- `app.py` — Streamlit application
- `stt.py` — Whisper ASR module
- `tts.py` — gTTS TTS module
- `classifier.py` — bilingual answer matcher
- `data/questions.csv` — quiz dataset
- `evaluation/session_log.py` — HCI session logger
- `requirements.txt` — dependencies
- `README.md` — setup, research framing, demo instructions
- Sample session log CSV (demonstrating the evaluation pipeline)

---

## 13. Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| Whisper slow on low-end hardware | Use `tiny` or `base` model; note latency in evaluation |
| gTTS requires internet | Cache audio files; note offline limitation |
| Twi answer coverage is limited | Start with numeric/basic vocabulary; flag as research gap |
| STT errors on accented speech | Log all transcripts — this IS the research finding |

---

## 14. Research Contribution

This project contributes to the UG HCI Lab's ASR research track by:

1. **Demonstrating a full ASR+TTS pipeline** applied to a Ghanaian educational context
2. **Generating interaction data** on how students respond to voice AI (error rates, response times)
3. **Opening the bilingual question** — when do students code-switch to Twi when answering a voice quiz?
4. **Identifying ASR failure modes** on Ghanaian-accented English — a gap in existing literature
5. **Providing an extensible platform** for future integration of low-resource Twi ASR models
