"""
app.py — VoiceSense Streamlit Application
A voice-driven quiz assistant for Ghanaian primary/JHS students.
Browser-based recording via st.audio_input — works on Streamlit Cloud.
"""

import streamlit as st
import pandas as pd
import time
from pathlib import Path

from classifier import load_questions, evaluate_answer
from tts import get_question_audio, get_correct_audio, get_incorrect_audio, get_session_start_audio, get_session_end_audio
from stt import load_model, transcribe_upload
from evaluation.session_log import SessionLogger


DATA_PATH = Path(__file__).parent / "data" / "questions.csv"


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceSense",
    page_icon="🎙️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f4f6f9; }

    .vs-header {
        background: #0a3d2e;
        padding: 2.2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 2rem;
    }
    .vs-header-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #4caf87;
        margin-bottom: 0.4rem;
    }
    .vs-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .vs-header p { font-size: 0.875rem; color: #a0bdb2; margin: 0; }

    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.8rem 2rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        margin-bottom: 1.2rem;
    }

    .welcome-title { font-size: 1.5rem; font-weight: 700; color: #0a3d2e; margin-bottom: 0.5rem; }
    .welcome-sub { font-size: 0.95rem; color: #666; line-height: 1.6; margin-bottom: 1.4rem; }
    .steps-row { display: flex; gap: 0.8rem; margin-top: 1rem; }
    .step-box { flex: 1; background: #f4f6f9; border-radius: 10px; padding: 1rem; text-align: center; }
    .step-num {
        display: inline-block; width: 26px; height: 26px; line-height: 26px;
        background: #0a3d2e; color: white; border-radius: 50%;
        font-size: 0.75rem; font-weight: 700; margin-bottom: 0.5rem;
    }
    .step-text { font-size: 0.8rem; color: #444; font-weight: 500; }

    .q-meta { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
    .tag { font-size: 0.72rem; font-weight: 600; padding: 0.25rem 0.7rem; border-radius: 20px; letter-spacing: 0.3px; }
    .tag-subject { background: #e6f4ee; color: #0a3d2e; }
    .tag-level   { background: #eef2ff; color: #3a49c4; }
    .q-text { font-size: 1.35rem; font-weight: 700; color: #111827; line-height: 1.45; }

    .result-box { border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
    .result-correct   { background: #edfdf5; border: 1.5px solid #34d399; }
    .result-incorrect { background: #fff1f1; border: 1.5px solid #f87171; }
    .result-title  { font-size: 1rem; font-weight: 700; color: #111; margin-bottom: 0.25rem; }
    .result-detail { font-size: 0.82rem; color: #666; }

    .transcript-card {
        background: #f0faf5; border: 1.5px solid #34d399;
        border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem;
    }
    .transcript-label {
        font-size: 0.75rem; font-weight: 600; color: #059669;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.3rem;
    }
    .transcript-text { font-size: 1.1rem; font-weight: 600; color: #111; }

    .summary-hero { text-align: center; padding: 1rem 0 1.5rem; }
    .summary-score { font-size: 4rem; font-weight: 800; color: #0a3d2e; line-height: 1; }
    .summary-denom { font-size: 1.5rem; color: #aaa; font-weight: 500; }
    .summary-label { font-size: 0.85rem; color: #888; margin-top: 0.3rem; }

    .progress-label { font-size: 0.8rem; color: #888; text-align: right; margin-bottom: 0.3rem; }

    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "questions": None,
        "current_idx": 0,
        "logger": None,
        "whisper_model": None,
        "quiz_active": False,
        "last_result": None,
        "pending_transcript": None,
        "pending_response_time": 0.0,
        "recording_key": 0,
        "audio_to_play": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ASR model...")
def get_whisper_model():
    return load_model("tiny")

@st.cache_data
def get_questions():
    return load_questions(str(DATA_PATH))


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vs-header">
    <div class="vs-header-label">University of Ghana &nbsp;·&nbsp; HCI Lab</div>
    <h1>VoiceSense</h1>
    <p>A voice-driven quiz assistant for Ghanaian primary and JHS students</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    subject = st.selectbox("Subject", ["All", "Math", "English", "Science", "Social Studies", "Art"])
    level = st.selectbox("Level", ["All", "Primary 2", "Primary 3", "Primary 4", "Primary 5", "Primary 6", "JHS 1", "JHS 2"])
    shuffle = st.checkbox("Shuffle questions", value=True)
    st.divider()
    st.caption("VoiceSense is a research prototype exploring voice-based quiz interfaces for Ghanaian learners.")


# ── Quiz setup ────────────────────────────────────────────────────────────────
def prepare_quiz():
    df = get_questions()
    if subject != "All":
        df = df[df["subject"] == subject]
    if level != "All":
        df = df[df["level"] == level]
    if df.empty:
        st.error("No questions found for these filters. Adjust the settings.")
        return
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    st.session_state.questions = df
    st.session_state.current_idx = 0
    st.session_state.logger = SessionLogger()
    st.session_state.whisper_model = get_whisper_model()
    st.session_state.quiz_active = True
    st.session_state.last_result = None
    st.session_state.pending_transcript = None
    st.session_state.recording_key = 0
    st.session_state.audio_to_play = get_session_start_audio()


# ── Welcome screen ────────────────────────────────────────────────────────────
if not st.session_state.quiz_active:
    st.markdown("""
    <div class="card">
        <div class="welcome-title">Welcome</div>
        <div class="welcome-sub">
            Answer curriculum questions by speaking into your microphone.
            Record your answer using the mic widget, then click <strong>Submit Answer</strong>.
        </div>
        <div class="steps-row">
            <div class="step-box"><div class="step-num">1</div><div class="step-text">Read the question</div></div>
            <div class="step-box"><div class="step-num">2</div><div class="step-text">Record your answer</div></div>
            <div class="step-box"><div class="step-num">3</div><div class="step-text">Receive instant feedback</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start Quiz", type="primary", use_container_width=True):
        prepare_quiz()
        st.rerun()
    st.stop()


# ── Queued audio playback ─────────────────────────────────────────────────────
if st.session_state.audio_to_play:
    st.audio(st.session_state.audio_to_play, format="audio/mp3", autoplay=True)
    st.session_state.audio_to_play = None


# ── Active quiz ───────────────────────────────────────────────────────────────
questions: pd.DataFrame = st.session_state.questions
idx: int = st.session_state.current_idx
total = len(questions)

# ── Quiz complete ─────────────────────────────────────────────────────────────
if idx >= total:
    summary = st.session_state.logger.summary()
    st.audio(get_session_end_audio(summary["correct"], summary["total"]), format="audio/mp3", autoplay=True)

    st.markdown(f"""
    <div class="card">
        <div class="summary-hero">
            <div class="summary-score">{summary['correct']}<span class="summary-denom">/{summary['total']}</span></div>
            <div class="summary-label">Questions answered correctly</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{summary['score_pct']}%")
    col2.metric("Avg Response Time", f"{summary['avg_response_time']}s")
    col3.metric("Session Duration", f"{summary['session_duration_s']}s")

    if summary["missed_questions"]:
        st.markdown("#### Questions to Review")
        for q in summary["missed_questions"]:
            st.markdown(f"- {q}")

    session_df = st.session_state.logger.to_dataframe()
    if len(session_df) >= 2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Accuracy by subject (%)**")
            st.bar_chart(session_df.groupby("subject")["correct"].mean() * 100)
        with col_b:
            st.markdown("**Response time per question (s)**")
            st.line_chart(session_df["response_time_s"].reset_index(drop=True))

    with open(summary["log_path"], "rb") as f:
        st.download_button("Download Session Log (CSV)", f, file_name="voicesense_session.csv", mime="text/csv", use_container_width=True)

    if st.button("Start New Quiz", type="primary", use_container_width=True):
        st.session_state.quiz_active = False
        st.rerun()
    st.stop()


# ── Current question ──────────────────────────────────────────────────────────
row = questions.iloc[idx]

st.markdown(f'<div class="progress-label">Question {idx + 1} of {total}</div>', unsafe_allow_html=True)
st.progress(idx / total)

st.markdown(f"""
<div class="card">
    <div class="q-meta">
        <span class="tag tag-subject">{row['subject']}</span>
        <span class="tag tag-level">{row['level']}</span>
    </div>
    <div class="q-text">{row['question']}</div>
</div>
""", unsafe_allow_html=True)

if st.button("Read question aloud"):
    st.audio(get_question_audio(row["question"]), format="audio/mp3", autoplay=True)

st.divider()

# ── Previous result feedback ──────────────────────────────────────────────────
if st.session_state.last_result:
    result = st.session_state.last_result
    if result["correct"]:
        st.markdown(f"""
        <div class="result-box result-correct">
            <div class="result-title">Correct</div>
            <div class="result-detail">You said: "{result['transcript']}" &nbsp;&middot;&nbsp; Confidence: {result['score']:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        correct_display = str(row["correct_answer"]).split("|")[0]
        st.markdown(f"""
        <div class="result-box result-incorrect">
            <div class="result-title">Incorrect</div>
            <div class="result-detail">You said: "{result['transcript']}" &nbsp;&middot;&nbsp; Answer: <strong>{correct_display}</strong></div>
        </div>
        """, unsafe_allow_html=True)
    st.session_state.last_result = None

# ── Voice input ───────────────────────────────────────────────────────────────
st.markdown("**Speak your answer:**")

# ── State A: transcript pending confirmation ──────────────────────────────────
if st.session_state.pending_transcript is not None:
    st.markdown(f"""
    <div class="transcript-card">
        <div class="transcript-label">Whisper heard</div>
        <div class="transcript-text">"{st.session_state.pending_transcript}"</div>
    </div>
    """, unsafe_allow_html=True)

    col_submit, col_rerecord, col_skip = st.columns([3, 2, 1])

    with col_submit:
        if st.button("Submit Answer", type="primary", use_container_width=True):
            transcript = st.session_state.pending_transcript
            response_time = st.session_state.pending_response_time

            result = evaluate_answer(transcript, row)
            result["transcript"] = transcript

            st.session_state.logger.log(
                question=row["question"],
                subject=row["subject"],
                level=row["level"],
                transcript=transcript,
                correct=result["correct"],
                matched_answer=result["matched_answer"],
                similarity_score=result["score"],
                response_time_s=response_time,
            )

            if result["correct"]:
                st.session_state.audio_to_play = get_correct_audio()
            else:
                correct_display = str(row["correct_answer"]).split("|")[0]
                st.session_state.audio_to_play = get_incorrect_audio(correct_display)

            st.session_state.last_result = result
            st.session_state.pending_transcript = None
            st.session_state.pending_response_time = 0.0
            st.session_state.recording_key += 1
            st.session_state.current_idx += 1
            st.rerun()

    with col_rerecord:
        if st.button("Re-record", use_container_width=True):
            st.session_state.pending_transcript = None
            st.session_state.pending_response_time = 0.0
            st.session_state.recording_key += 1
            st.rerun()

    with col_skip:
        if st.button("Skip", use_container_width=True):
            st.session_state.logger.log(
                question=row["question"], subject=row["subject"], level=row["level"],
                transcript="[skipped]", correct=False, matched_answer=None,
                similarity_score=0.0, response_time_s=0.0,
            )
            st.session_state.pending_transcript = None
            st.session_state.recording_key += 1
            st.session_state.current_idx += 1
            st.rerun()

# ── State B: waiting to record ────────────────────────────────────────────────
else:
    col_mic, col_skip = st.columns([4, 1])

    with col_mic:
        audio_input = st.audio_input(
            "Click the mic to record your answer",
            key=f"mic_{st.session_state.recording_key}"
        )

        if audio_input is not None:
            start = time.time()
            with st.spinner("Transcribing..."):
                try:
                    result_raw = transcribe_upload(st.session_state.whisper_model, audio_input)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    st.stop()

            transcript = result_raw["text"]
            if not transcript:
                st.warning("No speech detected. Try recording again.")
                st.stop()

            st.session_state.pending_transcript = transcript
            st.session_state.pending_response_time = round(time.time() - start, 2)
            st.rerun()

    with col_skip:
        if st.button("Skip", use_container_width=True):
            st.session_state.logger.log(
                question=row["question"], subject=row["subject"], level=row["level"],
                transcript="[skipped]", correct=False, matched_answer=None,
                similarity_score=0.0, response_time_s=0.0,
            )
            st.session_state.recording_key += 1
            st.session_state.current_idx += 1
            st.rerun()
