"""AI Question Generator — Streamlit Application.

Main entry point for the web interface.  Provides three tabs:
  • Generate Questions  – paste or upload text, configure settings, run MCQ pipeline
  • Analytics          – visualise Bloom's level distribution and question statistics
  • Export             – download generated MCQs as CSV, Excel, or PDF
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

# Make src importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

from src.bloom_classifier.classifier import BloomTaxonomyClassifier
from src.mcq_generator.generator import MCQGenerator
from src.short_answer_generator.short_answer_generator import ShortAnswerGenerator
from src.mcq_generator.true_false_generator import TrueFalseGenerator
from src.retriever.rag_retriever import RAGRetriever
from src.utils.export_utils import export_to_csv, export_to_excel, export_to_pdf
from src.utils.text_preprocessing import clean_text


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_TEXT_LENGTH = 100   # characters
APP_TITLE       = "AI Question Generator"

BLOOM_COLOURS: dict[str, str] = {
    "Remember":   "#4CAF50",
    "Understand": "#2196F3",
    "Apply":      "#FF9800",
    "Analyze":    "#9C27B0",
    "Evaluate":   "#F44336",
    "Create":     "#00BCD4",
}

DIFFICULTY_COLOURS: dict[str, str] = {
    "Easy":   "#4CAF50",
    "Medium": "#FF9800",
    "Hard":   "#F44336",
}

# ---------------------------------------------------------------------------
# Page configuration  (must be the FIRST Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Global ── */
    .main { background-color: #f8f9fa; }

    /* ── Question card ── */
    .question-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 18px;
        border-left: 5px solid #2E4057;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .question-text {
        font-size: 16px;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 12px;
        line-height: 1.5;
    }
    .option-item {
        background: #f4f6f8;
        border-radius: 6px;
        padding: 8px 14px;
        margin: 4px 0;
        font-size: 14px;
        color: #333333;
        border: 1px solid #e0e4ea;
    }
    .option-item:hover { background: #e8edf3; }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        color: #ffffff;
        margin-right: 6px;
        vertical-align: middle;
    }

    /* ── Answer reveal ── */
    .answer-box {
        background: #e8f5e9;
        border: 1px solid #a5d6a7;
        border-radius: 8px;
        padding: 10px 16px;
        color: #1b5e20;
        font-weight: 600;
        font-size: 14px;
    }

    /* ── Stat card ── */
    .stat-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .stat-number {
        font-size: 36px;
        font-weight: 700;
        color: #2E4057;
    }
    .stat-label {
        font-size: 13px;
        color: #666666;
        margin-top: 4px;
    }

    /* ── Sidebar ── */
    .css-1d391kg { background-color: #1a1a2e !important; }
    
    /* ── CRITICAL FIX: Radio buttons - FORCE BLACK TEXT ── */
    .stRadio label {
        color: #000000 !important;
        background-color: white !important;
        padding: 10px !important;
        border-radius: 6px !important;
        border: 1px solid #ddd !important;
        margin: 5px 0 !important;
        display: block !important;
    }
    .stRadio label > div {
        color: #000000 !important;
    }
    .stRadio label > div > div {
        color: #000000 !important;
    }
    .stRadio label span {
        color: #000000 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    /* Force all text inside radio to be black */
    .stRadio * {
        color: #000000 !important;
    }
    
    /* ── Text input and textarea - BLACK text, VISIBLE cursor ── */
    .stTextInput input {
        background-color: white !important;
        color: #000000 !important;
        border: 2px solid #ccc !important;
        padding: 10px !important;
        font-size: 14px !important;
        caret-color: #000000 !important;
    }
    .stTextArea textarea {
        background-color: white !important;
        color: #000000 !important;
        border: 2px solid #ccc !important;
        padding: 10px !important;
        font-size: 14px !important;
        caret-color: #000000 !important;
        line-height: 1.5 !important;
    }
    
    /* ── CRITICAL FIX: Streamlit info/success/warning boxes - DARK TEXT ── */
    div[data-testid="stAlert"] {
        color: #000000 !important;
    }
    div[data-testid="stAlert"] * {
        color: #000000 !important;
    }
    .stSuccess, .stSuccess * {
        color: #155724 !important;
        font-weight: 600 !important;
    }
    .stInfo, .stInfo * {
        color: #004085 !important;
        font-weight: 600 !important;
    }
    .stWarning, .stWarning * {
        color: #856404 !important;
        font-weight: 600 !important;
    }
    .stError, .stError * {
        color: #721c24 !important;
        font-weight: 600 !important;
    }
    
    /* ── Form buttons ── */
    .stButton button {
        font-weight: bold !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model loading with caching
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading MCQ Generator model …")
def load_mcq_generator() -> MCQGenerator | None:
    """Load and cache the MCQ generator model."""
    try:
        return MCQGenerator()
    except Exception as exc:
        logger.error("MCQGenerator failed to load: %s", exc)
        return None


@st.cache_resource(show_spinner="Loading Bloom Classifier model …")
def load_bloom_classifier() -> BloomTaxonomyClassifier | None:
    """Load and cache the Bloom taxonomy classifier."""
    try:
        return BloomTaxonomyClassifier()
    except Exception as exc:
        logger.error("BloomTaxonomyClassifier failed to load: %s", exc)
        return None


@st.cache_resource(show_spinner="Loading RAG Retriever model …")
def load_rag_retriever() -> RAGRetriever | None:
    """Load and cache the RAG retriever."""
    try:
        return RAGRetriever()
    except Exception as exc:
        logger.error("RAGRetriever failed to load: %s", exc)
        return None


@st.cache_resource(show_spinner="Loading True/False Generator …")
def load_tf_generator() -> TrueFalseGenerator | None:
    """Load and cache the True/False question generator."""
    try:
        return TrueFalseGenerator()
    except Exception as exc:
        logger.error("TrueFalseGenerator failed to load: %s", exc)
        return None


@st.cache_resource(show_spinner="Loading Short Answer Generator …")
def load_sa_generator() -> ShortAnswerGenerator | None:
    """Load and cache the Short Answer generator."""
    try:
        return ShortAnswerGenerator()
    except Exception as exc:
        logger.error("ShortAnswerGenerator failed to load: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults: dict = {
        "generated_mcqs":  [],
        "source_text":     "",
        "generation_done": False,
        "error_message":   "",
        # Quiz / interactive mode
        "quiz_mode":       False,
        "quiz_submitted":  False,
        "user_answers":    {},   # {idx: user_answer}
        "question_types":  ["MCQ"],
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# PDF text extraction helper
# ---------------------------------------------------------------------------
def _extract_text_from_pdf(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """Extract plain text from an uploaded PDF file."""
    try:
        import pdfplumber
        text_parts: list[str] = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except Exception as exc:
        st.error(f"Could not read PDF: {exc}")
        return ""


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def _bloom_badge(level: str) -> str:
    colour = BLOOM_COLOURS.get(level, "#888888")
    return f'<span class="badge" style="background:{colour}">🧠 {level}</span>'


def _difficulty_badge(difficulty: str) -> str:
    colour = DIFFICULTY_COLOURS.get(difficulty, "#888888")
    return f'<span class="badge" style="background:{colour}">⚡ {difficulty}</span>'


def _render_mcq_card(index: int, mcq: dict, show_bloom: bool) -> None:
    """Render a single MCQ as a styled card."""
    question   = mcq.get("question", "")
    options    = mcq.get("options", [])
    answer     = mcq.get("answer", "")
    answer_lbl = mcq.get("answer_label", "")
    bloom      = mcq.get("level", "")
    difficulty = mcq.get("difficulty", "")
    confidence = mcq.get("confidence", None)

    badges = ""
    if show_bloom and bloom:
        badges += _bloom_badge(bloom)
    if difficulty:
        badges += _difficulty_badge(difficulty)
    if confidence is not None:
        badges += (
            f'<span class="badge" style="background:#607D8B">'
            f'🎯 {confidence:.0%}</span>'
        )

    options_html = "".join(
        f'<div class="option-item"><b>({label})</b>  {options[i] if i < len(options) else "—"}</div>'
        for i, label in enumerate(["A", "B", "C", "D"])
    )

    st.markdown(
        f"""
        <div class="question-card">
            <div class="question-text">Q{index}. {question}</div>
            <div style="margin-bottom:10px">{badges}</div>
            {options_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Show Answer"):
        st.markdown(
            f'<div class="answer-box">✅ Correct Answer: ({answer_lbl}) {answer}</div>',
            unsafe_allow_html=True,
        )
        if show_bloom and bloom:
            desc = BloomTaxonomyClassifier.LEVEL_DESCRIPTIONS.get(bloom, "")
            if desc:
                st.caption(f"Bloom's level – *{bloom}*: {desc}")


# ---------------------------------------------------------------------------
# Interactive question renderer (for quiz mode)
# ---------------------------------------------------------------------------
def _render_interactive_question(
    idx: int,
    question: dict,
    sa_gen: "ShortAnswerGenerator | None" = None,
) -> None:
    """Render a question interactively with answer input and feedback."""
    qtype      = question.get("type", "MCQ")
    q_text     = question.get("question", "")
    bloom      = question.get("level", "")
    difficulty = question.get("difficulty", "")
    answered   = st.session_state.user_answers.get(idx) is not None

    # Header badges HTML
    badges = ""
    if bloom:
        badges += _bloom_badge(bloom)
    if difficulty:
        badges += _difficulty_badge(difficulty)
    type_colours = {"MCQ": "#3F51B5", "True/False": "#009688", "Short Answer": "#E91E63"}
    type_colour  = type_colours.get(qtype, "#607D8B")
    badges += f'<span class="badge" style="background:{type_colour};color:white;font-weight:bold;">{qtype}</span>'

    st.markdown(
        f"""
        <div class="question-card">
            <div class="question-text">Q{idx}. {q_text}</div>
            <div style="margin-bottom:10px">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    form_key = f"quiz_form_{idx}"

    # ── MCQ ──────────────────────────────────────────────────────────────────
    if qtype == "MCQ":
        options  = question.get("options", [])
        answer   = question.get("answer", "")
        ans_lbl  = question.get("answer_label", "")
        opt_labels = question.get("option_labels", ["A", "B", "C", "D"])

        labelled = [f"({opt_labels[i]}) {opt}" for i, opt in enumerate(options)]
        #st.write("🔍 DEBUG:", labelled)

        with st.form(form_key):
            choice = st.radio("Your answer:", labelled, index=None, key=f"radio_{idx}")
            submitted = st.form_submit_button("Submit Answer", use_container_width=True)

        if submitted and choice is not None:
            st.session_state.user_answers[idx] = choice
            answered = True

        if answered:
            user_choice = st.session_state.user_answers.get(idx, "")
            # Extract label from "(A) ..." pattern
            user_lbl = user_choice[1] if user_choice and len(user_choice) > 1 else ""
            if user_lbl == ans_lbl:
                st.markdown(
                    f"""<div style="background-color:#d4edda;border:2px solid #28a745;border-radius:5px;padding:15px;margin:10px 0;">
                    <strong style="color:#155724;">✅ Correct!</strong><br>
                    <span style="color:#155724;">Answer: ({ans_lbl}) {answer}</span>
                    </div>""",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div style="background-color:#f8d7da;border:2px solid #dc3545;border-radius:5px;padding:15px;margin:10px 0;">
                    <strong style="color:#721c24;">❌ Incorrect</strong><br>
                    <span style="color:#721c24;">Correct answer: ({ans_lbl}) {answer}</span>
                    </div>""",
                    unsafe_allow_html=True
                )

    # ── True/False ───────────────────────────────────────────────────────────
    elif qtype == "True/False":
        correct_answer = question.get("answer", True)
        explanation = question.get("explanation", "")

        with st.form(form_key):
            col1, col2 = st.columns(2)
            true_btn = col1.form_submit_button("✅ True", use_container_width=True)
            false_btn = col2.form_submit_button("❌ False", use_container_width=True)

        if true_btn:
            st.session_state.user_answers[idx] = True
            answered = True
        elif false_btn:
            st.session_state.user_answers[idx] = False
            answered = True

        if answered:
            user_ans = st.session_state.user_answers.get(idx)
            if user_ans == correct_answer:
                st.markdown(
                    f"""<div style="background-color:#d4edda;border:2px solid #28a745;border-radius:5px;padding:15px;margin:10px 0;">
                    <strong style="color:#155724;">✅ Correct!</strong><br>
                    <span style="color:#155724;">{explanation}</span>
                    </div>""",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div style="background-color:#f8d7da;border:2px solid #dc3545;border-radius:5px;padding:15px;margin:10px 0;">
                    <strong style="color:#721c24;">❌ Incorrect</strong><br>
                    <span style="color:#721c24;">Correct answer: {correct_answer}<br>{explanation}</span>
                    </div>""",
                    unsafe_allow_html=True
                )

    # ── Short Answer ─────────────────────────────────────────────────────────
    elif qtype == "Short Answer":
        correct_answer = question.get("answer", "")
        keywords = question.get("keywords", [])
        explanation = question.get("explanation", "")

        with st.form(form_key):
            user_text = st.text_area(
                "Your answer (2-4 sentences):",
                height=150,
                placeholder="Type your answer here...",
                key=f"sa_input_{idx}"
            )
            submitted = st.form_submit_button("Submit Answer", use_container_width=True)

        if submitted and user_text.strip():
            st.session_state.user_answers[idx] = user_text
            answered = True

        if answered:
            user_ans = st.session_state.user_answers.get(idx, "")
            
            # Check answer using the generator
            if sa_gen is not None:
                result = sa_gen.check_answer(user_ans, correct_answer, keywords)
                feedback = result.get("feedback", "")
                is_correct = result.get("is_correct", False)
                
                if is_correct:
                    bg_color = "#d4edda"
                    border_color = "#28a745"
                    text_color = "#155724"
                    icon = "✅"
                elif "Partially" in feedback or "⚠️" in feedback:
                    bg_color = "#fff3cd"
                    border_color = "#ffc107"
                    text_color = "#856404"
                    icon = "⚠️"
                else:
                    bg_color = "#f8d7da"
                    border_color = "#dc3545"
                    text_color = "#721c24"
                    icon = "❌"
                
                st.markdown(
                    f"""<div style="background-color:{bg_color};border:2px solid {border_color};border-radius:5px;padding:15px;margin:10px 0;">
                    <strong style="color:{text_color};">{icon} {feedback}</strong>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Show expected answer
                st.markdown(
                    f"""<div style="background-color:#d1ecf1;border:2px solid #17a2b8;border-radius:5px;padding:15px;margin:10px 0;">
                    <strong style="color:#0c5460;">💡 Expected Answer:</strong><br>
                    <span style="color:#0c5460;">{correct_answer}</span>
                    </div>""",
                    unsafe_allow_html=True
                )


# ---------------------------------------------------------------------------
# Tab 1 – Generate Questions
# ---------------------------------------------------------------------------
def tab_generate(
    mcq_gen: MCQGenerator | None,
    bloom_clf: BloomTaxonomyClassifier | None,
    rag: RAGRetriever | None,
    tf_gen: TrueFalseGenerator | None = None,
    sa_gen: ShortAnswerGenerator | None = None,
) -> None:
    st.subheader("📝 Generate & Practice Questions")

    # ── Input source ─────────────────────────────────────────────────────────
    input_method = st.radio(
        "Input source",
        ["Paste text", "Upload file"],
        horizontal=True,
    )

    source_text = ""

    if input_method == "Paste text":
        source_text = st.text_area(
            "Paste your text here",
            height=220,
            placeholder="Enter the passage or article you want to generate questions from …",
        )
    else:
        uploaded = st.file_uploader(
            "Upload a .txt or .pdf file",
            type=["txt", "pdf"],
        )
        if uploaded is not None:
            if uploaded.type == "application/pdf":
                with st.spinner("Extracting text from PDF …"):
                    source_text = _extract_text_from_pdf(uploaded)
            else:
                source_text = uploaded.read().decode("utf-8", errors="ignore")

            if source_text:
                with st.expander("Preview extracted text"):
                    st.text(source_text[:1200] + (" …" if len(source_text) > 1200 else ""))

    # ── Sidebar settings ─────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        num_questions = st.slider("Number of questions per type", 1, 10, 5)
        difficulty_filter = st.selectbox(
            "Target difficulty",
            ["Any", "Easy", "Medium", "Hard"],
        )
        question_types = st.multiselect(
            "Question Types",
            ["MCQ", "True/False", "Short Answer"],
            default=st.session_state.question_types,
        )
        if question_types:
            st.session_state.question_types = question_types
        use_rag = st.checkbox(
            "Enable RAG (context-aware retrieval)",
            value=False,
            help="Index the text with FAISS before generation for better context selection.",
        )
        auto_bloom = st.checkbox(
            "Auto-classify Bloom's level",
            value=True,
            help="Run the Bloom classifier on each generated question.",
        )
        quiz_mode = st.checkbox(
            "🎯 Interactive Quiz Mode",
            value=st.session_state.quiz_mode,
            help="Answer questions interactively and see your score.",
        )
        st.session_state.quiz_mode = quiz_mode
        st.divider()
        st.caption("Models loaded:")
        st.caption(f"{'✅' if mcq_gen else '❌'} MCQ Generator")
        st.caption(f"{'✅' if bloom_clf else '❌'} Bloom Classifier")
        st.caption(f"{'✅' if rag else '❌'} RAG Retriever")
        st.caption(f"{'✅' if tf_gen else '❌'} True/False Generator")
        st.caption(f"{'✅' if sa_gen else '❌'} Short Answer Generator")

    # ── Generate button ───────────────────────────────────────────────────────
    col_btn, col_clear = st.columns([2, 1])
    with col_btn:
        generate_clicked = st.button("🚀 Generate Questions", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.generated_mcqs  = []
            st.session_state.generation_done = False
            st.session_state.source_text     = ""
            st.session_state.user_answers    = {}
            st.session_state.quiz_submitted  = False
            st.rerun()

    # ── Validation + pipeline ─────────────────────────────────────────────────
    if generate_clicked:
        cleaned = clean_text(source_text) if source_text else ""
        active_types = st.session_state.question_types or ["MCQ"]

        if not cleaned or len(cleaned) < MIN_TEXT_LENGTH:
            st.error(
                f"⚠️ Please provide at least {MIN_TEXT_LENGTH} characters of text "
                f"(current: {len(cleaned)} chars)."
            )
        elif "MCQ" in active_types and mcq_gen is None:
            st.error("❌ MCQ Generator model failed to load. Check the logs for details.")
        else:
            with st.spinner("Generating questions …"):
                try:
                    all_questions: list[dict] = []

                    # Optional: index with RAG first
                    if use_rag and rag is not None:
                        rag.index_passages(cleaned)

                    # ── MCQ ───────────────────────────────────────────────
                    if "MCQ" in active_types and mcq_gen is not None:
                        mcqs = mcq_gen.generate_mcqs(cleaned, num_questions=num_questions)
                        if difficulty_filter != "Any":
                            filtered = [m for m in mcqs if m.get("difficulty") == difficulty_filter]
                            mcqs = filtered if filtered else mcqs
                        for m in mcqs:
                            m.setdefault("type", "MCQ")
                        all_questions.extend(mcqs)

                    # ── True/False ─────────────────────────────────────────
                    if "True/False" in active_types:
                        if tf_gen is not None:
                            tf_qs = tf_gen.generate_true_false(cleaned, num_questions=num_questions)
                            all_questions.extend(tf_qs)
                        else:
                            st.warning("⚠️ True/False Generator unavailable; skipping.")

                    # ── Short Answer ───────────────────────────────────────
                    if "Short Answer" in active_types:
                        if sa_gen is not None:
                            sa_qs = sa_gen.generate_short_answer(cleaned, num_questions=num_questions)
                            all_questions.extend(sa_qs)
                        else:
                            st.warning("⚠️ Short Answer Generator unavailable; skipping.")

                    if not all_questions:
                        st.warning("No questions could be generated from the provided text. "
                                   "Try a longer or more detailed passage.")
                    else:
                        # Auto-classify Bloom's level for all questions
                        if auto_bloom and bloom_clf is not None:
                            for q in all_questions:
                                try:
                                    result = bloom_clf.classify_question(q["question"])
                                    q["level"]       = result["level"]
                                    q["level_index"] = result["level_index"]
                                    q["confidence"]  = result["confidence"]
                                    q["description"] = result["description"]
                                except Exception as clf_exc:
                                    logger.warning("Bloom classification failed: %s", clf_exc)

                        st.session_state.generated_mcqs  = all_questions
                        st.session_state.source_text     = cleaned
                        st.session_state.generation_done = True
                        st.session_state.user_answers    = {}
                        st.session_state.quiz_submitted  = False
                        st.success(f"✅ Successfully generated **{len(all_questions)}** question(s)!")

                except Exception as exc:
                    logger.error("Generation pipeline error: %s", exc)
                    st.error(f"❌ Generation failed: {exc}")

    # ── Display results ───────────────────────────────────────────────────────
    questions: list[dict] = st.session_state.generated_mcqs

    if questions:
        st.divider()
        st.subheader(f"Generated Questions ({len(questions)})")

        if st.session_state.quiz_mode:
            # ── Interactive quiz layout ───────────────────────────────────────
            for i, q in enumerate(questions, start=1):
                _render_interactive_question(i, q, sa_gen=sa_gen)
                st.markdown("---")

            # ── Quiz results summary ──────────────────────────────────────────
            answered_count = len(st.session_state.user_answers)
            if answered_count > 0:
                st.divider()
                st.subheader("📊 Quiz Results Summary")

                total    = len(questions)
                correct  = 0
                by_type: dict[str, dict] = {}

                for i, q in enumerate(questions, start=1):
                    qtype   = q.get("type", "MCQ")
                    by_type.setdefault(qtype, {"correct": 0, "total": 0})
                    by_type[qtype]["total"] += 1

                    user_ans = st.session_state.user_answers.get(i)
                    if user_ans is None:
                        continue

                    is_correct = False
                    if qtype == "MCQ":
                        ans_lbl = q.get("answer_label", "")
                        user_lbl = str(user_ans)[1] if user_ans and len(str(user_ans)) > 1 else ""
                        is_correct = user_lbl == ans_lbl
                    elif qtype == "True/False":
                        is_correct = user_ans == q.get("answer", True)
                    elif qtype == "Short Answer":
                        if sa_gen is not None:
                            res = sa_gen.check_answer(
                                user_ans, q.get("answer", ""), q.get("keywords", [])
                            )
                            is_correct = res["is_correct"]
                        else:
                            from difflib import SequenceMatcher
                            ratio = SequenceMatcher(
                                None, str(user_ans).lower(), q.get("answer", "").lower()
                            ).ratio()
                            is_correct = ratio >= 0.80

                    if is_correct:
                        correct += 1
                        by_type[qtype]["correct"] += 1

                pct = correct / answered_count * 100 if answered_count else 0

                # Score banner
                if pct >= 80:
                    st.success(f"🏆 Score: **{correct}/{answered_count}** ({pct:.0f}%) — Excellent!")
                elif pct >= 60:
                    st.warning(f"👍 Score: **{correct}/{answered_count}** ({pct:.0f}%) — Good effort!")
                else:
                    st.error(f"📚 Score: **{correct}/{answered_count}** ({pct:.0f}%) — Keep practising!")

                if answered_count < total:
                    st.caption(f"ℹ️ {total - answered_count} question(s) not yet answered.")

                # Breakdown table
                if len(by_type) > 1:
                    st.markdown("**Breakdown by question type:**")
                    breakdown_data = [
                        {
                            "Type": t,
                            "Answered": by_type[t]["total"],
                            "Correct":  by_type[t]["correct"],
                            "Score":    f"{by_type[t]['correct']}/{by_type[t]['total']}",
                        }
                        for t in by_type
                    ]
                    st.dataframe(
                        pd.DataFrame(breakdown_data).set_index("Type"),
                        use_container_width=True,
                    )

                # Retry button
                if st.button("🔄 Retry Quiz", use_container_width=False):
                    st.session_state.user_answers   = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()

        else:
            # ── Static card layout (non-quiz mode) ───────────────────────────
            show_bloom = auto_bloom if generate_clicked else True
            for i, q in enumerate(questions, start=1):
                if q.get("type", "MCQ") == "MCQ":
                    _render_mcq_card(i, q, show_bloom=show_bloom)
                else:
                    # Minimal static card for non-MCQ types
                    qtype  = q.get("type", "")
                    q_text = q.get("question", "")
                    answer = q.get("answer", "")
                    type_colours = {
                        "True/False":   "#009688",
                        "Short Answer": "#E91E63",
                    }
                    tc = type_colours.get(qtype, "#607D8B")
                    badges = (
                        f'<span class="badge" style="background:{tc}">{qtype}</span>'
                    )
                    if q.get("level"):
                        badges += _bloom_badge(q["level"])

                    st.markdown(
                        f"""
                        <div class="question-card">
                            <div class="question-text">Q{i}. {q_text}</div>
                            <div style="margin-bottom:10px">{badges}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.expander("Show Answer"):
                        st.markdown(
                            f'<div class="answer-box">✅ Answer: {answer}</div>',
                            unsafe_allow_html=True,
                        )
                        if q.get("explanation"):
                            st.caption(q["explanation"])

    elif not generate_clicked:
        st.info("💡 Enter or upload text and click **Generate Questions** to get started.")


# ---------------------------------------------------------------------------
# Tab 2 – Analytics
# ---------------------------------------------------------------------------
def tab_analytics() -> None:
    st.subheader("📊 Question Analytics")

    mcqs: list[dict] = st.session_state.generated_mcqs

    if not mcqs:
        st.info("No questions generated yet. Go to **Generate Questions** first.")
        return

    df = pd.DataFrame(mcqs)

    # ── Top-level stats ───────────────────────────────────────────────────────
    total  = len(mcqs)
    levels = df["level"].nunique() if "level" in df.columns else 0
    avg_opts = df["options"].apply(len).mean() if "options" in df.columns else 4

    col1, col2, col3 = st.columns(3)
    for col, number, label in [
        (col1, total,            "Total Questions"),
        (col2, levels,           "Bloom Levels Used"),
        (col3, f"{avg_opts:.1f}", "Avg. Options per Q"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-number">{number}</div>'
                f'<div class="stat-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_col, diff_col = st.columns(2)

    with chart_col:
        st.markdown("#### Bloom's Level Distribution")
        if "level" in df.columns and df["level"].notna().any():
            bloom_counts = df["level"].value_counts().reset_index()
            bloom_counts.columns = ["Level", "Count"]

            # Colour map for chart
            colour_sequence = [
                BLOOM_COLOURS.get(lvl, "#888888")
                for lvl in bloom_counts["Level"]
            ]
            try:
                import plotly.express as px
                fig = px.bar(
                    bloom_counts,
                    x="Level",
                    y="Count",
                    color="Level",
                    color_discrete_sequence=colour_sequence,
                    text="Count",
                )
                fig.update_layout(showlegend=False, plot_bgcolor="#f8f9fa",
                                  paper_bgcolor="#f8f9fa")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(bloom_counts.set_index("Level"))
        else:
            st.info("Bloom level data not available.")

    with diff_col:
        st.markdown("#### Difficulty Distribution")
        if "difficulty" in df.columns and df["difficulty"].notna().any():
            diff_counts = df["difficulty"].value_counts().reset_index()
            diff_counts.columns = ["Difficulty", "Count"]
            try:
                import plotly.express as px
                fig2 = px.pie(
                    diff_counts,
                    values="Count",
                    names="Difficulty",
                    color="Difficulty",
                    color_discrete_map=DIFFICULTY_COLOURS,
                    hole=0.4,
                )
                fig2.update_layout(plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa")
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                st.bar_chart(diff_counts.set_index("Difficulty"))
        else:
            st.info("Difficulty data not available.")

    # ── Statistics table ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Question Details Table")
    display_cols = [c for c in ["question", "level", "difficulty", "confidence"]
                    if c in df.columns]
    display_df = df[display_cols].copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
    if "Confidence" in display_df.columns:
        display_df["Confidence"] = display_df["Confidence"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) and x else "N/A"
        )
    st.dataframe(display_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 – Export
# ---------------------------------------------------------------------------
def tab_export() -> None:
    st.subheader("📥 Export Questions")

    mcqs: list[dict] = st.session_state.generated_mcqs

    if not mcqs:
        st.info("No questions to export yet. Generate some questions first.")
        return

    st.markdown(f"**{len(mcqs)} question(s)** ready to export.")
    st.divider()

    col_csv, col_xl, col_pdf, col_pdf_ans = st.columns(4)

    # ── CSV ───────────────────────────────────────────────────────────────────
    with col_csv:
        st.markdown("#### 📄 CSV")
        st.caption("Spreadsheet-compatible comma-separated file.")
        try:
            csv_buf = export_to_csv(mcqs)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_buf,
                file_name="mcqs.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"CSV export failed: {exc}")

    # ── Excel ─────────────────────────────────────────────────────────────────
    with col_xl:
        st.markdown("#### 📊 Excel")
        st.caption("Formatted workbook with summary sheet.")
        try:
            xl_buf = export_to_excel(mcqs)
            st.download_button(
                label="⬇️ Download Excel",
                data=xl_buf,
                file_name="mcqs.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Excel export failed: {exc}")

    # ── PDF (paper) ───────────────────────────────────────────────────────────
    with col_pdf:
        st.markdown("#### 📑 PDF Paper")
        st.caption("Clean A4 test paper without answers.")
        try:
            pdf_buf = export_to_pdf(mcqs, title="MCQ Test Paper", show_answers=False)
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_buf,
                file_name="mcqs_paper.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"PDF export failed: {exc}")

    # ── PDF (with answers) ────────────────────────────────────────────────────
    with col_pdf_ans:
        st.markdown("#### 🔑 PDF + Answers")
        st.caption("Test paper with answer key appended.")
        try:
            pdf_ans_buf = export_to_pdf(mcqs, title="MCQ Test Paper", show_answers=True)
            st.download_button(
                label="⬇️ Download PDF + Key",
                data=pdf_ans_buf,
                file_name="mcqs_with_answers.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"PDF (with answers) export failed: {exc}")

    st.divider()
    st.success("💾 Select a format above to download your questions.")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main() -> None:
    _init_session_state()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#2E4057,#048A81);
                    border-radius:12px;padding:24px 32px;margin-bottom:24px;">
            <h1 style="color:#ffffff;margin:0;font-size:2.2rem;">🧠 AI Question Generator</h1>
            <p style="color:#cce8e5;margin:6px 0 0 0;font-size:1rem;">
                Generate Bloom's-Taxonomy-aligned MCQ, True/False, and Short Answer questions
                from any text — then practise interactively in Quiz Mode.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load models ───────────────────────────────────────────────────────────
    mcq_gen    = load_mcq_generator()
    bloom_clf  = load_bloom_classifier()
    rag        = load_rag_retriever()
    tf_gen     = load_tf_generator()
    sa_gen     = load_sa_generator()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["📝 Generate Questions", "📊 Analytics", "📥 Export"]
    )

    with tab1:
        tab_generate(mcq_gen, bloom_clf, rag, tf_gen=tf_gen, sa_gen=sa_gen)
    with tab2:
        tab_analytics()
    with tab3:
        tab_export()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#aaaaaa;font-size:12px;'>"
        "AI Question Generator · Built with Streamlit, HuggingFace Transformers &amp; FAISS"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
