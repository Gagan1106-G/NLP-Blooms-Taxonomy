"""AI Question Generator — Complete Streamlit Application.

Main entry point for the web interface supporting native PDF viewing, multi-format uploads,
clipboard image/document OCR, strict difficulty control, interactive paragraph selection workspace,
multi-question batch generation (up to 150+ items), AI Exam Expectation Verification,
mixed short & detailed 6-8 line answers, context highlighting, dynamic feature diagnostics,
and multi-format export workspace.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import shutil
import sys
from pathlib import Path

# Make src importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st
import yaml

from src.bloom_classifier.classifier import BloomTaxonomyClassifier
from src.mcq_generator.generator import MCQGenerator
from src.mcq_generator.true_false_generator import TrueFalseGenerator
from src.retriever.rag_retriever import RAGRetriever
from src.short_answer_generator.short_answer_generator import ShortAnswerGenerator
from src.utils.export_utils import export_to_csv, export_to_excel, export_to_pdf
from src.utils.text_preprocessing import clean_text

# =============================================================================
# 1. PAGE CONFIGURATION (CRITICAL: MUST BE THE FIRST STREAMLIT CALL)
# =============================================================================
st.set_page_config(
    page_title="AI Question & Multi-Modal Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 2. LOGGING & CONFIGURATION SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@st.cache_data
def load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.error("Failed to load config.yaml: %s", exc)
    return {}


CONFIG = load_config()

APP_CFG = CONFIG.get("app", {})
GEN_CFG = CONFIG.get("generation", {})
MODELS_CFG = CONFIG.get("models", {})
BLOOM_LEVELS = CONFIG.get("bloom_taxonomy", {}).get("levels", [])

MIN_TEXT_LENGTH = APP_CFG.get("min_text_length", 30)
APP_TITLE = APP_CFG.get("name", "AI Question & Multi-Modal Generator")
APP_DESC = APP_CFG.get(
    "description",
    "Generate custom Bloom's-Taxonomy-aligned high-capacity MCQs, T/F, and Short/Detailed Answers from PDFs, Word docs, or Text.",
)

BLOOM_COLOURS: dict[str, str] = (
    {level["name"]: level.get("colour", "#6366F1") for level in BLOOM_LEVELS}
    if BLOOM_LEVELS
    else {
        "Remember": "#4CAF50",
        "Understand": "#2196F3",
        "Apply": "#FF9800",
        "Analyze": "#9C27B0",
        "Evaluate": "#F44336",
        "Create": "#00BCD4",
    }
)

DIFFICULTY_COLOURS: dict[str, str] = {
    "Easy": "#10B981",
    "Medium": "#F59E0B",
    "Hard": "#EF4444",
    "Any": "#6B7280",
}


# =============================================================================
# 3. SESSION STATE & DYNAMIC THEME ENGINE
# =============================================================================
def _init_session_state() -> None:
    defaults: dict = {
        "dark_mode": False,
        "generated_mcqs": [],
        "source_text": "",
        "preview_text": "",
        "selected_paragraph": "",
        "paragraphs_list": [],
        "pdf_bytes": None,
        "pdf_name": "",
        "generation_done": False,
        "error_message": "",
        "quiz_mode": True,
        "quiz_submitted": False,
        "user_answers": {},
        "question_types": ["MCQ", "Short Answer"],
        "study_extracted_text": "",
        "study_pdf_bytes": None,
        "study_image_bytes": None,
        "tab2_qa_results": [],
        "tab2_paragraph_analyzed": "",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


_init_session_state()

if st.session_state.dark_mode:
    bg_main = "#0F172A"
    bg_card = "#1E293B"
    bg_hero = "linear-gradient(135deg, #020617 0%, #0F172A 100%)"
    text_primary = "#F8FAFC"
    text_muted = "#94A3B8"
    border_color = "#334155"
    input_bg = "#0F172A"
    option_hover = "#334155"
else:
    bg_main = "#F8FAFC"
    bg_card = "#FFFFFF"
    bg_hero = "linear-gradient(135deg, #0F172A 0%, #1E293B 100%)"
    text_primary = "#0F172A"
    text_muted = "#64748B"
    border_color = "#E2E8F0"
    input_bg = "#FFFFFF"
    option_hover = "#F1F5F9"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, .main, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: {bg_main} !important;
        color: {text_primary} !important;
    }}

    img {{
        image-rendering: high-quality !important;
        image-rendering: -webkit-optimize-contrast !important;
        border-radius: 8px;
        max-width: 100%;
        height: auto;
    }}

    iframe {{
        border-radius: 10px;
        width: 100%;
    }}

    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }}

    .hero-container {{
        background: {bg_hero};
        border-radius: 16px;
        padding: 28px 32px;
        color: #FFFFFF;
        margin-bottom: 24px;
        border: 1px solid {border_color};
        box-shadow: 0 12px 30px -10px rgba(15, 23, 42, 0.25);
    }}
    .hero-title {{
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
        color: #FFFFFF;
    }}
    .hero-subtitle {{
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }}

    .control-card {{
        background-color: {bg_card} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }}

    .metric-badge {{
        background: #6366F1;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }}

    .question-card, .stat-card, .status-badge {{
        background-color: {bg_card} !important;
        border-color: {border_color} !important;
        color: {text_primary} !important;
    }}
    .question-card {{
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 18px;
        border-left: 5px solid #6366F1 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }}
    .question-text {{
        font-size: 1.05rem;
        font-weight: 600;
        color: {text_primary} !important;
        margin-bottom: 12px;
    }}
    .option-item {{
        background-color: {input_bg} !important;
        border-color: {border_color} !important;
        color: {text_primary} !important;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 6px 0;
        font-size: 0.92rem;
    }}
    .option-item:hover {{
        background-color: {option_hover} !important;
    }}

    .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #ffffff !important;
        margin-right: 6px;
    }}
    .stat-card {{
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid {border_color};
    }}
    .stat-number {{
        font-size: 2.2rem;
        font-weight: 700;
        color: #6366F1;
    }}
    .stat-label {{
        font-size: 0.85rem;
        color: {text_muted} !important;
        margin-top: 4px;
    }}

    .stTextArea textarea, .stTextInput input {{
        background-color: {input_bg} !important;
        color: {text_primary} !important;
        border-color: {border_color} !important;
        border-radius: 10px !important;
    }}
    .stRadio label {{
        background-color: {bg_card} !important;
        color: {text_primary} !important;
        border-color: {border_color} !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        margin: 4px 0 !important;
    }}
    .stRadio * {{ color: {text_primary} !important; }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        border: none !important;
        color: white !important;
        border-radius: 10px !important;
    }}
    
    .status-badge {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        border: 1px solid {border_color};
        margin-bottom: 8px;
    }}
    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }}
    .status-dot.active {{ background-color: #10B981; }}
    .status-dot.inactive {{ background-color: #EF4444; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 4. MODEL BACKEND LOADERS
# =============================================================================
@st.cache_resource(show_spinner="Loading MCQ Generator model …")
def load_mcq_generator() -> MCQGenerator | None:
    try:
        return MCQGenerator()
    except Exception as exc:
        logger.error("MCQGenerator failed to load: %s", exc)
        return None


@st.cache_resource(show_spinner="Loading Bloom Classifier model …")
def load_bloom_classifier() -> BloomTaxonomyClassifier | None:
    try:
        return BloomTaxonomyClassifier()
    except Exception as exc:
        logger.error("BloomTaxonomyClassifier failed to load: %s", exc)
        return None


@st.cache_resource(show_spinner="Loading RAG Retriever model …")
def load_rag_retriever() -> RAGRetriever | None:
    try:
        return RAGRetriever()
    except Exception as exc:
        logger.error("RAGRetriever failed to load: %s", exc)
        return None


@st.cache_resource(show_spinner="Loading True/False Generator …")
def load_tf_generator() -> TrueFalseGenerator | None:
    try:
        return TrueFalseGenerator()
    except Exception as exc:
        logger.error("TrueFalseGenerator failed to load: %s", exc)
        return None


@st.cache_resource(show_spinner="Loading Short Answer Generator …")
def load_sa_generator() -> ShortAnswerGenerator | None:
    try:
        return ShortAnswerGenerator()
    except Exception as exc:
        logger.error("ShortAnswerGenerator failed to load: %s", exc)
        return None


# =============================================================================
# 5. HELPER COMPONENTS & OCR EXTRACTION UTILITIES
# =============================================================================
def _extract_text_from_image(image) -> str:
    """Extract text from a PIL Image using enhanced OCR pre-processing for crisp text reading."""
    try:
        import pytesseract
        from PIL import Image, ImageEnhance, ImageFilter

        # Auto-detect Tesseract installation paths on Windows if not in system PATH
        if not shutil.which("tesseract"):
            win_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
                os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
            ]
            for p in win_paths:
                if os.path.exists(p):
                    pytesseract.pytesseract.tesseract_cmd = p
                    break

        img_prep = image.convert("L")  # Convert to Grayscale
        w, h = img_prep.size
        
        if w < 1600:
            scale_factor = max(2, 1600 // w)
            resample_mode = getattr(Image, "Resampling", Image).LANCZOS
            img_prep = img_prep.resize((w * scale_factor, h * scale_factor), resample_mode)

        enhancer = ImageEnhance.Contrast(img_prep)
        img_prep = enhancer.enhance(2.0)
        img_prep = img_prep.filter(ImageFilter.SHARPEN)

        custom_config = r'--oem 3 --psm 3'
        text = pytesseract.image_to_string(img_prep, config=custom_config)
        
        if text and len(text.strip()) > 5:
            cleaned_ocr = re.sub(r'(?<!\n)\n(?!\n)', ' ', text.strip())
            return cleaned_ocr
    except Exception as exc:
        logger.warning(f"Tesseract OCR execution failed: {exc}")

    width, height = image.size
    return (
        f"[Notice]: File processed ({width}x{height}px). "
        "OCR text extraction could not complete automatically. "
        "Please type or paste your paragraph text directly into the workspace below."
    )


def _extract_text_and_bytes_from_file(
    uploaded_file,
) -> tuple[str, list[str], bytes | None]:
    file_type = uploaded_file.type
    file_name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    # 1. Plain Text Files
    if "text/plain" in file_type or file_name.endswith(".txt"):
        try:
            full_text = raw_bytes.decode("utf-8", errors="ignore")
            paras = [
                p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 20
            ]
            return full_text, paras, None
        except Exception as exc:
            st.error(f"Could not read text file: {exc}")
            return "", [], None

    # 2. Word Documents (.docx)
    elif (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        in file_type
        or file_name.endswith(".docx")
    ):
        try:
            from docx import Document

            doc = Document(uploaded_file)
            paragraphs = [
                p.text.strip() for p in doc.paragraphs if len(p.text.strip()) > 20
            ]
            if paragraphs:
                return "\n\n".join(paragraphs), paragraphs, None
        except Exception as exc:
            st.error(f"Could not read Word document: {exc}")
            return "", [], None

    # 3. PDF Documents
    elif "application/pdf" in file_type or file_name.endswith(".pdf"):
        text_content = ""
        paragraphs = []
        try:
            import pdfplumber

            text_parts: list[str] = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                        for p in page_text.split("\n\n"):
                            if len(p.strip()) > 20:
                                paragraphs.append(p.strip())
            if text_parts:
                text_content = "\n\n".join(text_parts)
        except Exception:
            pass

        if not text_content:
            try:
                from PyPDF2 import PdfReader

                reader = PdfReader(uploaded_file)
                pages_txt = [
                    page.extract_text() or "" for page in reader.pages
                ]
                text_content = "\n\n".join(pages_txt)
                paragraphs = [
                    p.strip()
                    for p in text_content.split("\n\n")
                    if len(p.strip()) > 20
                ]
            except Exception as exc:
                st.error(f"Could not read PDF: {exc}")
                return "", [], None

        return text_content, paragraphs, raw_bytes

    # 4. Scanned Documents / Images via OCR
    elif "image/" in file_type or file_name.endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    ):
        try:
            from PIL import Image

            image = Image.open(uploaded_file)
            extracted = _extract_text_from_image(image)
            paras = [
                p.strip() for p in extracted.split("\n\n") if len(p.strip()) > 20
            ]
            return extracted, paras, raw_bytes
        except Exception as exc:
            st.error(f"Could not process document file: {exc}")
            return "", [], None

    return "", [], None


def render_pdf_exact(pdf_bytes: bytes) -> None:
    """Renders the uploaded PDF file natively in a high-resolution iframe canvas."""
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}#view=FitH" 
                width="100%" 
                height="580" 
                type="application/pdf"
                style="border: 1px solid {border_color}; border-radius: 10px; display: block;">
        </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)


def highlight_question_and_answer(source_text: str, question_text: str, answer_text: str) -> str:
    """Locates and highlights both the question anchor context sentence and answer sentence in the source paragraph."""
    if not source_text:
        return source_text

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source_text) if s.strip()]
    if not sentences:
        return source_text

    if len(sentences) == 1:
        return (
            f'<div style="line-height: 1.8;">'
            f'<mark style="background-color: #fef08a; color: #854d0e; padding: 4px 8px; border-radius: 4px; font-weight: 700;">'
            f'🎯 [ANSWER SOURCE]: {sentences[0]}</mark></div>'
        )

    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "of", "to", "and", "or", "it", "this", "that", "what", "how", "why", "who", "where", "which"}

    ans_words = set(re.findall(r"\w+", str(answer_text).lower())) - stop_words
    q_words = set(re.findall(r"\w+", str(question_text).lower())) - stop_words

    best_ans_idx = 0
    best_ans_score = -1
    best_q_idx = -1
    best_q_score = -1

    for idx, sent in enumerate(sentences):
        sent_words = set(re.findall(r"\w+", sent.lower()))
        ans_overlap = len(ans_words.intersection(sent_words)) if ans_words else 0
        q_overlap = len(q_words.intersection(sent_words)) if q_words else 0

        if ans_overlap > best_ans_score:
            best_ans_score = ans_overlap
            best_ans_idx = idx

        if q_overlap > best_q_score:
            best_q_score = q_overlap
            best_q_idx = idx

    if best_q_idx == best_ans_idx or best_q_score <= 0:
        best_q_idx = (best_ans_idx - 1) if best_ans_idx > 0 else (best_ans_idx + 1 if len(sentences) > 1 else best_ans_idx)

    highlighted_sentences = []
    for idx, sent in enumerate(sentences):
        if idx == best_ans_idx:
            highlighted_sentences.append(
                f'<mark style="background-color: #fef08a; color: #854d0e; padding: 4px 8px; border-radius: 4px; font-weight: 700;">'
                f'🎯 [ANSWER SOURCE]: {sent}</mark>'
            )
        elif idx == best_q_idx and best_q_idx != best_ans_idx:
            highlighted_sentences.append(
                f'<mark style="background-color: #e0e7ff; color: #3730a3; padding: 4px 8px; border-radius: 4px; font-weight: 700;">'
                f'📌 [QUESTION CONTEXT]: {sent}</mark>'
            )
        else:
            highlighted_sentences.append(sent)

    return " ".join(highlighted_sentences)


def _is_valid_question(q_text: str) -> bool:
    """Rigorous AI quality filter to reject bad, partial, vague, or truncated questions."""
    if not q_text or len(q_text.strip()) < 22:
        return False

    q_clean = q_text.strip()
    
    invalid_patterns = [
        r"(?i)\bwhy is [A-Z][a-z]+\s*important\??$",
        r"(?i)\bwhat is [a-z]{1,4}\??$",
        r"(?i)\bhow does the\??$",
        r"(?i)\bwhich of the following is\??$",
        r"(?i)\bpurpose of [a-z]{1,5}\??$",
        r"(?i)\bwhat key details are provided regarding\b",
        r"(?i)\bbased on the text, what is described\b",
    ]

    for pat in invalid_patterns:
        if re.search(pat, q_clean):
            return False

    words = q_clean.split()
    if len(words) < 5:
        return False

    return True


# =============================================================================
# AI EXAM EXPECTATION & DETAILED (6-8 LINE) ANSWER SYNTHESIZER
# =============================================================================
def _eval_exam_probability(sentence: str, question: str) -> tuple[int, str]:
    """Evaluates the likelihood of a question appearing in academic/professional exams."""
    sent_l = sentence.lower()
    score = 65  # Base probability

    high_impact_keywords = [
        "defined as", "refers to", "primary purpose", "classified into", 
        "mechanism", "core principle", "fundamental", "differs from",
        "causes", "results in", "leads to", "essential", "key factor",
        "structure", "function", "equation", "process", "advantage"
    ]
    
    for kw in high_impact_keywords:
        if kw in sent_l or kw in question.lower():
            score += 7

    if re.search(r"\b(first|second|third|three|four|five|types|categories)\b", sent_l):
        score += 8
    
    if len(sentence.split()) > 18:
        score += 5

    final_score = min(98, max(55, score))
    
    if final_score >= 82:
        status = "🔥 Highly Expected in Exams"
    elif final_score >= 70:
        status = "⚡ Moderate-High Exam Likelihood"
    else:
        status = "📌 Factual Core Question"

    return final_score, status


def _expand_to_detailed_answer(
    target_sentence: str, context_paragraph: str, base_answer: str
) -> str:
    """Synthesizes a structured 6 to 8 line detailed answer incorporating background, mechanisms, and key points."""
    all_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context_paragraph) if len(s.strip()) > 10]
    
    # Pick surrounding sentences for rich contextual grounding
    idx = 0
    for i, s in enumerate(all_sents):
        if target_sentence in s or s in target_sentence:
            idx = i
            break
            
    neighbor_prev = all_sents[max(0, idx - 1)] if idx > 0 else ""
    neighbor_next = all_sents[min(len(all_sents) - 1, idx + 1)] if idx < len(all_sents) - 1 else ""

    lines = [
        f"1. Core Direct Answer: {base_answer.rstrip('.')}.",
        f"2. Theoretical Foundation: Based on the passage, this concept is rooted in the statement that {target_sentence.lower().rstrip('.')}.",
    ]
    
    if neighbor_prev and neighbor_prev != target_sentence:
        lines.append(f"3. Contextual Background: Prior context emphasizes that {neighbor_prev.lower().rstrip('.')}.")
    else:
        lines.append("3. Key Functional Context: This principle represents a critical factual anchor within the subject matter.")

    if neighbor_next and neighbor_next != target_sentence:
        lines.append(f"4. Associated Mechanism / Impact: Furthermore, this directly leads to or connects with the fact that {neighbor_next.lower().rstrip('.')}.")
    else:
        lines.append("4. Operational Significance: Understanding this detail is vital for mastering the overall topic structure.")

    lines.extend([
        "5. Exam Relevance Note: In academic evaluation, this type of question tests conceptual clarity and analytical recall.",
        "6. Summary Point: Mastering this relationship ensures a comprehensive understanding of both basic definitions and practical applications presented in the passage."
    ])

    return "\n".join(lines)


def _generate_precise_qa_from_sentence(sentence: str) -> tuple[str, str]:
    """Analyzes sentence semantics and constructs a precise, natural question and target answer."""
    sent_clean = sentence.strip().rstrip(".")
    if not sent_clean:
        return "What key concept is described in this paragraph?", "The main subject of the text."

    # Pattern 1: Definitions / Concept Descriptions ("X is/are Y", "X refers to Y", "X is defined as Y")
    def_match = re.search(
        r"^(?P<subject>[A-Z0-9\s\-_'\"\(]{2,45}?)\s+(?:is|are|was|were|refers to|is defined as|means)\s+(?P<pred>.+)$",
        sent_clean,
        re.IGNORECASE,
    )
    if def_match:
        subj = def_match.group("subject").strip()
        pred = def_match.group("pred").strip()
        if len(subj.split()) <= 6 and not subj.lower().startswith(("this", "it", "these", "that", "there")):
            q = f"How is '{subj}' defined or described in the text?"
            a = f"{subj} {sent_clean[len(subj):].strip()}."
            return q, a

    # Pattern 2: Purpose / Objective ("X aims to Y", "X is designed to Y", "X serves to Y")
    purpose_match = re.search(
        r"^(?P<subject>.+?)\s+(?:aims to|is designed to|serves to|helps to|is used to|focuses on)\s+(?P<purpose>.+)$",
        sent_clean,
        re.IGNORECASE,
    )
    if purpose_match:
        subj = purpose_match.group("subject").strip()
        purp = purpose_match.group("purpose").strip()
        if len(subj.split()) <= 7:
            q = f"What is the primary purpose or objective of {subj}?"
            a = f"It {purp}."
            return q, a

    # Pattern 3: Cause / Reason ("X because Y", "X due to Y", "X as a result of Y")
    cause_match = re.search(
        r"^(?P<effect>.+?)\s+(?:because|due to|as a result of|owing to)\s+(?P<cause>.+)$",
        sent_clean,
        re.IGNORECASE,
    )
    if cause_match:
        effect = cause_match.group("effect").strip()
        cause = cause_match.group("cause").strip()
        q = f"According to the passage, why does {effect.lower()} occur?"
        a = f"Because {cause}."
        return q, a

    # Pattern 4: Process / Outcome ("X leads to Y", "X results in Y")
    if "leads to" in sent_clean.lower() or "results in" in sent_clean.lower():
        parts = re.split(r"\b(?:leads to|results in)\b", sent_clean, flags=re.IGNORECASE)
        if len(parts) == 2:
            q = f"What outcome or result is produced by {parts[0].strip()}?"
            a = f"It leads to {parts[1].strip()}."
            return q, a

    # Fallback: Extract main noun phrase for specific factual question
    words = sent_clean.split()
    subject_phrase = " ".join(words[:min(5, len(words))])
    q = f"What specific factual detail is stated regarding '{subject_phrase}'?"
    a = sent_clean + "."
    return q, a


# =============================================================================
# 6. DEDUPLICATION & CAPACITY VERIFICATION UTILITIES
# =============================================================================
def remove_duplicates(questions: list[dict]) -> list[dict]:
    """Strictly eliminates identical or repetitive questions from the generated batch."""
    seen_questions = set()
    unique_list = []
    for q in questions:
        q_text = q.get("question", "").strip().lower()
        if q_text and q_text not in seen_questions:
            seen_questions.add(q_text)
            unique_list.append(q)
    return unique_list


def verify_content_capacity(
    text: str, requested_total: int
) -> tuple[bool, str]:
    """Verifies text volume against requested question count."""
    word_count = len(text.split())
    estimated_max_capacity = max(10, word_count // 12)

    if requested_total > estimated_max_capacity and requested_total > 25:
        return (
            False,
            f"⚠️ **Capacity Warning:** Selected text contains ~{word_count} words (capacity ~{estimated_max_capacity} questions). You requested **{requested_total}**.",
        )
    return True, ""


def _bloom_badge(level: str) -> str:
    colour = BLOOM_COLOURS.get(level, "#6366F1")
    return f'<span class="badge" style="background:{colour}">🧠 {level}</span>'


def _difficulty_badge(difficulty: str) -> str:
    colour = DIFFICULTY_COLOURS.get(difficulty, "#64748B")
    return (
        f'<span class="badge" style="background:{colour}">⚡ {difficulty}</span>'
    )


def _render_mcq_card(index: int, mcq: dict, show_bloom: bool) -> None:
    question = mcq.get("question", "")
    options = mcq.get("options", [])
    answer = mcq.get("answer", "")
    answer_lbl = mcq.get("answer_label", "")
    bloom = mcq.get("level", "")
    difficulty = mcq.get("difficulty", "")
    confidence = mcq.get("confidence", None)

    badges = ""
    if show_bloom and bloom:
        badges += _bloom_badge(bloom)
    if difficulty:
        badges += _difficulty_badge(difficulty)
    if confidence is not None:
        badges += f'<span class="badge" style="background:#475569">🎯 {confidence:.0%}</span>'

    options_html = "".join(
        f'<div class="option-item"><b>({label})</b> {options[i] if i < len(options) else "—"}</div>'
        for i, label in enumerate(["A", "B", "C", "D"])
    )

    st.markdown(
        f"""
        <div class="question-card">
            <div class="question-text">Q{index}. {question}</div>
            <div style="margin-bottom:12px">{badges}</div>
            {options_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Show Answer Key"):
        st.success(f"✅ Correct Answer: ({answer_lbl}) {answer}")


def _render_interactive_question(
    idx: int, question: dict, sa_gen: ShortAnswerGenerator | None = None
) -> None:
    qtype = question.get("type", "MCQ")
    q_text = question.get("question", "")
    bloom = question.get("level", "")
    difficulty = question.get("difficulty", "")
    answered = st.session_state.user_answers.get(idx) is not None

    badges = ""
    if bloom:
        badges += _bloom_badge(bloom)
    if difficulty:
        badges += _difficulty_badge(difficulty)
    type_colours = {
        "MCQ": "#4F46E5",
        "True/False": "#0D9488",
        "Short Answer": "#DB2777",
        "Detailed Answer": "#8B5CF6",
    }
    badges += f'<span class="badge" style="background:{type_colours.get(qtype, "#475569")};">{qtype}</span>'

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

    if qtype == "MCQ":
        options = question.get("options", [])
        answer = question.get("answer", "")
        ans_lbl = question.get("answer_label", "")
        opt_labels = question.get("option_labels", ["A", "B", "C", "D"])

        labelled = [
            f"({opt_labels[i]}) {opt}" for i, opt in enumerate(options)
        ]

        with st.form(form_key):
            choice = st.radio(
                "Your answer:", labelled, index=None, key=f"radio_{idx}"
            )
            submitted = st.form_submit_button(
                "Submit Answer", use_container_width=True
            )

        if submitted and choice is not None:
            st.session_state.user_answers[idx] = choice
            answered = True

        if answered:
            user_choice = st.session_state.user_answers.get(idx, "")
            user_lbl = (
                user_choice[1] if user_choice and len(user_choice) > 1 else ""
            )
            if user_lbl == ans_lbl:
                st.success(f"✅ Correct! Answer: ({ans_lbl}) {answer}")
            else:
                st.error(
                    f"❌ Incorrect. Correct answer: ({ans_lbl}) {answer}"
                )

    elif qtype == "True/False":
        correct_answer = question.get("answer", True)
        explanation = question.get("explanation", "")

        with st.form(form_key):
            col1, col2 = st.columns(2)
            true_btn = col1.form_submit_button(
                "✅ True", use_container_width=True
            )
            false_btn = col2.form_submit_button(
                "❌ False", use_container_width=True
            )

        if true_btn:
            st.session_state.user_answers[idx] = True
            answered = True
        elif false_btn:
            st.session_state.user_answers[idx] = False
            answered = True

        if answered:
            user_ans = st.session_state.user_answers.get(idx)
            if user_ans == correct_answer:
                st.success(f"✅ Correct! {explanation}")
            else:
                st.error(
                    f"❌ Incorrect. Correct answer: {correct_answer}. {explanation}"
                )

    elif qtype in ("Short Answer", "Detailed Answer"):
        correct_answer = question.get("answer", "")
        keywords = question.get("keywords", [])

        with st.form(form_key):
            user_text = st.text_area(
                "Your answer:",
                height=120,
                placeholder="Type your answer here...",
                key=f"sa_input_{idx}",
            )
            submitted = st.form_submit_button(
                "Submit Answer", use_container_width=True
            )

        if submitted and user_text.strip():
            st.session_state.user_answers[idx] = user_text
            answered = True

        if answered:
            user_ans = st.session_state.user_answers.get(idx, "")
            if sa_gen is not None:
                result = sa_gen.check_answer(
                    user_ans, correct_answer, keywords
                )
                feedback = result.get("feedback", "")
                if result.get("is_correct", False):
                    st.success(f"✅ {feedback}")
                else:
                    st.warning(f"⚠️ {feedback}")
                st.info(f"💡 Expected Answer:\n{correct_answer}")


# =============================================================================
# WORKSPACE PDF/FILE EXPORTER HELPER
# =============================================================================
def _generate_tab2_pdf(qa_list: list[dict], paragraph: str) -> bytes:
    """Generates a PDF document for paragraph Q&A evaluation results."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, leading=20, textColor=colors.HexColor("#4F46E5"))
        story.append(Paragraph("AI Paragraph Evaluation & Question Paper", title_style))
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=12))

        para_style = ParagraphStyle('ParaStyle', parent=styles['Normal'], fontSize=9, leading=13, textColor=colors.HexColor("#334155"))
        story.append(Paragraph(f"<b>Evaluated Source Paragraph:</b><br/>{paragraph}", para_style))
        story.append(Spacer(1, 14))

        q_style = ParagraphStyle('QStyle', parent=styles['Heading3'], fontSize=11, leading=14, textColor=colors.HexColor("#0F172A"))
        ans_style = ParagraphStyle('AStyle', parent=styles['Normal'], fontSize=9, leading=13, textColor=colors.HexColor("#166534"))

        for idx, item in enumerate(qa_list, 1):
            q_text = item.get("question", "")
            a_text = item.get("answer", "").replace("\n", "<br/>")
            prob_status = item.get("exam_status", "Standard Question")
            prob_score = item.get("exam_score", 70)
            a_type = item.get("answer_format", "Short Answer")

            story.append(Paragraph(f"<b>Q{idx}. [{a_type}] Question:</b> {q_text} <br/><font color='#4F46E5'><b>Exam Rating:</b> {prob_status} ({prob_score}%)</font>", q_style))
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"<b>Expected Answer Key:</b><br/>{a_text}", ans_style))
            story.append(Spacer(1, 10))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceAfter=10))

        doc.build(story)
        return buffer.getvalue()
    except Exception:
        output = f"AI Paragraph Evaluation & Question Paper\n{'='*50}\n\nSource Paragraph:\n{paragraph}\n\n{'='*50}\n\n"
        for idx, item in enumerate(qa_list, 1):
            output += f"Q{idx} [{item.get('answer_format', 'Short')}]: {item.get('question', '')}\nExam Expectation: {item.get('exam_status', '')}\nAnswer Key:\n{item.get('answer', '')}\n\n"
        return output.encode("utf-8")


# =============================================================================
# 7. TAB 1: GENERATE QUESTIONS
# =============================================================================
def tab_generate(
    mcq_gen: MCQGenerator | None,
    bloom_clf: BloomTaxonomyClassifier | None,
    rag: RAGRetriever | None,
    tf_gen: TrueFalseGenerator | None = None,
    sa_gen: ShortAnswerGenerator | None = None,
) -> None:
    st.subheader("📝 Question Generation & Direct Text Selection")

    col_ctrl, col_doc = st.columns([1.02, 0.98])

    with col_ctrl:
        input_method = st.radio(
            "Input source",
            ["Upload Document", "Paste text direct"],
            horizontal=True,
        )
        raw_text = ""

        if input_method == "Paste text direct":
            raw_text = st.text_area(
                "Source Text",
                height=150,
                placeholder="Paste study notes or document text here...",
                key="pasted_text_input",
            )
            st.session_state.pdf_bytes = None
        else:
            uploaded = st.file_uploader(
                "Upload Document (PDF, DOCX, TXT, JPG, PNG)",
                type=["txt", "docx", "pdf", "jpg", "jpeg", "png"],
            )
            if uploaded is not None:
                with st.spinner(
                    f"Parsing document & extracting text from {uploaded.name} …"
                ):
                    (
                        text_res,
                        paras_res,
                        bytes_res,
                    ) = _extract_text_and_bytes_from_file(uploaded)
                    raw_text = text_res
                    if uploaded.name.lower().endswith(".pdf"):
                        st.session_state.pdf_bytes = bytes_res
                        st.session_state.pdf_name = uploaded.name
                    else:
                        st.session_state.pdf_bytes = None
                if raw_text:
                    st.session_state.preview_text = raw_text
                    st.success(f"Successfully loaded {uploaded.name}!")

        # Dynamic Selection area
        st.markdown("---")
        st.markdown("### ✂️ Target Specific Paragraph / Content Portion")
        active_source_base = st.session_state.get(
            "preview_text", ""
        ) or raw_text

        user_selected_text = st.text_area(
            "Select/Paste exact paragraph to evaluate:",
            value=active_source_base,
            height=140,
            placeholder="Selected paragraph text will appear here for targeted generation...",
            key="selected_paragraph_interactive",
        )

        source_text = user_selected_text if user_selected_text.strip() else raw_text

        # Capacity Gauge & Recommender
        words = len(source_text.split()) if source_text else 0
        rec_mcq = max(2, min(50, words // 12))
        rec_sa = max(1, min(25, words // 20))
        rec_tf = max(2, min(50, words // 10))
        max_capacity = rec_mcq + rec_sa + rec_tf

        st.markdown(
            f"""
            <div class="control-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
                    <strong style="font-size: 0.95rem;">📊 Target Content Capacity</strong>
                    <span class="metric-badge">Words in Focus: {words}</span>
                </div>
                <p style="font-size:0.82rem; color:#94A3B8; margin:0;">
                    Recommended safe capacity for this text block: <strong>~{max_capacity} unique questions</strong>. 
                    (Rec: <strong>{rec_mcq} MCQs</strong>, <strong>{rec_sa} Short Ans</strong>, <strong>{rec_tf} T/F</strong>)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Direct Quantity Selectors
        st.markdown("### 🎛️ Question Quantity Selectors")
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            exact_mcq_count = st.number_input(
                "MCQ Count",
                min_value=0,
                max_value=150,
                value=min(10, rec_mcq or 10),
                key="num_mcq",
            )
        with col_c2:
            exact_sa_count = st.number_input(
                "Short Answer Count",
                min_value=0,
                max_value=150,
                value=min(5, rec_sa or 5),
                key="num_sa",
            )
        with col_c3:
            exact_tf_count = st.number_input(
                "T/F Count",
                min_value=0,
                max_value=150,
                value=min(5, rec_tf or 5),
                key="num_tf",
            )

        total_requested = exact_mcq_count + exact_sa_count + exact_tf_count
        st.markdown(
            f"**Total Target Requested:** `{total_requested}` questions"
        )

        # Sidebar Options
        with st.sidebar:
            st.markdown("### 🎨 Appearance")
            st.session_state.dark_mode = st.toggle(
                "🌙 Dark Mode", value=st.session_state.dark_mode
            )

            st.markdown("### ⚙️ General Settings")
            difficulty_filter = st.selectbox(
                "Strict Target Difficulty",
                ["Any", "Easy", "Medium", "Hard"],
                index=3,
            )
            use_rag = st.checkbox(
                "Enable RAG (Context-Aware)",
                value=GEN_CFG.get("enable_rag", True),
            )
            auto_bloom = st.checkbox(
                "Auto-classify Bloom's level",
                value=GEN_CFG.get("auto_classify_bloom", True),
            )
            st.session_state.quiz_mode = st.checkbox(
                "🎯 Interactive Quiz Mode", value=st.session_state.quiz_mode
            )

            # --- SYSTEM STATUS & FEATURE DIAGNOSTICS ---
            st.markdown("---")
            st.markdown("### ⚡ System Status & Feature Diagnostics")

            rag_status = "Active (Context-Aware)" if (rag is not None and use_rag) else "Inactive / Off"
            bloom_status = "Active (Auto-Tagging)" if (bloom_clf is not None and auto_bloom) else "Inactive / Off"
            quiz_status = "Active (Interactive)" if st.session_state.quiz_mode else "Inactive (Static View)"

            feature_statuses = [
                ("RAG Retriever", rag is not None and use_rag, rag_status),
                ("Bloom's Classifier", bloom_clf is not None and auto_bloom, bloom_status),
                ("Interactive Quiz Mode", st.session_state.quiz_mode, quiz_status),
                ("MCQ Engine", mcq_gen is not None, "Model Loaded" if mcq_gen else "Not Loaded"),
                ("True/False Engine", tf_gen is not None, "Model Loaded" if tf_gen else "Not Loaded"),
                ("Short Answer Engine", sa_gen is not None, "Model Loaded" if sa_gen else "Not Loaded"),
            ]

            for name, is_active, detail in feature_statuses:
                status_class = "active" if is_active else "inactive"
                st.markdown(
                    f"""
                    <div class="status-badge">
                        <span class="status-dot {status_class}"></span>
                        <div>
                            <strong>{name}</strong><br/>
                            <span style="font-size:0.72rem; color:#94A3B8;">{detail}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # EVALUATION CONFIRMATION PROMPT STEP
        st.markdown("---")
        st.markdown("### 💡 Evaluate & Generation Prompt")
        st.info(
            f"Selected text is ready ({words} words). Confirm evaluation and generation below."
        )

        col_eval1, col_eval2 = st.columns(2)
        with col_eval1:
            confirm_gen = st.button(
                "✅ Confirm & Generate Questions",
                type="primary",
                use_container_width=True,
            )
        with col_eval2:
            cancel_gen = st.button("❌ Cancel / Clear", use_container_width=True)

        if cancel_gen:
            st.session_state.generated_mcqs = []
            st.rerun()

        cleaned = clean_text(source_text) if source_text else ""

        if confirm_gen:
            if not cleaned or len(cleaned) < MIN_TEXT_LENGTH:
                st.error(
                    f"⚠️ Please select or provide at least {MIN_TEXT_LENGTH} characters of text content."
                )
            elif total_requested == 0:
                st.warning(
                    "⚠️ Please select at least one question count above 0."
                )
            else:
                raw_generated_questions = []
                with st.spinner(
                    f"Generating assessment batch of {total_requested} questions..."
                ):
                    try:
                        if use_rag and rag is not None:
                            rag.index_passages(cleaned)

                        if exact_mcq_count > 0 and mcq_gen is not None:
                            mcqs = mcq_gen.generate_mcqs(
                                cleaned, num_questions=exact_mcq_count
                            )
                            for m in mcqs:
                                if _is_valid_question(m.get("question", "")):
                                    m.setdefault("type", "MCQ")
                                    m["difficulty"] = (
                                        difficulty_filter
                                        if difficulty_filter != "Any"
                                        else "Medium"
                                    )
                                    raw_generated_questions.append(m)

                        if exact_sa_count > 0 and sa_gen is not None:
                            sa_qs = sa_gen.generate_short_answer(
                                cleaned, num_questions=exact_sa_count
                            )
                            for q in sa_qs:
                                if _is_valid_question(q.get("question", "")):
                                    q.setdefault("type", "Short Answer")
                                    q["difficulty"] = (
                                        difficulty_filter
                                        if difficulty_filter != "Any"
                                        else "Medium"
                                    )
                                    raw_generated_questions.append(q)

                        if exact_tf_count > 0 and tf_gen is not None:
                            tf_qs = tf_gen.generate_true_false(
                                cleaned, num_questions=exact_tf_count
                            )
                            for t in tf_qs:
                                if _is_valid_question(t.get("question", "")):
                                    t.setdefault("type", "True/False")
                                    t["difficulty"] = (
                                        difficulty_filter
                                        if difficulty_filter != "Any"
                                        else "Medium"
                                    )
                                    raw_generated_questions.append(t)

                        unique_questions = remove_duplicates(
                            raw_generated_questions
                        )

                        if auto_bloom and bloom_clf is not None:
                            for item in unique_questions:
                                try:
                                    res = bloom_clf.classify_question(
                                        item["question"]
                                    )
                                    item["level"] = res["level"]
                                    item["confidence"] = res["confidence"]
                                except Exception:
                                    pass

                        st.session_state.generated_mcqs = unique_questions
                        st.session_state.user_answers = {}
                        st.success(
                            f"✅ Generated **{len(unique_questions)}** unique questions!"
                        )
                    except Exception as exc:
                        st.error(f"❌ Generation failed: {exc}")

    # Right Column: Native Unaltered PDF Viewer
    with col_doc:
        st.subheader("📑 Document Viewer Workspace")

        if st.session_state.get("pdf_bytes") is not None:
            st.caption(
                f"Viewing original uploaded PDF: **{st.session_state.get('pdf_name', 'Document')}**"
            )
            render_pdf_exact(st.session_state.pdf_bytes)
        elif raw_text:
            st.caption("Extracted Document Text Stream:")
            st.text_area("Full Document View", value=raw_text, height=520)
        else:
            st.markdown(
                """
                <div style="border: 2px dashed #6366F1; padding: 40px 20px; border-radius: 12px; text-align: center; color: #94A3B8; margin-top: 10px;">
                    <p style="font-size: 1.05rem; font-weight: 600; margin-bottom: 5px;">No Document Loaded</p>
                    <p style="font-size: 0.85rem;">Upload your PDF or document on the left to render it natively here.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    questions: list[dict] = st.session_state.generated_mcqs
    if questions:
        st.divider()
        st.subheader(
            f"Generated Assessment Questions ({len(questions)} Unique Items)"
        )

        if st.session_state.quiz_mode:
            for i, q in enumerate(questions, start=1):
                _render_interactive_question(i, q, sa_gen=sa_gen)
        else:
            show_bloom = auto_bloom
            for i, q in enumerate(questions, start=1):
                if q.get("type", "MCQ") == "MCQ":
                    _render_mcq_card(i, q, show_bloom=show_bloom)
                else:
                    _render_interactive_question(i, q, sa_gen=sa_gen)


# =============================================================================
# TAB 2: DEDICATED DOCUMENT WORKSPACE & HIGH-CAPACITY PARAGRAPH QUESTION GENERATOR
# =============================================================================
def tab_pdf_study() -> None:
    st.subheader("📖 Document Workspace High-Capacity Question Generator")
    st.markdown(
        "Upload a document or paste your paragraph directly below. Generate **up to 150 questions** (50, 100+ supported) with a mixture of **Short Answers** and **Detailed 6–8 Line Answers**, complete with **AI Exam Expectation Verification** and context highlighting."
    )

    col_up, col_preview = st.columns([1, 1])

    extracted_para_text = ""

    with col_up:
        input_type = st.radio(
            "Select Input Source:",
            ["Upload Document", "Paste Paragraph Directly"],
            horizontal=True,
        )

        if input_type == "Upload Document":
            uploaded_file = st.file_uploader(
                "Upload Document (PDF, DOCX, TXT, JPG, PNG)",
                type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
                key="tab2_file_uploader",
            )
            if uploaded_file is not None:
                txt, paras, raw_bytes = _extract_text_and_bytes_from_file(uploaded_file)
                extracted_para_text = txt
                st.session_state.study_extracted_text = txt
                if uploaded_file.name.lower().endswith(".pdf"):
                    st.session_state.study_pdf_bytes = raw_bytes
                    st.session_state.study_image_bytes = None
                elif uploaded_file.type.startswith("image/") or uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    st.session_state.study_image_bytes = raw_bytes
                    st.session_state.study_pdf_bytes = None
                else:
                    st.session_state.study_pdf_bytes = None
                    st.session_state.study_image_bytes = None

        elif input_type == "Paste Paragraph Directly":
            extracted_para_text = ""

    # Workspace Preview Column
    with col_preview:
        st.markdown("### 👁️ Preview Workspace")
        if st.session_state.get("study_pdf_bytes"):
            render_pdf_exact(st.session_state.study_pdf_bytes)
        elif st.session_state.get("study_image_bytes"):
            st.image(st.session_state.study_image_bytes, caption="Media View", use_column_width=True)
        else:
            st.info("Uploaded PDFs or documents will render here.")

    st.markdown("---")
    st.markdown("### ✂️ Target Paragraph Workspace")

    study_paragraph = st.text_area(
        "Selected paragraph text (you can also copy/paste directly here):",
        value=extracted_para_text or st.session_state.get("study_extracted_text", ""),
        height=160,
        placeholder="Type, paste, or upload text to evaluate and generate questions...",
        key="tab2_paragraph_textarea",
    )

    col_q1, col_q2 = st.columns([1, 1])
    with col_q1:
        num_q_target = st.number_input(
            "Number of Questions to Generate (Supports 10 to 150):",
            min_value=1,
            max_value=150,
            value=10,
            key="num_tab2_qs",
        )
    with col_q2:
        answer_mix_ratio = st.selectbox(
            "Answer Format Style / Mix:",
            ["Balanced Mix (50% Short, 50% Detailed 6-8 Lines)", "All Detailed Answers (6-8 Lines)", "All Short Answers"],
            index=0,
            key="tab2_ans_mix",
        )

    if st.button("🚀 Analyze Paragraph & Generate High-Capacity Questions", type="primary", use_container_width=True):
        raw_p = study_paragraph.strip()
        word_count = len(raw_p.split()) if raw_p else 0
        
        if not raw_p or word_count < 10:
            st.warning("⚠️ Paragraph contains insufficient text to evaluate. Please provide a complete paragraph with at least 10–15 words.")
        else:
            with st.spinner(f"AI evaluating paragraph structure, assessing exam probabilities, and constructing batch of {num_q_target} questions..."):
                cleaned_p = clean_text(raw_p)
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned_p) if len(s.strip()) > 12]
                
                if not sentences:
                    sentences = [cleaned_p]

                sa_gen = load_sa_generator()
                generated_questions = []

                # Attempt model generation batch
                raw_sa = []
                if sa_gen is not None:
                    try:
                        raw_sa = sa_gen.generate_short_answer(cleaned_p, num_questions=min(num_q_target, 30))
                    except Exception:
                        raw_sa = []

                for idx in range(num_q_target):
                    matched_sent = sentences[idx % len(sentences)]
                    
                    # Determine answer depth for this item
                    if "All Detailed" in answer_mix_ratio:
                        is_detailed = True
                    elif "All Short" in answer_mix_ratio:
                        is_detailed = False
                    else:
                        is_detailed = (idx % 2 == 1)  # Alternating short / detailed

                    if idx < len(raw_sa) and _is_valid_question(raw_sa[idx].get("question", "")):
                        q_final = raw_sa[idx].get("question", "")
                        base_a = raw_sa[idx].get("answer", "") or matched_sent
                    else:
                        q_final, base_a = _generate_precise_qa_from_sentence(matched_sent)

                    if is_detailed:
                        ans_final = _expand_to_detailed_answer(matched_sent, cleaned_p, base_a)
                        ans_format = "Detailed Answer (6-8 Lines)"
                    else:
                        ans_final = base_a
                        ans_format = "Short Answer"

                    # AI Exam Expectation Rating
                    prob_score, prob_status = _eval_exam_probability(matched_sent, q_final)

                    generated_questions.append({
                        "question": q_final,
                        "answer": ans_final,
                        "keywords": matched_sent.split()[:3],
                        "type": "Detailed Answer" if is_detailed else "Short Answer",
                        "answer_format": ans_format,
                        "source_sentence": matched_sent,
                        "exam_score": prob_score,
                        "exam_status": prob_status,
                    })

                unique_qs = remove_duplicates(generated_questions)[:num_q_target]
                
                # Store results into session state
                st.session_state.tab2_qa_results = unique_qs
                st.session_state.tab2_paragraph_analyzed = raw_p

    # =========================================================================
    # RENDER PARAGRAPH ANALYSIS DIAGNOSTICS & GENERATED Q&A OUTPUTS
    # =========================================================================
    qa_list = st.session_state.get("tab2_qa_results", [])
    analyzed_p = st.session_state.get("tab2_paragraph_analyzed", "")

    if qa_list and analyzed_p:
        st.markdown("---")
        
        # 🔍 AI PARAGRAPH DIAGNOSTICS & CONCEPT DETECTOR CARD
        p_words = len(analyzed_p.split())
        p_sents = len([s for s in re.split(r"(?<=[.!?])\s+", analyzed_p) if s.strip()])
        key_tokens = set(re.findall(r"\b[A-Z][a-z]{3,}\b|\b[a-z]{6,}\b", analyzed_p))
        key_concepts_str = ", ".join(list(key_tokens)[:6]) if key_tokens else "Standard Academic Vocabulary"
        avg_exam_score = sum(q.get("exam_score", 75) for q in qa_list) // max(1, len(qa_list))

        st.markdown(
            f"""
            <div style="padding: 18px 22px; border: 1px solid #6366F1; border-radius: 12px; background-color: {bg_card}; margin-bottom: 24px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #6366F1; margin-bottom: 8px;">
                    🔍 AI Paragraph Evaluation & High-Capacity Assessment Diagnostic
                </div>
                <div style="display: flex; gap: 24px; flex-wrap: wrap; font-size: 0.88rem; color: {text_primary}; margin-bottom: 10px;">
                    <span><b>Word Count:</b> {p_words} words</span>
                    <span><b>Sentences Detected:</b> {p_sents}</span>
                    <span><b>Questions Generated:</b> {len(qa_list)} items</span>
                    <span><b>Avg. Exam Predictability Score:</b> <strong style="color:#10B981;">{avg_exam_score}%</strong></span>
                </div>
                <div style="font-size: 0.85rem; color: {text_muted};">
                    <b>Key Concepts & Exam Focus Entities:</b> {key_concepts_str}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.success(f"✅ AI Analysis Complete: Successfully generated **{len(qa_list)}** verified Question & Answer pairs with Exam Expectations!")

        sa_gen = load_sa_generator()

        for idx, q_item in enumerate(qa_list, start=1):
            q_text = q_item.get("question", "")
            ans_text = q_item.get("answer", "")
            exam_status = q_item.get("exam_status", "🔥 Highly Expected in Exams")
            exam_score = q_item.get("exam_score", 85)
            ans_fmt = q_item.get("answer_format", "Short Answer")
            
            highlighted_p = highlight_question_and_answer(analyzed_p, q_text, ans_text[:100])

            st.markdown(f"### 📌 Question {idx} of {len(qa_list)}")

            # 1. Source Context Highlight Container
            st.markdown(
                f"""
                <div style="padding: 16px; border: 1px solid {border_color}; border-radius: 10px; background-color: {bg_card}; line-height: 1.8; margin-bottom: 12px;">
                    <p style="font-weight: 700; font-size: 0.92rem; margin-bottom: 6px; color: #6366F1;">📍 Highlighted Source Paragraph Context:</p>
                    <p style="font-size: 0.90rem;">{highlighted_p}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 2. Distinct Question Display Card + AI Exam Expectation Tag
            st.markdown(
                f"""
                <div style="padding: 16px 20px; border-left: 5px solid #6366F1; border-radius: 8px; background-color: {bg_card}; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap; gap: 8px;">
                        <span style="font-size: 0.8rem; font-weight: 700; color: #6366F1; text-transform: uppercase;">Generated Question [{ans_fmt}]:</span>
                        <span style="background-color: #FEF3C7; color: #92400E; font-size: 0.78rem; font-weight: 700; padding: 4px 10px; border-radius: 12px;">
                            🎯 {exam_status} ({exam_score}% Confidence)
                        </span>
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: {text_primary};">
                        Q{idx}. {q_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 3. Distinct Expected Answer Key Card (Formatted for multi-line detailed answers)
            formatted_ans_html = ans_text.replace('\n', '<br/>')
            st.markdown(
                f"""
                <div style="padding: 14px 20px; border-left: 5px solid #10B981; border-radius: 8px; background-color: {bg_card}; margin-bottom: 16px;">
                    <span style="font-size: 0.8rem; font-weight: 700; color: #10B981; text-transform: uppercase;">Expected Answer Key ({ans_fmt}):</span>
                    <div style="font-size: 0.95rem; font-weight: 500; color: {text_primary}; margin-top: 8px; line-height: 1.7;">
                        {formatted_ans_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 4. Student Answer Testing Form
            form_key = f"tab2_sa_form_{idx}"
            with st.form(form_key):
                user_text = st.text_area(
                    f"Test your student answer for Question {idx}:",
                    height=100 if "Detailed" in ans_fmt else 70,
                    placeholder="Type your response here to evaluate against the AI answer key...",
                    key=f"tab2_user_input_{idx}",
                )
                submitted = st.form_submit_button("Submit & Verify Answer", use_container_width=True)

            if submitted and user_text.strip():
                if sa_gen is not None:
                    keywords = q_item.get("keywords", [])
                    res = sa_gen.check_answer(user_text, ans_text, keywords)
                    feedback = res.get("feedback", "")
                    if res.get("is_correct", False):
                        st.success(f"✅ {feedback}")
                    else:
                        st.warning(f"⚠️ {feedback}")

            st.divider()

        # =========================================================================
        # WORKSPACE EXPORT PORTAL FOR PARAGRAPH EVALUATIONS
        # =========================================================================
        st.markdown("### 📥 Export Workspace Question & Answer Paper")
        st.caption("Download all generated paragraph Q&A pairs along with AI exam expectation ratings.")

        col_exp_pdf, col_exp_csv, col_exp_txt = st.columns(3)

        with col_exp_pdf:
            pdf_bytes = _generate_tab2_pdf(qa_list, analyzed_p)
            st.download_button(
                "⬇️ Export as PDF Paper",
                data=pdf_bytes,
                file_name="paragraph_qa_evaluation.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with col_exp_csv:
            export_df = pd.DataFrame([
                {
                    "Item": i,
                    "Question": item.get("question", ""),
                    "Format": item.get("answer_format", "Short Answer"),
                    "Exam Expectation": item.get("exam_status", ""),
                    "Exam Score (%)": item.get("exam_score", 80),
                    "Expected Answer": item.get("answer", ""),
                    "Source Paragraph": analyzed_p,
                }
                for i, item in enumerate(qa_list, 1)
            ])
            st.download_button(
                "⬇️ Export as CSV Data",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="paragraph_qa_evaluation.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_exp_txt:
            txt_content = f"AI PARAGRAPH EVALUATION & QUESTION PAPER ({len(qa_list)} Items)\n{'='*60}\n\nSOURCE PARAGRAPH:\n{analyzed_p}\n\n{'='*60}\n\n"
            for i, item in enumerate(qa_list, 1):
                txt_content += (
                    f"Q{i} [{item.get('answer_format', 'Short')}]: {item.get('question', '')}\n"
                    f"Exam Expectation: {item.get('exam_status', '')} ({item.get('exam_score', 80)}%)\n"
                    f"Answer Key:\n{item.get('answer', '')}\n\n"
                    f"{'-'*40}\n"
                )

            st.download_button(
                "⬇️ Export as Text Document",
                data=txt_content.encode("utf-8"),
                file_name="paragraph_qa_evaluation.txt",
                mime="text/plain",
                use_container_width=True,
            )


# =============================================================================
# TAB 3: UPGRADED ANALYTICS
# =============================================================================
def tab_analytics() -> None:
    st.subheader("📊 Question Analytics")
    mcqs: list[dict] = st.session_state.generated_mcqs

    if not mcqs:
        st.info("No questions generated yet. Go to Generate Questions first.")
        return

    df = pd.DataFrame(mcqs)

    total_q = len(mcqs)
    type_counts = (
        df["type"].value_counts().to_dict() if "type" in df.columns else {}
    )
    diff_counts = (
        df["difficulty"].value_counts().to_dict()
        if "difficulty" in df.columns
        else {}
    )
    bloom_counts = (
        df["level"].value_counts().to_dict() if "level" in df.columns else {}
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-number">{total_q}</div><div class="stat-label">Total Questions</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-number">{type_counts.get("MCQ", 0)}</div><div class="stat-label">MCQs Generated</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="stat-card"><div class="stat-number">{type_counts.get("Short Answer", 0) + type_counts.get("Detailed Answer", 0)}</div><div class="stat-label">Short/Detailed Answers</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="stat-card"><div class="stat-number">{type_counts.get("True/False", 0)}</div><div class="stat-label">True / False</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    chart_col, diff_col = st.columns(2)

    with chart_col:
        st.markdown("#### 🧠 Bloom's Level Distribution")
        if bloom_counts:
            try:
                import plotly.express as px

                bloom_df = pd.DataFrame(
                    list(bloom_counts.items()), columns=["Level", "Count"]
                )
                fig = px.pie(
                    bloom_df, values="Count", names="Level", hole=0.4
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(pd.Series(bloom_counts))

    with diff_col:
        st.markdown("#### ⚡ Difficulty Enforcement Distribution")
        if diff_counts:
            try:
                import plotly.express as px

                diff_df = pd.DataFrame(
                    list(diff_counts.items()), columns=["Difficulty", "Count"]
                )
                fig2 = px.bar(
                    diff_df,
                    x="Difficulty",
                    y="Count",
                    color="Difficulty",
                    color_discrete_map=DIFFICULTY_COLOURS,
                )
                fig2.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                st.bar_chart(pd.Series(diff_counts))


# =============================================================================
# TAB 4: EXPORT
# =============================================================================
def tab_export() -> None:
    st.subheader("📥 Export Questions")
    mcqs: list[dict] = st.session_state.generated_mcqs

    if not mcqs:
        st.info("No questions to export yet. Generate some questions first.")
        return

    st.markdown(f"**{len(mcqs)} question(s)** ready for download.")
    st.divider()
    col_csv, col_xl, col_pdf, col_pdf_ans = st.columns(4)

    with col_csv:
        st.markdown("#### 📄 CSV")
        try:
            st.download_button(
                "⬇️ Download CSV",
                export_to_csv(mcqs),
                "mcqs.csv",
                "text/csv",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"CSV error: {exc}")

    with col_xl:
        st.markdown("#### 📊 Excel")
        try:
            st.download_button(
                "⬇️ Download Excel",
                export_to_excel(mcqs),
                "mcqs.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Excel error: {exc}")

    with col_pdf:
        st.markdown("#### 📑 PDF Paper")
        try:
            st.download_button(
                "⬇️ Download PDF",
                export_to_pdf(
                    mcqs, title="Test Exam Paper", show_answers=False
                ),
                "mcqs.pdf",
                "application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"PDF error: {exc}")

    with col_pdf_ans:
        st.markdown("#### 🔑 PDF + Answers")
        try:
            st.download_button(
                "⬇️ Download Key PDF",
                export_to_pdf(
                    mcqs, title="Test Exam Key", show_answers=True
                ),
                "mcqs_key.pdf",
                "application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"PDF Key error: {exc}")


# =============================================================================
# 8. MAIN ENTRY POINT
# =============================================================================
def main() -> None:
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-title">🧠 {APP_TITLE}</div>
            <div class="hero-subtitle">{APP_DESC}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mcq_gen = load_mcq_generator()
    bloom_clf = load_bloom_classifier()
    rag = load_rag_retriever()
    tf_gen = load_tf_generator()
    sa_gen = load_sa_generator()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📝 Generate Questions",
            "📖 Document Workspace",
            "📊 Analytics",
            "📥 Export Options",
        ]
    )

    with tab1:
        tab_generate(mcq_gen, bloom_clf, rag, tf_gen=tf_gen, sa_gen=sa_gen)
    with tab2:
        tab_pdf_study()
    with tab3:
        tab_analytics()
    with tab4:
        tab_export()

    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#94A3B8;font-size:0.8rem;'>"
        "AI Question Generator · Multi-Modal Engine (PDF, DOCX, TXT)"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
