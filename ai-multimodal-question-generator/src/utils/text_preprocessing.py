"""Text preprocessing utilities.

Lightweight, dependency-minimal helpers for cleaning and normalising raw text
before it is passed to NLP models or stored as structured output.
"""

from __future__ import annotations

import logging
import re

import nltk

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure required NLTK data is present
# ---------------------------------------------------------------------------
def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded"""
    required_resources = [
        'punkt_tab',
        'averaged_perceptron_tagger_eng', 
        'maxent_ne_chunker_tab',
        'words',
        'wordnet',
        'omw-1.4',
        'stopwords'
    ]
    
    for name in required_resources:
        try:
            nltk.download(name, quiet=True, raise_on_error=True)
        except:
            pass

ensure_nltk_data()


# ---------------------------------------------------------------------------
# URL removal
# ---------------------------------------------------------------------------
_URL_PATTERN: re.Pattern[str] = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)


def remove_urls(text: str, replacement: str = "") -> str:
    """Strip all URLs from *text*.

    Removes tokens that start with ``http://``, ``https://``, or ``www.``.

    Args:
        text: Input string that may contain URLs.
        replacement: String to substitute in place of each removed URL.
            Defaults to ``""`` (delete).

    Returns:
        The input string with all URL tokens replaced by *replacement*.
        Returns an empty string when *text* is blank.

    Example:
        >>> remove_urls("Visit https://example.com for details.")
        'Visit  for details.'
        >>> remove_urls("See www.example.org.", replacement="[LINK]")
        'See [LINK].'
    """
    if not text:
        return ""
    return _URL_PATTERN.sub(replacement, text)


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    """Collapse all sequences of whitespace into a single space.

    Trims leading/trailing whitespace and replaces every internal run of
    spaces, tabs, newlines, or other whitespace characters with one space.

    Args:
        text: Input string with potentially irregular whitespace.

    Returns:
        A whitespace-normalised string.  Returns ``""`` for blank input.

    Example:
        >>> normalize_whitespace("  Hello\\t\\tworld.  \\n  How are you?  ")
        'Hello world. How are you?'
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# General text cleaning
# ---------------------------------------------------------------------------
def clean_text(
    text: str,
    *,
    remove_urls_flag: bool = True,
    remove_html: bool = True,
    remove_special_chars: bool = True,
    lowercase: bool = False,
    keep_sentence_punct: bool = True,
) -> str:
    """Clean *text* by applying a configurable sequence of normalisation steps.

    Processing order:

    1. Strip HTML / XML tags (if *remove_html* is ``True``).
    2. Remove URLs (if *remove_urls_flag* is ``True``).
    3. Remove non-ASCII characters.
    4. Remove special characters, keeping alphanumerics, spaces, and
       optionally sentence-ending punctuation (if *remove_special_chars*
       is ``True``).
    5. Collapse whitespace via :func:`normalize_whitespace`.
    6. Optionally lower-case the result.

    Args:
        text: Raw input text to clean.
        remove_urls_flag: When ``True``, strip URLs before other processing.
            Defaults to ``True``.
        remove_html: When ``True``, remove HTML / XML tags.
            Defaults to ``True``.
        remove_special_chars: When ``True``, remove characters that are not
            alphanumeric, whitespace, or basic sentence punctuation
            (``.``, ``!``, ``?``, ``,``, ``:``, ``-``).
            Defaults to ``True``.
        lowercase: When ``True``, lower-case the final string.
            Defaults to ``False``.
        keep_sentence_punct: When ``True`` and *remove_special_chars* is
            ``True``, preserve ``.``, ``!``, ``?``, ``,``, ``:``, ``-``.
            Defaults to ``True``.

    Returns:
        The cleaned text string.  Returns ``""`` for blank or ``None``-like
        input.

    Example:
        >>> raw = "<p>Visit https://example.com! It's great—really.</p>"
        >>> clean_text(raw)
        "Visit It's great-really."
        >>> clean_text(raw, lowercase=True)
        "visit it's great-really."
    """
    if not text or not text.strip():
        return ""

    # 1. Strip HTML tags
    if remove_html:
        text = re.sub(r"<[^>]+>", " ", text)

    # 2. Remove URLs
    if remove_urls_flag:
        text = remove_urls(text)

    # 3. Drop non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # 4. Remove special characters
    if remove_special_chars:
        if keep_sentence_punct:
            text = re.sub(r"[^a-zA-Z0-9\s.!?,:\-']", " ", text)
        else:
            text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # 5. Collapse whitespace
    text = normalize_whitespace(text)

    # 6. Lowercase
    if lowercase:
        text = text.lower()

    return text


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
def split_into_sentences(text: str, min_length: int = 5) -> list[str]:
    """Split *text* into individual sentences using the NLTK sentence tokeniser.

    Uses the Punkt sentence tokeniser, which handles abbreviations and other
    edge-cases better than a naive split on ``.``.

    Args:
        text: Input text to split.  Should be plain text without HTML markup.
        min_length: Minimum character length a sentence must have to be included
            in the output.  Short fragments (e.g. stray punctuation) are
            discarded.  Defaults to ``5``.

    Returns:
        A list of sentence strings.  Returns ``[]`` for blank input.

    Example:
        >>> split_into_sentences("Dr. Smith went to Washington. He loved it.")
        ['Dr. Smith went to Washington.', 'He loved it.']
    """
    if not text or not text.strip():
        return []

    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as exc:  # noqa: BLE001
        logger.error("Sentence tokenisation failed: %s", exc)
        # Graceful fallback: split on newlines
        sentences = [s.strip() for s in text.splitlines()]

    return [s.strip() for s in sentences if s.strip() and len(s.strip()) >= min_length]


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------
def preprocess_text(
    text: str,
    *,
    lowercase: bool = False,
    min_sentence_length: int = 5,
) -> dict[str, str | list[str]]:
    """Apply the full preprocessing pipeline to *text*.

    Runs :func:`remove_urls`, :func:`clean_text`, :func:`normalize_whitespace`,
    and :func:`split_into_sentences` in sequence and returns all intermediate
    and final results.

    Args:
        text: Raw input text.
        lowercase: Pass-through to :func:`clean_text`.  Defaults to ``False``.
        min_sentence_length: Pass-through to :func:`split_into_sentences`.
            Defaults to ``5``.

    Returns:
        A dictionary with keys:

        .. code-block:: python

            {
                "original":           str,   # unchanged input
                "no_urls":            str,   # after URL removal
                "cleaned":            str,   # after full clean_text pass
                "normalized":         str,   # after whitespace normalisation
                "sentences":          list[str],
                "sentence_count":     str,   # str repr of int
            }

    Example:
        >>> result = preprocess_text("<b>Hello   World!</b> Visit https://x.com")
        >>> result["cleaned"]
        'Hello World!'
        >>> result["sentences"]
        ['Hello World!']
    """
    no_urls = remove_urls(text)
    cleaned = clean_text(text, remove_urls_flag=True, lowercase=lowercase)
    normalized = normalize_whitespace(cleaned)
    sentences = split_into_sentences(normalized, min_length=min_sentence_length)

    return {
        "original":       text,
        "no_urls":        no_urls,
        "cleaned":        cleaned,
        "normalized":     normalized,
        "sentences":      sentences,
        "sentence_count": str(len(sentences)),
    }


# ---------------------------------------------------------------------------
# Tests / example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import textwrap

    _SEP = "=" * 65

    def _show(label: str, value: object) -> None:
        print(f"  {label:<22}: {value}")

    # ── remove_urls ──────────────────────────────────────────────────────────
    print(f"\n{_SEP}\n  remove_urls\n{_SEP}")
    cases = [
        "Visit https://example.com for more.",
        "See http://foo.org and www.bar.net.",
        "No URL here at all.",
    ]
    for c in cases:
        _show("input", c)
        _show("output", remove_urls(c))
        print()

    # ── normalize_whitespace ─────────────────────────────────────────────────
    print(f"{_SEP}\n  normalize_whitespace\n{_SEP}")
    ws_cases = [
        "  Hello\t\tworld.\n  How are you?  ",
        "Already clean.",
        "\n\n\n   ",
    ]
    for c in ws_cases:
        _show("input ", repr(c))
        _show("output", repr(normalize_whitespace(c)))
        print()

    # ── clean_text ───────────────────────────────────────────────────────────
    print(f"{_SEP}\n  clean_text\n{_SEP}")
    raw = "<p>Visit https://example.com!  It's <b>great</b>—really. 😊</p>"
    _show("input",            raw)
    _show("default",          clean_text(raw))
    _show("lowercase",        clean_text(raw, lowercase=True))
    _show("no punct",         clean_text(raw, keep_sentence_punct=False))
    print()

    # ── split_into_sentences ─────────────────────────────────────────────────
    print(f"{_SEP}\n  split_into_sentences\n{_SEP}")
    para = (
        "Dr. Smith went to Washington D.C. last Tuesday.  "
        "She presented her findings on COVID-19 vaccines.  "
        "The results? Extraordinary."
    )
    _show("input", textwrap.shorten(para, 60))
    for i, s in enumerate(split_into_sentences(para), 1):
        _show(f"sentence {i}", s)
    print()

    # ── full pipeline ────────────────────────────────────────────────────────
    print(f"{_SEP}\n  preprocess_text (full pipeline)\n{_SEP}")
    result = preprocess_text(
        "<h1>Photosynthesis</h1>  Plants use https://en.wikipedia.org/wiki/Light "
        "to produce glucose.  The process occurs in chloroplasts.",
    )
    for k, v in result.items():
        _show(k, v)
