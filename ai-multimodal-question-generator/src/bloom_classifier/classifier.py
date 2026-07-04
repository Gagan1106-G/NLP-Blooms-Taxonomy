"""Bloom's Taxonomy Classifier module.

Classifies questions into one of the six cognitive levels defined by Bloom's
Revised Taxonomy using a fine-tuned BERT-based model with a DistilBERT fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)

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
# BloomTaxonomyClassifier
# ---------------------------------------------------------------------------
class BloomTaxonomyClassifier:
    """Classify questions according to Bloom's Revised Taxonomy.

    Attempts to load the specialised ``RyanLauQF/BloomBERT-base`` model.
    If that fails (e.g. no network access or model unavailable), falls back
    to ``distilbert-base-uncased`` with a randomly-initialised 6-class head.
    In the fallback case the predictions are **not** fine-tuned and serve only
    as a structural placeholder until a proper model is available.

    Attributes:
        BLOOM_LEVELS (dict[int, str]): Mapping from level index to level name.
        LEVEL_DESCRIPTIONS (dict[str, str]): Short cognitive description per level.
        device (torch.device): Inference device (``cuda`` or ``cpu``).
        tokenizer: Loaded tokeniser.
        model: Loaded classification model.
        max_length (int): Maximum token length for the encoder.

    Example:
        >>> clf = BloomTaxonomyClassifier()
        >>> result = clf.classify_question("What is the capital of France?")
        >>> print(result["level"])
        'Remember'
    """

    # ── Bloom's Revised Taxonomy ─────────────────────────────────────────────
    BLOOM_LEVELS: dict[int, str] = {
        0: "Remember",
        1: "Understand",
        2: "Apply",
        3: "Analyze",
        4: "Evaluate",
        5: "Create",
    }

    LEVEL_DESCRIPTIONS: dict[str, str] = {
        "Remember":   "Recall facts and basic concepts",
        "Understand": "Explain ideas or concepts",
        "Apply":      "Use information in new situations",
        "Analyze":    "Draw connections among ideas",
        "Evaluate":   "Justify decisions or choices",
        "Create":     "Produce new or original work",
    }

    # Primary and fallback model identifiers
    _PRIMARY_MODEL: str = "RyanLauQF/BloomBERT-base"
    _FALLBACK_MODEL: str = "distilbert-base-uncased"
    _NUM_LABELS: int = 6

    def __init__(self, max_length: int = 128) -> None:
        """Load the classifier model and tokeniser.

        Tries ``RyanLauQF/BloomBERT-base`` first; on any failure gracefully
        falls back to ``distilbert-base-uncased`` with a 6-label head.

        Args:
            max_length: Maximum number of tokens per input sequence.
                Defaults to ``128``.
        """
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        self.tokenizer, self.model, self._model_id = self._load_model()
        self.model.eval()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(
        self,
    ) -> tuple[Any, Any, str]:
        """Attempt to load the primary model; fall back if unavailable.

        Returns:
            A 3-tuple ``(tokenizer, model, model_id)`` where *model_id* is the
            identifier string of whichever model was successfully loaded.

        Raises:
            RuntimeError: If neither the primary nor the fallback model can be
                loaded.
        """
        # ── Try primary model ────────────────────────────────────────────────
        try:
            logger.info("Loading primary model '%s' …", self._PRIMARY_MODEL)
            tokenizer = AutoTokenizer.from_pretrained(self._PRIMARY_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(
                self._PRIMARY_MODEL,
                num_labels=self._NUM_LABELS,
                ignore_mismatched_sizes=True,
            ).to(self.device)
            logger.info("Primary model loaded successfully.")
            return tokenizer, model, self._PRIMARY_MODEL

        except Exception as primary_exc:  # noqa: BLE001
            logger.warning(
                "Could not load primary model '%s' (%s). "
                "Falling back to '%s'.",
                self._PRIMARY_MODEL,
                primary_exc,
                self._FALLBACK_MODEL,
            )

        # ── Fallback model ───────────────────────────────────────────────────
        try:
            logger.info("Loading fallback model '%s' …", self._FALLBACK_MODEL)
            tokenizer = DistilBertTokenizer.from_pretrained(self._FALLBACK_MODEL)
            model = DistilBertForSequenceClassification.from_pretrained(
                self._FALLBACK_MODEL,
                num_labels=self._NUM_LABELS,
                ignore_mismatched_sizes=True,
            ).to(self.device)
            logger.warning(
                "Using fallback model '%s'. Predictions are NOT fine-tuned "
                "for Bloom's Taxonomy and should be treated as placeholders.",
                self._FALLBACK_MODEL,
            )
            return tokenizer, model, self._FALLBACK_MODEL

        except Exception as fallback_exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load both primary ('{self._PRIMARY_MODEL}') and "
                f"fallback ('{self._FALLBACK_MODEL}') models. "
                f"Last error: {fallback_exc}"
            ) from fallback_exc

    # ------------------------------------------------------------------
    # Single question classification
    # ------------------------------------------------------------------
    def classify_question(self, question: str) -> dict[str, Any]:
        """Classify a single question into a Bloom's Taxonomy level.

        Tokenises *question*, runs a forward pass through the loaded model,
        and converts the logits to calibrated probabilities via softmax.

        Args:
            question: The question text to classify.

        Returns:
            A dictionary with the following keys:

            .. code-block:: python

                {
                    "question":    str,           # original input
                    "level":       str,           # e.g. "Remember"
                    "level_index": int,           # 0–5
                    "confidence":  float,         # probability of top level
                    "description": str,           # cognitive description
                    "scores": {                   # all 6 level probabilities
                        "Remember":   float,
                        "Understand": float,
                        "Apply":      float,
                        "Analyze":    float,
                        "Evaluate":   float,
                        "Create":     float,
                    }
                }

        Raises:
            ValueError: If *question* is an empty string.

        Example:
            >>> clf = BloomTaxonomyClassifier()
            >>> clf.classify_question("How would you redesign the water cycle?")
            {
                'question': 'How would you redesign the water cycle?',
                'level': 'Create',
                'level_index': 5,
                'confidence': 0.812,
                ...
            }
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string.")

        try:
            inputs = self.tokenizer(
                question,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs: torch.Tensor = F.softmax(outputs.logits[0], dim=-1)
            level_index: int = int(probs.argmax().item())
            confidence: float = float(probs[level_index].item())
            level_name: str = self.BLOOM_LEVELS[level_index]

            scores: dict[str, float] = {
                self.BLOOM_LEVELS[i]: round(float(probs[i].item()), 4)
                for i in range(self._NUM_LABELS)
            }

            return {
                "question":    question,
                "level":       level_name,
                "level_index": level_index,
                "confidence":  round(confidence, 4),
                "description": self.LEVEL_DESCRIPTIONS[level_name],
                "scores":      scores,
            }

        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Classification failed for question '%s': %s", question, exc)
            # Return a safe default rather than propagating the exception so
            # that a single bad question doesn't abort a bulk run.
            return {
                "question":    question,
                "level":       "Remember",
                "level_index": 0,
                "confidence":  0.0,
                "description": self.LEVEL_DESCRIPTIONS["Remember"],
                "scores":      {lvl: 0.0 for lvl in self.BLOOM_LEVELS.values()},
            }

    # ------------------------------------------------------------------
    # Batch classification
    # ------------------------------------------------------------------
    def classify_batch(
        self,
        questions: list[str],
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """Classify a list of questions.

        Processes each question individually so that a failure on one item does
        not abort the entire batch.

        Args:
            questions: List of question strings to classify.
            show_progress: When ``True``, display a ``tqdm`` progress bar.
                Defaults to ``True``.

        Returns:
            A list of classification result dictionaries in the same order as
            *questions*.  See :meth:`classify_question` for the dict schema.

        Raises:
            ValueError: If *questions* is empty.

        Example:
            >>> clf = BloomTaxonomyClassifier()
            >>> qs = ["Define osmosis.", "Compare mitosis and meiosis."]
            >>> results = clf.classify_batch(qs)
            >>> [r["level"] for r in results]
            ['Remember', 'Analyze']
        """
        if not questions:
            raise ValueError("questions list must not be empty.")

        results: list[dict[str, Any]] = []

        iterable = questions
        if show_progress:
            try:
                from tqdm import tqdm  # local import keeps the module optional
                iterable = tqdm(questions, desc="Classifying questions", unit="q")
            except ImportError:
                logger.debug("tqdm not installed; progress bar disabled.")

        for question in iterable:
            result = self.classify_question(question)
            results.append(result)

        logger.info("Classified %d question(s).", len(results))
        return results

    # ------------------------------------------------------------------
    # Difficulty score
    # ------------------------------------------------------------------
    def get_level_difficulty_score(self, level: str) -> int:
        """Return a numeric 1–6 difficulty score for a Bloom level name.

        The score maps directly to the cognitive complexity of the level:
        ``Remember`` → 1, …, ``Create`` → 6.

        Args:
            level: A Bloom level name (case-insensitive), e.g. ``"Apply"``.

        Returns:
            An integer in the range ``[1, 6]``.

        Raises:
            ValueError: If *level* is not a recognised Bloom level name.

        Example:
            >>> clf = BloomTaxonomyClassifier()
            >>> clf.get_level_difficulty_score("Evaluate")
            5
        """
        # Build a case-insensitive reverse map: name → 1-based index
        reverse_map: dict[str, int] = {
            name.lower(): idx + 1
            for idx, name in self.BLOOM_LEVELS.items()
        }
        key = level.strip().lower()
        if key not in reverse_map:
            raise ValueError(
                f"Unknown Bloom level '{level}'. "
                f"Valid levels: {list(self.BLOOM_LEVELS.values())}"
            )
        return reverse_map[key]


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SAMPLE_QUESTIONS = [
        "What is photosynthesis?",                              # Remember
        "Explain how vaccines work.",                           # Understand
        "How would you apply the Pythagorean theorem here?",    # Apply
        "Compare and contrast plant and animal cells.",         # Analyze
        "Do you agree with the scientist's conclusion? Why?",   # Evaluate
        "Design an experiment to test water filtration.",       # Create
    ]

    try:
        clf = BloomTaxonomyClassifier()

        print(f"\n{'='*65}")
        print(f"  Model: {clf._model_id}")
        print(f"  Device: {clf.device}")
        print(f"{'='*65}\n")

        results = clf.classify_batch(SAMPLE_QUESTIONS)

        for res in results:
            score = clf.get_level_difficulty_score(res["level"])
            bar = "█" * score + "░" * (6 - score)
            print(f"Q : {res['question']}")
            print(f"    Level      : {res['level']} (index {res['level_index']}, "
                  f"difficulty {score}/6)  [{bar}]")
            print(f"    Confidence : {res['confidence']:.2%}")
            print(f"    Description: {res['description']}")
            print(f"    Scores     : { {k: f'{v:.2%}' for k, v in res['scores'].items()} }\n")

    except Exception as e:
        logger.exception("Bloom classifier pipeline failed: %s", e)
