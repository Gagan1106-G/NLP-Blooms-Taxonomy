"""MCQ Generator module using T5-based question generation model.

This module provides a production-ready pipeline for automatically generating
multiple-choice questions (MCQs) from raw text using HuggingFace Transformers,
NLTK for entity extraction, and WordNet for distractor generation.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

import nltk
import torch
from nltk.corpus import wordnet
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import ssl
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
# Ensure required NLTK data is available
# ---------------------------------------------------------------------------
_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
    ("corpora/words", "words"),
    ("corpora/wordnet", "wordnet"),
]

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded"""
    required_resources = [
        'punkt_tab',
        'averaged_perceptron_tagger_eng', 
        'maxent_ne_chunker_tab',
        'words',
        'wordnet',
        'omw-1.4'
    ]
    
    for name in required_resources:
        try:
            nltk.download(name, quiet=True, raise_on_error=True)
        except:
            pass  # Already downloaded


# ---------------------------------------------------------------------------
# MCQGenerator
# ---------------------------------------------------------------------------
class MCQGenerator:
    """Automatic multiple-choice question generator.

    Uses the *valhalla/t5-base-qg-hl* question-generation model to produce
    natural-language questions from a passage of text, then builds plausible
    distractors via WordNet synonym look-up and entity-based fallback.

    Attributes:
        model_name (str): HuggingFace model identifier.
        device (torch.device): Inference device (``cuda`` or ``cpu``).
        tokenizer (T5Tokenizer): Tokenizer paired with the loaded model.
        model (T5ForConditionalGeneration): Loaded generative model.
        max_input_length (int): Maximum token length for encoder input.
        max_output_length (int): Maximum token length for decoder output.
        num_beams (int): Beam-search width used during generation.

    Example:
        >>> generator = MCQGenerator()
        >>> text = "Photosynthesis is the process by which plants use sunlight..."
        >>> mcqs = generator.generate_mcqs(text, num_questions=3)
        >>> for mcq in mcqs:
        ...     print(mcq["question"])
    """

    _MODEL_NAME: str = "valhalla/t5-base-qg-hl"

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        max_input_length: int = 512,
        max_output_length: int = 64,
        num_beams: int = 4,
    ) -> None:
        """Initialise the MCQ generator and load the generative model.

        Device selection is automatic: CUDA when available, otherwise CPU.

        Args:
            model_name: HuggingFace model identifier or local path.
                Defaults to ``"valhalla/t5-base-qg-hl"``.
            max_input_length: Maximum number of tokens for the encoder.
                Defaults to ``512``.
            max_output_length: Maximum number of tokens for the decoder.
                Defaults to ``64``.
            num_beams: Beam width for beam-search decoding.
                Defaults to ``4``.

        Raises:
            OSError: If the model or tokenizer cannot be loaded from the
                specified ``model_name``.
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Using device: %s", self.device)
        logger.info("Loading model '%s' …", self.model_name)

        try:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model: T5ForConditionalGeneration = (
                T5ForConditionalGeneration.from_pretrained(self.model_name).to(
                    self.device
                )
            )
            self.model.eval()
            logger.info("Model loaded successfully.")
        except OSError as exc:
            logger.error("Failed to load model '%s': %s", self.model_name, exc)
            raise

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------
    def extract_key_entities(self, text: str) -> list[str]:
        """Extract named entities and important noun phrases from *text*.

        Uses NLTK part-of-speech tagging followed by maximum-entropy named-
        entity chunking.  Proper nouns (``NNP``, ``NNPS``) that are *not*
        caught by the chunker are also collected as a fallback.

        Args:
            text: Raw input text from which entities are extracted.

        Returns:
            A deduplicated list of entity strings, preserving their order of
            first appearance.  Returns an empty list when *text* is blank or
            an error occurs.

        Example:
            >>> gen = MCQGenerator()
            >>> gen.extract_key_entities("Albert Einstein was born in Germany.")
            ['Albert Einstein', 'Germany']
        """
        if not text or not text.strip():
            logger.warning("extract_key_entities received empty text.")
            return []

        entities: list[str] = []

        try:
            sentences = nltk.sent_tokenize(text)

            for sentence in sentences:
                tokens = nltk.word_tokenize(sentence)
                tagged = nltk.pos_tag(tokens)
                chunked = nltk.ne_chunk(tagged, binary=False)

                # Named entity chunks
                for subtree in chunked:
                    if isinstance(subtree, nltk.Tree):
                        entity = " ".join(word for word, _tag in subtree.leaves())
                        if entity and entity not in entities:
                            entities.append(entity)

                # Proper nouns not already captured
                for word, tag in tagged:
                    if tag in {"NNP", "NNPS"} and word not in entities:
                        entities.append(word)

        except Exception as exc:  # noqa: BLE001
            logger.error("Entity extraction failed: %s", exc)

        logger.debug("Extracted %d entities.", len(entities))
        return entities

    # ------------------------------------------------------------------
    # Question generation
    # ------------------------------------------------------------------
    def generate_question(self, context: str, answer: str) -> str:
        """Generate a question whose answer is *answer* given *context*.

        Formats the input in the highlight style expected by
        ``valhalla/t5-base-qg-hl`` (``answer: <ans>  context: <ctx>``) and
        decodes the model's output.

        Args:
            context: The passage of text that contains the answer.
            answer: The target answer that the question should address.

        Returns:
            A natural-language question string.  Returns a generic fallback
            string (``"What is <answer>?"``) if generation fails or the
            model returns an empty string.

        Example:
            >>> gen = MCQGenerator()
            >>> gen.generate_question("Water boils at 100 °C.", "100 °C")
            'At what temperature does water boil?'
        """
        fallback = f"What is {answer}?"

        if not context.strip() or not answer.strip():
            logger.warning("generate_question received empty context or answer.")
            return fallback

        try:
            # Highlight the answer span inside the context
            highlighted = context.replace(answer, f"<hl> {answer} <hl>", 1)
            input_text = f"generate question: {highlighted}"

            inputs = self.tokenizer(
                input_text,
                max_length=self.max_input_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_output_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )

            question = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return question if question else fallback

        except Exception as exc:  # noqa: BLE001
            logger.error("Question generation failed for answer '%s': %s", answer, exc)
            return fallback

    # ------------------------------------------------------------------
    # Distractor generation
    # ------------------------------------------------------------------
    def generate_distractors(
        self,
        answer: str,
        all_entities: list[str],
        num_distractors: int = 3,
    ) -> list[str]:
        """Generate plausible wrong-answer options (distractors) for *answer*.

        Strategy (in order of preference):

        1. **WordNet synonyms** – collect synonym lemmas from all synsets of
           every word in *answer*, excluding the answer word itself.
        2. **Entity fallback** – randomly sample from *all_entities*, excluding
           the correct *answer*.

        If fewer than *num_distractors* unique candidates are found, the list
        is padded with generic placeholders (``"Option X"``).

        Args:
            answer: The correct answer string for which distractors are needed.
            all_entities: Pool of other candidate answers extracted from the
                source text, used as a fallback distractor source.
            num_distractors: Number of wrong answers to generate.
                Defaults to ``3``.

        Returns:
            A list of exactly *num_distractors* distractor strings.

        Example:
            >>> gen = MCQGenerator()
            >>> gen.generate_distractors("photosynthesis", ["respiration", "osmosis"])
            ['photosynthesise', 'respiration', 'osmosis']
        """
        distractors: list[str] = []
        answer_lower = answer.lower()

        # ── 1. WordNet synonyms ──────────────────────────────────────────────
        try:
            for word in answer.split():
                for synset in wordnet.synsets(word):
                    for lemma in synset.lemmas():
                        candidate = lemma.name().replace("_", " ")
                        if (
                            candidate.lower() != answer_lower
                            and candidate not in distractors
                            and len(candidate) > 1
                        ):
                            distractors.append(candidate)
                            if len(distractors) >= num_distractors:
                                break
                    if len(distractors) >= num_distractors:
                        break
        except Exception as exc:  # noqa: BLE001
            logger.warning("WordNet distractor generation failed: %s", exc)

        # ── 2. Entity fallback ───────────────────────────────────────────────
        if len(distractors) < num_distractors:
            entity_pool = [
                e
                for e in all_entities
                if e.lower() != answer_lower and e not in distractors
            ]
            random.shuffle(entity_pool)
            needed = num_distractors - len(distractors)
            distractors.extend(entity_pool[:needed])

        # ── 3. Generic placeholder padding ──────────────────────────────────
        while len(distractors) < num_distractors:
            placeholder = f"Option {len(distractors) + 1}"
            if placeholder not in distractors:
                distractors.append(placeholder)

        return distractors[:num_distractors]

    # ------------------------------------------------------------------
    # Difficulty estimation
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_difficulty(context: str, answer: str) -> str:
        """Heuristically estimate question difficulty.

        Difficulty is based on the word count of the surrounding context and
        the character length of the answer token.

        Args:
            context: Source sentence or paragraph for the question.
            answer: The answer string.

        Returns:
            One of ``"Easy"``, ``"Medium"``, or ``"Hard"``.
        """
        word_count = len(context.split())
        answer_len = len(answer.split())

        if word_count < 20 and answer_len == 1:
            return "Easy"
        if word_count < 50 and answer_len <= 3:
            return "Medium"
        return "Hard"

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def generate_mcqs(
        self,
        text: str,
        num_questions: int = 5,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate multiple-choice questions from *text*.

        Full pipeline:

        1. Extract key entities from the input text.
        2. For each candidate entity (up to *num_questions*):

           a. Find the sentence in *text* that contains the entity.
           b. Generate a question conditioned on that sentence.
           c. Generate 3 distractors.
           d. Shuffle the four options and record the correct label.

        Args:
            text: Source text from which MCQs are generated.  Should be at
                least a few sentences long for best results.
            num_questions: Maximum number of MCQs to generate.
                Defaults to ``5``.
            seed: Optional random seed for reproducible option shuffling.
                Defaults to ``None``.

        Returns:
            A list of MCQ dictionaries, each with the following keys:

            .. code-block:: python

                {
                    "question":     str,          # generated question text
                    "options":      list[str],     # 4 shuffled answer options
                    "answer":       str,           # correct option text
                    "answer_label": str,           # correct option label ('A'–'D')
                    "difficulty":   str,           # 'Easy' | 'Medium' | 'Hard'
                    "option_labels": list[str],    # always ['A','B','C','D']
                }

            Returns an empty list if no entities are found or all generation
            attempts fail.

        Raises:
            ValueError: If *num_questions* is not a positive integer.

        Example:
            >>> gen = MCQGenerator()
            >>> text = (
            ...     "The mitochondria is the powerhouse of the cell. "
            ...     "It produces ATP through cellular respiration."
            ... )
            >>> mcqs = gen.generate_mcqs(text, num_questions=2, seed=42)
            >>> len(mcqs)
            2
        """
        if num_questions < 1:
            raise ValueError(f"num_questions must be ≥ 1, got {num_questions}.")

        if not text or not text.strip():
            logger.error("generate_mcqs received empty text.")
            return []

        if seed is not None:
            random.seed(seed)

        logger.info("Extracting entities from input text …")
        entities = self.extract_key_entities(text)

        if not entities:
            logger.warning("No entities found in the provided text.")
            return []

        logger.info("Found %d candidate entities; generating up to %d MCQs …",
                    len(entities), num_questions)

        # Pre-tokenise sentences for context look-up
        sentences = nltk.sent_tokenize(text)

        mcqs: list[dict[str, Any]] = []
        label_map = ["A", "B", "C", "D"]

        for answer in tqdm(
            entities[:num_questions],
            desc="Generating MCQs",
            unit="question",
        ):
            try:
                # Find the sentence that best contains the answer
                context = next(
                    (s for s in sentences if answer.lower() in s.lower()),
                    text[:512],
                )

                question = self.generate_question(context, answer)

                if not question or question == f"What is {answer}?":
                    # Skip trivially fallback questions if a real one is preferred
                    logger.debug("Skipping trivial question for answer '%s'.", answer)

                distractors = self.generate_distractors(answer, entities)
                difficulty = self._estimate_difficulty(context, answer)

                # Build and shuffle options
                options: list[str] = [answer] + distractors
                random.shuffle(options)

                correct_index = options.index(answer)
                correct_label = label_map[correct_index]

                mcqs.append(
                    {
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "answer_label": correct_label,
                        "difficulty": difficulty,
                        "option_labels": label_map,
                    }
                )

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to generate MCQ for answer '%s': %s", answer, exc
                )
                continue

        logger.info("Successfully generated %d MCQ(s).", len(mcqs))
        return mcqs


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SAMPLE_TEXT = """
    Photosynthesis is the process by which green plants, algae, and some bacteria
    convert light energy—usually from the Sun—into chemical energy stored as glucose.
    This process occurs primarily in the chloroplasts of plant cells, which contain
    a green pigment called chlorophyll. Chlorophyll absorbs sunlight and uses its
    energy to convert carbon dioxide and water into glucose and oxygen.
    The overall chemical equation for photosynthesis is:
    6CO2 + 6H2O + light energy → C6H12O6 + 6O2.
    Photosynthesis is essential for life on Earth because it produces oxygen and
    forms the base of most food chains.
    """

    try:
        generator = MCQGenerator()
        mcqs = generator.generate_mcqs(SAMPLE_TEXT, num_questions=5, seed=42)

        print(f"\n{'='*60}")
        print(f"  Generated {len(mcqs)} MCQ(s)")
        print(f"{'='*60}\n")

        for i, mcq in enumerate(mcqs, start=1):
            print(f"Q{i} [{mcq['difficulty']}]: {mcq['question']}")
            for label, option in zip(mcq["option_labels"], mcq["options"]):
                marker = "✓" if label == mcq["answer_label"] else " "
                print(f"  {label}. {option}  {marker}")
            print(f"  Answer: {mcq['answer_label']}. {mcq['answer']}\n")

    except Exception as e:
        logger.exception("MCQ generation pipeline failed: %s", e)
