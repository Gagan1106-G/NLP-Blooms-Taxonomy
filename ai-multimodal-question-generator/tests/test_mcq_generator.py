"""Unit tests for MCQGenerator.

Tests cover initialisation, entity extraction, question generation,
distractor quality, batch MCQ generation, and error-handling paths.
Heavy model I/O is patched with lightweight mocks so the suite runs in
seconds without GPU/network access.

Run with:
    pytest tests/test_mcq_generator.py -v
    pytest tests/test_mcq_generator.py -v --tb=short   # terse traceback
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mcq_generator.generator import MCQGenerator

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
SAMPLE_TEXT = (
    "Photosynthesis is the process by which green plants convert sunlight into "
    "glucose. This occurs in the chloroplasts of plant cells. Chlorophyll absorbs "
    "light energy and uses it to combine carbon dioxide and water into glucose. "
    "Oxygen is released as a by-product of the reaction. Plants depend on "
    "photosynthesis for their energy needs."
)

SHORT_TEXT = "Short text."

ENTITY_TEXT = "Albert Einstein was born in Ulm, Germany. He developed the theory of relativity."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mock_tokenizer() -> MagicMock:
    """Lightweight mock for T5Tokenizer."""
    tok = MagicMock()
    # encode → returns a mock that has .to() and can be unpacked as **kwargs
    encoded = MagicMock()
    encoded.to.return_value = encoded
    # Make it behave like a dict for **inputs unpacking
    encoded.__iter__ = MagicMock(return_value=iter([]))
    encoded.keys.return_value = []
    tok.return_value = encoded
    # decode returns a plausible question string
    tok.decode.return_value = "What is the process of photosynthesis?"
    return tok


@pytest.fixture(scope="module")
def mock_model() -> MagicMock:
    """Lightweight mock for T5ForConditionalGeneration."""
    mdl = MagicMock()
    # generate → returns a single-item list of tensor-like ids
    mdl.generate.return_value = [torch.tensor([0, 1, 2])]
    mdl.eval.return_value = None
    return mdl


@pytest.fixture(scope="module")
def generator(mock_tokenizer: MagicMock, mock_model: MagicMock) -> MCQGenerator:
    """MCQGenerator with mocked tokenizer and model – avoids network/GPU."""
    with (
        patch("src.mcq_generator.generator.T5Tokenizer.from_pretrained",
              return_value=mock_tokenizer),
        patch(
            "src.mcq_generator.generator.T5ForConditionalGeneration.from_pretrained",
            return_value=mock_model,
        ),
    ):
        gen = MCQGenerator(
            model_name="mock-t5",
            max_input_length=512,
            max_output_length=64,
            num_beams=2,
        )
    return gen


# ---------------------------------------------------------------------------
# 1. Initialisation tests
# ---------------------------------------------------------------------------
class TestInitialization:
    """Verify MCQGenerator initialises with the expected attributes."""

    def test_model_name_stored(self, generator: MCQGenerator) -> None:
        assert generator.model_name == "mock-t5"

    def test_device_is_torch_device(self, generator: MCQGenerator) -> None:
        assert isinstance(generator.device, torch.device)

    def test_device_valid_value(self, generator: MCQGenerator) -> None:
        assert generator.device.type in {"cpu", "cuda"}

    def test_max_input_length(self, generator: MCQGenerator) -> None:
        assert generator.max_input_length == 512

    def test_max_output_length(self, generator: MCQGenerator) -> None:
        assert generator.max_output_length == 64

    def test_num_beams(self, generator: MCQGenerator) -> None:
        assert generator.num_beams == 2

    def test_tokenizer_not_none(self, generator: MCQGenerator) -> None:
        assert generator.tokenizer is not None

    def test_model_not_none(self, generator: MCQGenerator) -> None:
        assert generator.model is not None

    def test_model_set_to_eval(self, generator: MCQGenerator) -> None:
        generator.model.eval.assert_called()

    def test_os_error_on_bad_model_name(self) -> None:
        with (
            patch(
                "src.mcq_generator.generator.T5Tokenizer.from_pretrained",
                side_effect=OSError("model not found"),
            ),
            pytest.raises(OSError, match="model not found"),
        ):
            MCQGenerator(model_name="nonexistent/model-xyz")


# ---------------------------------------------------------------------------
# 2. Entity extraction tests
# ---------------------------------------------------------------------------
class TestEntityExtraction:
    """Verify extract_key_entities returns sensible results."""

    def test_returns_list(self, generator: MCQGenerator) -> None:
        result = generator.extract_key_entities(ENTITY_TEXT)
        assert isinstance(result, list)

    def test_returns_strings(self, generator: MCQGenerator) -> None:
        result = generator.extract_key_entities(ENTITY_TEXT)
        assert all(isinstance(e, str) for e in result)

    def test_no_duplicates(self, generator: MCQGenerator) -> None:
        result = generator.extract_key_entities(ENTITY_TEXT)
        assert len(result) == len(set(result)), "Duplicate entities returned"

    def test_proper_nouns_captured(self, generator: MCQGenerator) -> None:
        """At least one clearly proper noun should appear."""
        result = generator.extract_key_entities(ENTITY_TEXT)
        lowered = [e.lower() for e in result]
        assert any("einstein" in e or "germany" in e or "ulm" in e for e in lowered), (
            f"Expected at least one of 'Einstein/Germany/Ulm' in {result}"
        )

    def test_empty_string_returns_empty_list(self, generator: MCQGenerator) -> None:
        assert generator.extract_key_entities("") == []

    def test_whitespace_only_returns_empty_list(self, generator: MCQGenerator) -> None:
        assert generator.extract_key_entities("   \n\t  ") == []

    def test_plain_sentence_no_crash(self, generator: MCQGenerator) -> None:
        """Ensure no exception for a sentence with no proper nouns."""
        result = generator.extract_key_entities("the quick brown fox jumps over the lazy dog")
        assert isinstance(result, list)

    def test_longer_text_returns_entities(self, generator: MCQGenerator) -> None:
        result = generator.extract_key_entities(SAMPLE_TEXT)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 3. Question generation tests
# ---------------------------------------------------------------------------
class TestQuestionGeneration:
    """Verify generate_question output format and behaviour."""

    def test_returns_string(self, generator: MCQGenerator) -> None:
        q = generator.generate_question(SAMPLE_TEXT, "photosynthesis")
        assert isinstance(q, str)

    def test_non_empty_output(self, generator: MCQGenerator) -> None:
        q = generator.generate_question(SAMPLE_TEXT, "photosynthesis")
        assert len(q.strip()) > 0

    def test_ends_with_question_mark_or_text(self, generator: MCQGenerator) -> None:
        """Question should be meaningful; we don't enforce '?' strictly."""
        q = generator.generate_question(SAMPLE_TEXT, "chlorophyll")
        assert isinstance(q, str) and len(q) > 3

    def test_empty_context_returns_fallback(self, generator: MCQGenerator) -> None:
        q = generator.generate_question("", "glucose")
        assert "glucose" in q.lower()

    def test_empty_answer_returns_fallback(self, generator: MCQGenerator) -> None:
        q = generator.generate_question(SAMPLE_TEXT, "")
        # fallback is "What is ?"
        assert isinstance(q, str)

    def test_fallback_contains_answer(self, generator: MCQGenerator) -> None:
        """When model mock returns empty string the fallback includes the answer."""
        generator.tokenizer.decode.return_value = ""
        q = generator.generate_question(SAMPLE_TEXT, "mitochondria")
        assert "mitochondria" in q
        # Restore for other tests
        generator.tokenizer.decode.return_value = "What is the process of photosynthesis?"

    def test_model_generate_called(self, generator: MCQGenerator) -> None:
        generator.generate_question(SAMPLE_TEXT, "sunlight")
        generator.model.generate.assert_called()


# ---------------------------------------------------------------------------
# 4. Distractor quality tests
# ---------------------------------------------------------------------------
class TestDistractorQuality:
    """Verify distractors are plausible, unique, and different from the answer."""

    @pytest.fixture()
    def entities(self) -> list[str]:
        return ["glucose", "oxygen", "carbon dioxide", "water", "chlorophyll"]

    def test_returns_list(self, generator: MCQGenerator, entities: list[str]) -> None:
        d = generator.generate_distractors("photosynthesis", entities)
        assert isinstance(d, list)

    def test_correct_count_default(self, generator: MCQGenerator, entities: list[str]) -> None:
        d = generator.generate_distractors("photosynthesis", entities)
        assert len(d) == 3

    def test_correct_count_custom(self, generator: MCQGenerator, entities: list[str]) -> None:
        d = generator.generate_distractors("photosynthesis", entities, num_distractors=2)
        assert len(d) == 2

    def test_answer_not_in_distractors(self, generator: MCQGenerator, entities: list[str]) -> None:
        answer = "photosynthesis"
        d = generator.generate_distractors(answer, entities)
        assert answer not in d, "Correct answer should not appear as a distractor"

    def test_no_duplicate_distractors(self, generator: MCQGenerator, entities: list[str]) -> None:
        d = generator.generate_distractors("photosynthesis", entities)
        assert len(d) == len(set(d)), "Distractors contain duplicates"

    def test_all_distractors_are_strings(
        self, generator: MCQGenerator, entities: list[str]
    ) -> None:
        d = generator.generate_distractors("photosynthesis", entities)
        assert all(isinstance(x, str) for x in d)

    def test_fallback_when_no_entities(self, generator: MCQGenerator) -> None:
        """With empty entity pool and no WordNet hits, returns placeholders."""
        d = generator.generate_distractors("xyzzy_nonsense_word_12345", [])
        assert len(d) == 3
        assert all(isinstance(x, str) and len(x) > 0 for x in d)

    def test_distractors_case_insensitive_exclude(
        self, generator: MCQGenerator
    ) -> None:
        """Answer check must be case-insensitive."""
        d = generator.generate_distractors("Photosynthesis", ["photosynthesis", "glucose"])
        for dist in d:
            assert dist.lower() != "photosynthesis"


# ---------------------------------------------------------------------------
# 5. Batch MCQ generation tests
# ---------------------------------------------------------------------------
class TestBatchGeneration:
    """Verify generate_mcqs output structure and integrity."""

    @pytest.fixture(scope="class")
    def mcqs(self, generator: MCQGenerator) -> list[dict[str, Any]]:
        return generator.generate_mcqs(SAMPLE_TEXT, num_questions=3, seed=42)

    def test_returns_list(self, mcqs: list[dict]) -> None:
        assert isinstance(mcqs, list)

    def test_respects_num_questions(self, mcqs: list[dict]) -> None:
        assert len(mcqs) <= 3

    def test_each_item_is_dict(self, mcqs: list[dict]) -> None:
        for mcq in mcqs:
            assert isinstance(mcq, dict)

    def test_required_keys_present(self, mcqs: list[dict]) -> None:
        required = {"question", "options", "answer", "answer_label", "difficulty", "option_labels"}
        for mcq in mcqs:
            assert required.issubset(mcq.keys()), (
                f"MCQ missing keys: {required - mcq.keys()}"
            )

    def test_question_is_non_empty_string(self, mcqs: list[dict]) -> None:
        for mcq in mcqs:
            assert isinstance(mcq["question"], str)
            assert len(mcq["question"].strip()) > 0

    def test_options_count(self, mcqs: list[dict]) -> None:
        for mcq in mcqs:
            assert len(mcq["options"]) == 4, (
                f"Expected 4 options, got {len(mcq['options'])}"
            )

    def test_options_are_strings(self, mcqs: list[dict]) -> None:
        for mcq in mcqs:
            assert all(isinstance(o, str) for o in mcq["options"])

    def test_answer_in_options(self, mcqs: list[dict]) -> None:
        for mcq in mcqs:
            assert mcq["answer"] in mcq["options"], (
                f"Correct answer '{mcq['answer']}' not found in options {mcq['options']}"
            )

    def test_answer_label_valid(self, mcqs: list[dict]) -> None:
        valid_labels = {"A", "B", "C", "D"}
        for mcq in mcqs:
            assert mcq["answer_label"] in valid_labels

    def test_answer_label_matches_position(self, mcqs: list[dict]) -> None:
        label_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        for mcq in mcqs:
            idx = label_to_idx[mcq["answer_label"]]
            assert mcq["options"][idx] == mcq["answer"], (
                f"Label '{mcq['answer_label']}' does not point to '{mcq['answer']}'"
            )

    def test_option_labels_field(self, mcqs: list[dict]) -> None:
        for mcq in mcqs:
            assert mcq["option_labels"] == ["A", "B", "C", "D"]

    def test_difficulty_valid_value(self, mcqs: list[dict]) -> None:
        valid = {"Easy", "Medium", "Hard"}
        for mcq in mcqs:
            assert mcq["difficulty"] in valid

    def test_no_duplicate_options(self, mcqs: list[dict]) -> None:
        for mcq in mcqs:
            opts = [o.lower() for o in mcq["options"]]
            assert len(opts) == len(set(opts)), (
                f"Duplicate options found: {mcq['options']}"
            )

    def test_seed_reproducibility(self, generator: MCQGenerator) -> None:
        run_a = generator.generate_mcqs(SAMPLE_TEXT, num_questions=2, seed=0)
        run_b = generator.generate_mcqs(SAMPLE_TEXT, num_questions=2, seed=0)
        if run_a and run_b:
            assert run_a[0]["options"] == run_b[0]["options"], (
                "Same seed should produce same option ordering"
            )


# ---------------------------------------------------------------------------
# 6. Error handling tests
# ---------------------------------------------------------------------------
class TestErrorHandling:
    """Verify the generator handles invalid / edge-case inputs gracefully."""

    def test_empty_text_returns_empty_list(self, generator: MCQGenerator) -> None:
        result = generator.generate_mcqs("", num_questions=3)
        assert result == []

    def test_whitespace_only_returns_empty_list(self, generator: MCQGenerator) -> None:
        result = generator.generate_mcqs("   \n  ", num_questions=3)
        assert result == []

    def test_num_questions_zero_raises(self, generator: MCQGenerator) -> None:
        with pytest.raises(ValueError):
            generator.generate_mcqs(SAMPLE_TEXT, num_questions=0)

    def test_num_questions_negative_raises(self, generator: MCQGenerator) -> None:
        with pytest.raises(ValueError):
            generator.generate_mcqs(SAMPLE_TEXT, num_questions=-1)

    def test_text_with_no_entities_returns_empty(self, generator: MCQGenerator) -> None:
        """Text that yields no entities should return an empty list."""
        with patch.object(generator, "extract_key_entities", return_value=[]):
            result = generator.generate_mcqs(SAMPLE_TEXT, num_questions=3)
        assert result == []

    def test_generate_question_model_exception_returns_fallback(
        self, generator: MCQGenerator
    ) -> None:
        """If model.generate raises, the method should return the fallback string."""
        generator.model.generate.side_effect = RuntimeError("mock GPU OOM")
        q = generator.generate_question(SAMPLE_TEXT, "oxygen")
        assert isinstance(q, str)
        assert "oxygen" in q  # fallback contains answer
        # Reset side_effect
        generator.model.generate.side_effect = None
        generator.model.generate.return_value = [torch.tensor([0, 1, 2])]

    def test_single_char_text_no_crash(self, generator: MCQGenerator) -> None:
        result = generator.generate_mcqs("A", num_questions=1)
        assert isinstance(result, list)

    def test_very_long_text_no_crash(self, generator: MCQGenerator) -> None:
        long_text = SAMPLE_TEXT * 30
        result = generator.generate_mcqs(long_text, num_questions=2)
        assert isinstance(result, list)

    def test_num_questions_larger_than_entities(self, generator: MCQGenerator) -> None:
        """Asking for more questions than entities should not crash."""
        with patch.object(generator, "extract_key_entities", return_value=["Oxygen"]):
            result = generator.generate_mcqs(SAMPLE_TEXT, num_questions=10)
        assert isinstance(result, list)
        assert len(result) <= 1


# ---------------------------------------------------------------------------
# 7. Difficulty estimation tests
# ---------------------------------------------------------------------------
class TestDifficultyEstimation:
    """Verify the static difficulty heuristic boundaries."""

    def test_short_single_word_answer_is_easy(self) -> None:
        context = "Water boils at one hundred degrees."  # 6 words < 20
        assert MCQGenerator._estimate_difficulty(context, "water") == "Easy"

    def test_medium_length_context(self) -> None:
        context = " ".join(["word"] * 30)  # 30 words
        assert MCQGenerator._estimate_difficulty(context, "word") == "Medium"

    def test_long_context_is_hard(self) -> None:
        context = " ".join(["word"] * 60)  # 60 words
        answer  = "multi word answer"        # 3 words
        assert MCQGenerator._estimate_difficulty(context, answer) == "Hard"

    def test_returns_string(self) -> None:
        result = MCQGenerator._estimate_difficulty("some context text here", "text")
        assert isinstance(result, str)

    def test_valid_difficulty_values(self) -> None:
        valid = {"Easy", "Medium", "Hard"}
        for ctx_len in [5, 30, 80]:
            ctx = " ".join(["w"] * ctx_len)
            assert MCQGenerator._estimate_difficulty(ctx, "w") in valid
