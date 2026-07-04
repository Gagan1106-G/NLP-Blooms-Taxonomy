"""RAG (Retrieval-Augmented Generation) Retriever module.

Provides fast, embedding-based passage retrieval using sentence-transformers
for dense encoding and FAISS for approximate nearest-neighbour search.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
# RAGRetriever
# ---------------------------------------------------------------------------
class RAGRetriever:
    """Embedding-based passage retriever backed by FAISS.

    Splits a source document into overlapping text chunks, encodes them with
    a ``sentence-transformers`` model, stores the resulting vectors in a FAISS
    flat L2 index, and answers retrieval queries in sub-millisecond time even
    for large corpora.

    Attributes:
        model_name (str): Sentence-transformer model identifier.
        model (SentenceTransformer): Loaded embedding model.
        passages (list[str]): Indexed passage texts, set after
            :meth:`index_passages`.
        index (faiss.Index | None): FAISS index, set after
            :meth:`index_passages`.
        embedding_dim (int): Dimensionality of the embedding space.

    Example:
        >>> retriever = RAGRetriever()
        >>> retriever.index_passages(long_text)
        >>> results = retriever.retrieve_relevant_passages("What is ATP?", top_k=3)
        >>> for passage, score in results:
        ...     print(f"[{score:.4f}] {passage[:80]}")
    """

    _DEFAULT_MODEL: str = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        """Initialise the retriever and load the sentence-transformer model.

        Args:
            model_name: HuggingFace / sentence-transformers model identifier.
                Defaults to ``"all-MiniLM-L6-v2"``.

        Raises:
            OSError: If the embedding model cannot be loaded.
        """
        self.model_name = model_name
        self.passages: list[str] = []
        self.index: faiss.Index | None = None
        self.embedding_dim: int = 0

        logger.info("Loading embedding model '%s' …", self.model_name)
        try:
            self.model = SentenceTransformer(self.model_name)
            # Determine embedding dimension from a probe encoding
            probe: np.ndarray = self.model.encode(["probe"], show_progress_bar=False)
            self.embedding_dim = probe.shape[1]
            logger.info(
                "Model loaded. Embedding dimension: %d", self.embedding_dim
            )
        except OSError as exc:
            logger.error("Failed to load model '%s': %s", self.model_name, exc)
            raise

    # ------------------------------------------------------------------
    # Text splitting
    # ------------------------------------------------------------------
    def split_into_passages(
        self,
        text: str,
        chunk_size: int = 250,
        overlap: int = 50,
    ) -> list[str]:
        """Split *text* into overlapping word-level chunks.

        Words (whitespace-delimited tokens) are used as the unit of measure
        so that no word is split mid-token.  Consecutive chunks share *overlap*
        words to preserve sentence context across boundaries.

        Args:
            text: Raw input text to split.
            chunk_size: Number of words per chunk.  Defaults to ``250``.
            overlap: Number of words shared between consecutive chunks.
                Must be strictly less than *chunk_size*.  Defaults to ``50``.

        Returns:
            A list of passage strings.  Returns ``[]`` when *text* is blank.

        Raises:
            ValueError: If *overlap* ≥ *chunk_size*.

        Example:
            >>> retriever = RAGRetriever()
            >>> passages = retriever.split_into_passages(text, chunk_size=100, overlap=20)
            >>> len(passages)
            14
        """
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})."
            )

        text = text.strip()
        if not text:
            logger.warning("split_into_passages received empty text.")
            return []

        # Normalise whitespace
        text = re.sub(r"\s+", " ", text)
        words = text.split()

        if len(words) <= chunk_size:
            return [text]

        passages: list[str] = []
        step = chunk_size - overlap
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            passages.append(chunk)
            if end == len(words):
                break
            start += step

        logger.debug(
            "Split %d words into %d passages (chunk_size=%d, overlap=%d).",
            len(words),
            len(passages),
            chunk_size,
            overlap,
        )
        return passages

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_passages(
        self,
        text: str,
        chunk_size: int = 250,
        overlap: int = 50,
        batch_size: int = 64,
    ) -> int:
        """Encode all passages from *text* and build a FAISS index.

        Splits *text* into overlapping chunks, encodes them in batches with a
        progress bar, and stores the resulting vectors in a ``faiss.IndexFlatIP``
        (inner-product / cosine similarity) index after L2 normalisation.

        Args:
            text: Source document text to index.
            chunk_size: Words per passage chunk.  Defaults to ``250``.
            overlap: Overlapping words between chunks.  Defaults to ``50``.
            batch_size: Number of passages encoded per forward pass.
                Defaults to ``64``.

        Returns:
            The number of passages successfully indexed.

        Raises:
            ValueError: If *text* is empty or no passages can be created.

        Example:
            >>> n = retriever.index_passages(document_text)
            >>> print(f"Indexed {n} passages.")
            Indexed 42 passages.
        """
        if not text or not text.strip():
            raise ValueError("index_passages: text must not be empty.")

        self.passages = self.split_into_passages(text, chunk_size, overlap)

        if not self.passages:
            raise ValueError("index_passages: no passages generated from the input text.")

        logger.info(
            "Encoding %d passages (batch_size=%d) …", len(self.passages), batch_size
        )

        all_embeddings: list[np.ndarray] = []

        for i in tqdm(
            range(0, len(self.passages), batch_size),
            desc="Indexing passages",
            unit="batch",
        ):
            batch = self.passages[i : i + batch_size]
            embeddings: np.ndarray = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,   # enables cosine sim via inner product
            )
            all_embeddings.append(embeddings)

        matrix = np.vstack(all_embeddings).astype("float32")

        # Inner-product index (cosine sim after L2 normalisation)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(matrix)  # type: ignore[arg-type]

        logger.info("FAISS index built with %d vectors.", self.index.ntotal)
        return len(self.passages)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve_relevant_passages(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Retrieve the *top_k* most relevant passages for *query*.

        Encodes *query*, normalises the vector, and performs an approximate
        nearest-neighbour search against the FAISS index.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of passages to return.  Defaults to ``5``.

        Returns:
            A list of ``(passage_text, similarity_score)`` tuples sorted by
            descending similarity.  Similarity is the cosine similarity in
            ``[−1, 1]`` (inner product of L2-normalised vectors).
            Returns ``[]`` if the index is empty or *query* is blank.

        Raises:
            RuntimeError: If :meth:`index_passages` has not been called yet.

        Example:
            >>> results = retriever.retrieve_relevant_passages("cellular respiration", top_k=3)
            >>> passage, score = results[0]
            >>> print(f"Score: {score:.4f}  →  {passage[:60]}")
        """
        if self.index is None:
            raise RuntimeError(
                "No FAISS index found. Call index_passages() before retrieving."
            )

        if not query or not query.strip():
            logger.warning("retrieve_relevant_passages received an empty query.")
            return []

        k = min(top_k, len(self.passages))

        try:
            query_vec: np.ndarray = self.model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

            scores, indices = self.index.search(query_vec, k)  # type: ignore[misc]

            results: list[tuple[str, float]] = [
                (self.passages[int(idx)], float(scores[0][rank]))
                for rank, idx in enumerate(indices[0])
                if idx != -1
            ]

            logger.debug(
                "Retrieved %d passage(s) for query: '%s'", len(results), query[:60]
            )
            return results

        except Exception as exc:  # noqa: BLE001
            logger.error("Retrieval failed for query '%s': %s", query, exc)
            return []

    # ------------------------------------------------------------------
    # Answer-aware retrieval
    # ------------------------------------------------------------------
    def get_answer_relevant_passages(
        self,
        answer: str,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Get passages that are most likely to contain *answer*.

        Combines two signals:

        1. **Exact substring match** – passages that literally contain *answer*
           (case-insensitive) are prioritised and given a synthetic score of
           ``1.0``.
        2. **Semantic similarity** – remaining slots are filled from a standard
           semantic search against *answer*.

        Args:
            answer: The answer string to search for within the indexed passages.
            top_k: Maximum number of passages to return.  Defaults to ``3``.

        Returns:
            A list of ``(passage_text, similarity_score)`` tuples sorted by
            descending score.  Returns ``[]`` if the index is empty.

        Raises:
            RuntimeError: If :meth:`index_passages` has not been called yet.

        Example:
            >>> passages = retriever.get_answer_relevant_passages("ATP", top_k=2)
            >>> for p, s in passages:
            ...     print(s, p[:80])
        """
        if self.index is None:
            raise RuntimeError(
                "No FAISS index found. Call index_passages() before retrieving."
            )

        if not answer or not answer.strip():
            logger.warning("get_answer_relevant_passages received an empty answer.")
            return []

        answer_lower = answer.lower()
        results: list[tuple[str, float]] = []
        seen: set[str] = set()

        # ── 1. Exact match priority ──────────────────────────────────────────
        for passage in self.passages:
            if answer_lower in passage.lower() and passage not in seen:
                results.append((passage, 1.0))
                seen.add(passage)
                if len(results) >= top_k:
                    break

        # ── 2. Semantic similarity fill ──────────────────────────────────────
        if len(results) < top_k:
            remaining = top_k - len(results)
            semantic = self.retrieve_relevant_passages(answer, top_k=top_k + len(seen))
            for passage, score in semantic:
                if passage not in seen:
                    results.append((passage, score))
                    seen.add(passage)
                    if len(results) >= top_k:
                        break

        logger.debug(
            "get_answer_relevant_passages: %d result(s) for answer '%s'.",
            len(results),
            answer,
        )
        return results[:top_k]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def is_indexed(self) -> bool:
        """Return ``True`` if passages have been indexed."""
        return self.index is not None and len(self.passages) > 0

    def get_stats(self) -> dict[str, Any]:
        """Return a summary of the current index state.

        Returns:
            Dictionary with keys ``model_name``, ``num_passages``,
            ``embedding_dim``, ``index_type``, and ``is_indexed``.
        """
        return {
            "model_name":    self.model_name,
            "num_passages":  len(self.passages),
            "embedding_dim": self.embedding_dim,
            "index_type":    type(self.index).__name__ if self.index else None,
            "is_indexed":    self.is_indexed,
        }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SAMPLE_DOCUMENT = """
    Cellular respiration is the process by which cells break down glucose and
    other organic molecules to produce ATP (adenosine triphosphate), the primary
    energy currency of the cell. This process occurs in three main stages:
    glycolysis, the Krebs cycle, and oxidative phosphorylation.

    Glycolysis takes place in the cytoplasm and splits one molecule of glucose
    into two molecules of pyruvate, producing a net gain of 2 ATP and 2 NADH.
    This stage does not require oxygen and is therefore anaerobic.

    The Krebs cycle (also called the citric acid cycle) occurs in the mitochondrial
    matrix. Pyruvate is converted to acetyl-CoA, which enters the cycle and is
    progressively oxidised. Each turn of the cycle produces 3 NADH, 1 FADH2,
    1 ATP (or GTP), and 2 CO2.

    Oxidative phosphorylation takes place on the inner mitochondrial membrane.
    NADH and FADH2 donate electrons to the electron transport chain, driving
    the synthesis of approximately 32–34 ATP molecules via ATP synthase. Oxygen
    acts as the final electron acceptor, combining with hydrogen ions to form water.

    The overall equation for aerobic cellular respiration is:
    C6H12O6 + 6O2 → 6CO2 + 6H2O + ~36–38 ATP.

    Photosynthesis is essentially the reverse process, occurring in chloroplasts.
    Plants use light energy, water, and carbon dioxide to synthesise glucose and
    release oxygen. Chlorophyll, the primary photosynthetic pigment, absorbs light
    mainly in the blue and red wavelengths.
    """

    try:
        retriever = RAGRetriever()

        print("\n" + "=" * 65)
        print("  Step 1 – Index document")
        print("=" * 65)
        n = retriever.index_passages(SAMPLE_DOCUMENT, chunk_size=80, overlap=20)
        print(f"\nIndexed {n} passages.")
        print(f"Stats: {retriever.get_stats()}\n")

        print("=" * 65)
        print("  Step 2 – Semantic retrieval")
        print("=" * 65)
        query = "How does oxidative phosphorylation produce ATP?"
        print(f"\nQuery: {query}\n")
        for passage, score in retriever.retrieve_relevant_passages(query, top_k=3):
            print(f"  [{score:.4f}] {passage[:100]} …")

        print("\n" + "=" * 65)
        print("  Step 3 – Answer-aware retrieval")
        print("=" * 65)
        answer = "glycolysis"
        print(f"\nAnswer term: '{answer}'\n")
        for passage, score in retriever.get_answer_relevant_passages(answer, top_k=2):
            print(f"  [{score:.4f}] {passage[:100]} …")

    except Exception as e:
        logger.exception("RAGRetriever pipeline failed: %s", e)
