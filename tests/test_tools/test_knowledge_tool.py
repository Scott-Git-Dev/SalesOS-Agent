"""
tests/test_tools/test_knowledge_tool.py

Unit tests for tools/knowledge_tool.py.

What we test:
  - Vectorstore initialises only once (singleton pattern)
  - Score filter drops low-relevance results  
  - High-relevance results are returned and formatted correctly
  - Missing ChromaDB path returns a helpful error
  - Tool handles vectorstore exceptions gracefully
  - Semantic accuracy: known queries find seeded documents (needs tmp_chroma_db)
"""

import os
import pytest
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_doc(content: str, source: str = "kb/test.md"):
    from langchain_core.documents import Document
    return Document(page_content=content, metadata={"source": source})


def _make_vectorstore(doc_score_pairs):
    """Return a mock vectorstore whose similarity_search_with_score returns given pairs."""
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = doc_score_pairs
    return vs


# ═══════════════════════════════════════════════════════════════════════════
# 1. Score filtering
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreFiltering:
    """
    Scores below MIN_RAG_SCORE threshold should be included;
    scores above the threshold should be dropped.
    (Distance metric: lower = more similar)
    """

    def _search(self, pairs, min_score=0.4):
        import tools.knowledge_tool as kt
        kt._VECTORSTORE = _make_vectorstore(pairs)

        with patch("tools.knowledge_tool.os.path.exists", return_value=True), \
             patch("tools.knowledge_tool.MIN_RAG_SCORE", min_score):
            return kt.search_local_docs.invoke({"query": "test"})

    def test_all_relevant_docs_returned(self):
        pairs = [
            (_make_doc("Revenue target is $15M"), 0.15),
            (_make_doc("Discount policy details"),  0.30),
        ]
        result = self._search(pairs)
        assert "Revenue target" in result
        assert "Discount policy" in result

    def test_irrelevant_docs_filtered_out(self):
        pairs = [
            (_make_doc("Highly relevant content"),  0.10),
            (_make_doc("Barely related content"),   0.75),  # > threshold → dropped
        ]
        result = self._search(pairs)
        assert "Highly relevant" in result
        assert "Barely related" not in result

    def test_no_relevant_docs_returns_not_found(self):
        pairs = [
            (_make_doc("Unrelated document"), 0.90),
        ]
        result = self._search(pairs)
        assert "No relevant documents" in result

    def test_empty_result_returns_not_found(self):
        result = self._search([])
        assert "No relevant documents" in result


# ═══════════════════════════════════════════════════════════════════════════
# 2. Result formatting
# ═══════════════════════════════════════════════════════════════════════════

class TestResultFormatting:
    """Output should include document index, score, source name, and content."""

    def _search(self, pairs):
        import tools.knowledge_tool as kt
        kt._VECTORSTORE = _make_vectorstore(pairs)

        with patch("tools.knowledge_tool.os.path.exists", return_value=True), \
             patch("tools.knowledge_tool.MIN_RAG_SCORE", 0.9):
            return kt.search_local_docs.invoke({"query": "test"})

    def test_includes_document_index(self):
        pairs = [(_make_doc("Content A", "kb/fileA.md"), 0.10)]
        result = self._search(pairs)
        assert "[Document 1]" in result

    def test_includes_source_filename(self):
        pairs = [(_make_doc("Some text", "kb/Q1_2025_Sales_Priorities.md"), 0.10)]
        result = self._search(pairs)
        assert "Q1_2025_Sales_Priorities.md" in result

    def test_includes_score(self):
        pairs = [(_make_doc("Content", "kb/test.md"), 0.22)]
        result = self._search(pairs)
        assert "0.22" in result

    def test_multiple_docs_separated(self):
        pairs = [
            (_make_doc("First doc",  "kb/a.md"), 0.10),
            (_make_doc("Second doc", "kb/b.md"), 0.20),
        ]
        result = self._search(pairs)
        assert "---" in result
        assert "[Document 2]" in result


# ═══════════════════════════════════════════════════════════════════════════
# 3. Error handling
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_missing_chroma_db_returns_helpful_message(self):
        import tools.knowledge_tool as kt
        kt._VECTORSTORE = None

        with patch("tools.knowledge_tool.os.path.exists", return_value=False):
            result = kt.search_local_docs.invoke({"query": "anything"})

        assert "setup_knowledge_base" in result or "not found" in result.lower()

    def test_vectorstore_exception_is_caught(self):
        import tools.knowledge_tool as kt

        broken_vs = MagicMock()
        broken_vs.similarity_search_with_score.side_effect = RuntimeError("Chroma exploded")
        kt._VECTORSTORE = broken_vs

        with patch("tools.knowledge_tool.os.path.exists", return_value=True):
            result = kt.search_local_docs.invoke({"query": "test"})

        assert "Error" in result


# ═══════════════════════════════════════════════════════════════════════════
# 4. Singleton initialisation
# ═══════════════════════════════════════════════════════════════════════════

class TestVectorstoreSingleton:
    """_init_vectorstore must not re-initialise when already set."""

    def test_vectorstore_reused_across_calls(self):
        import tools.knowledge_tool as kt

        kt._VECTORSTORE = None  # force clean state

        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = []

        with patch("tools.knowledge_tool.HuggingFaceEmbeddings"), \
             patch("tools.knowledge_tool.Chroma", return_value=mock_vs), \
             patch("tools.knowledge_tool.os.path.exists", return_value=True):

            kt._init_vectorstore()
            first = kt._VECTORSTORE
            kt._init_vectorstore()
            second = kt._VECTORSTORE

        assert first is second


# ═══════════════════════════════════════════════════════════════════════════
# 5. Semantic accuracy (requires real embedding model + tmp_chroma_db)
# ═══════════════════════════════════════════════════════════════════════════

class TestSemanticAccuracy:
    """
    These tests use the real sentence-transformer model and the seeded
    tmp_chroma_db fixture.  They validate that known queries find the
    right documents, which covers the embedding model quality.
    """

    QUERY_EXPECTED = [
        # (query, substring that must appear in results)
        ("Q1 sales target revenue goal",         "15M"),
        ("discount approval AE RSM",             "10%"),
        ("enterprise suite pro product price",   "299.99"),
        ("win rate against WorkSuite competitor","WorkSuite"),
    ]

    @pytest.mark.parametrize("query,expected_substring", QUERY_EXPECTED)
    def test_semantic_retrieval(self, query, expected_substring, tmp_chroma_db):
        import tools.knowledge_tool as kt
        kt._VECTORSTORE = None   # force re-init against test DB

        with patch("tools.knowledge_tool.CHROMA_DB_PATH", tmp_chroma_db), \
             patch("tools.knowledge_tool.MIN_RAG_SCORE", 0.8):   # generous for tiny corpus
            result = kt.search_local_docs.invoke({"query": query})

        assert expected_substring in result, (
            f"Query '{query}' did not retrieve document containing '{expected_substring}'.\n"
            f"Got: {result[:300]}"
        )
