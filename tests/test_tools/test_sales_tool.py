"""
tests/test_tools/test_sales_tool.py

Unit tests for tools/sales_tool.py.

What we test:
  - SQL safety validation (blocks DROP/DELETE etc.)
  - Schema caching (only hits DB once per session)
  - SQL generation via mocked LLM
  - Scope error passthrough
  - Query-refinement loop on SQL execution failure
  - Result formatting for single-row and multi-row results
  - Full tool call against the in-memory test database

We NEVER call the real LLM server here; the LLM is always mocked.
"""

import sqlite3
import json
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# Helpers / shared mocks
# ═══════════════════════════════════════════════════════════════════════════

def _make_response(text: str):
    r = MagicMock()
    r.content = text
    return r


# ═══════════════════════════════════════════════════════════════════════════
# 1. SQL Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateSQL:
    """_validate_sql must raise on any mutation keyword."""

    def setup_method(self):
        from tools.sales_tool import _validate_sql
        self.validate = _validate_sql

    @pytest.mark.parametrize("bad_sql", [
        "DROP TABLE sales",
        "DELETE FROM customers",
        "UPDATE products SET price=0",
        "INSERT INTO sales VALUES (1,2,3)",
        "ALTER TABLE sales ADD COLUMN foo TEXT",
        "TRUNCATE TABLE sales",
        "CREATE TABLE hacked (x TEXT)",
        "PRAGMA journal_mode=WAL",
    ])
    def test_rejects_mutation_keywords(self, bad_sql):
        with pytest.raises(ValueError, match="forbidden keyword"):
            self.validate(bad_sql)

    @pytest.mark.parametrize("good_sql", [
        "SELECT * FROM sales",
        "SELECT SUM(total_amount) FROM sales WHERE status = 'Completed'",
        "SELECT c.company, SUM(s.total_amount) FROM sales s JOIN customers c ON s.customer_id = c.customer_id GROUP BY c.company",
    ])
    def test_accepts_select_statements(self, good_sql):
        self.validate(good_sql)  # must not raise


# ═══════════════════════════════════════════════════════════════════════════
# 2. SQL Generation (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateSQLWithLLM:
    """_generate_sql_with_llm should call LLM and clean up its response."""

    def _call(self, llm_output: str, question: str = "test question"):
        from tools.sales_tool import _generate_sql_with_llm
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_response(llm_output)

        with patch("tools.sales_tool._get_sql_llm", return_value=mock_llm):
            return _generate_sql_with_llm(question, "fake schema")

    def test_strips_markdown_fences(self):
        result = self._call("```sql\nSELECT 1\n```")
        assert result == "SELECT 1"

    def test_strips_trailing_semicolon(self):
        result = self._call("SELECT 1;")
        assert result == "SELECT 1"

    def test_passes_through_scope_error(self):
        result = self._call("SCOPE_ERROR: Sales goals are not in the database.")
        assert result.startswith("SCOPE_ERROR:")

    def test_rejects_non_select(self):
        result = self._call("DROP TABLE sales")
        assert result.startswith("ERROR:")

    def test_system_and_user_messages_sent_to_llm(self):
        """Ensures we send a proper SystemMessage + HumanMessage pair."""
        from langchain_core.messages import SystemMessage, HumanMessage
        from tools.sales_tool import _generate_sql_with_llm

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_response("SELECT 1")

        with patch("tools.sales_tool._get_sql_llm", return_value=mock_llm):
            _generate_sql_with_llm("my question", "my schema")

        call_args = mock_llm.invoke.call_args[0][0]
        assert isinstance(call_args[0], SystemMessage)
        assert isinstance(call_args[1], HumanMessage)
        assert "my question" in call_args[1].content


# ═══════════════════════════════════════════════════════════════════════════
# 3. Schema caching
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemaCaching:
    """Schema must only be read from the DB once per process."""

    def test_schema_fetched_once(self, tmp_sales_db):
        import tools.sales_tool as st
        # Reset cache
        st._SCHEMA_CACHE = None

        with patch("tools.sales_tool.SALES_DB_PATH", tmp_sales_db):
            schema1 = st._get_schema_cached()
            schema2 = st._get_schema_cached()

        assert schema1 is schema2          # same object → cached
        assert "sales" in schema1.lower()  # sanity check content


# ═══════════════════════════════════════════════════════════════════════════
# 4. Query refinement loop
# ═══════════════════════════════════════════════════════════════════════════

class TestRefinement:
    """_refine_failed_query should call the LLM once and return cleaned SQL."""

    def test_refine_returns_corrected_sql(self):
        from tools.sales_tool import _refine_failed_query

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_response("SELECT 1 -- corrected")

        with patch("tools.sales_tool._get_sql_llm", return_value=mock_llm):
            result = _refine_failed_query(
                question="What is revenue?",
                failed_sql="SELECT bad_col FROM sales",
                error="no such column: bad_col",
                schema="fake schema",
            )

        assert result == "SELECT 1 -- corrected"

    def test_refine_returns_none_if_same_query(self):
        """If LLM returns the same broken query, return None to avoid infinite loop."""
        from tools.sales_tool import _refine_failed_query

        broken = "SELECT bad_col FROM sales"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_response(broken)

        with patch("tools.sales_tool._get_sql_llm", return_value=mock_llm):
            result = _refine_failed_query(
                question="revenue?",
                failed_sql=broken,
                error="no such column: bad_col",
                schema="fake schema",
            )

        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 5. Result formatting
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatResults:
    """_format_results_structured produces human-readable, agent-parseable output."""

    def setup_method(self):
        from tools.sales_tool import _format_results_structured
        self.fmt = _format_results_structured

    def test_single_aggregate_row(self):
        rows = [{"total_revenue": 271680.50}]
        out = self.fmt(rows, "What is total revenue?")
        assert "$271,680.50" in out
        assert "total_revenue" in out

    def test_multi_row_list(self):
        rows = [
            {"company": "Acme Corp", "total_revenue": 50000.0},
            {"company": "TechCo",    "total_revenue": 30000.0},
        ]
        out = self.fmt(rows, "top customers?")
        assert "Acme Corp" in out
        assert "TechCo" in out
        assert "2 row" in out

    def test_truncates_at_10_rows(self):
        rows = [{"company": f"Co{i}", "revenue": i * 1000.0} for i in range(20)]
        out = self.fmt(rows, "all customers")
        assert "10 more rows" in out

    def test_chart_suggestion_for_two_column_data(self):
        rows = [{"month": f"2025-0{i}", "revenue": i * 1000.0} for i in range(1, 5)]
        out = self.fmt(rows, "monthly trends")
        assert "create_chart" in out


# ═══════════════════════════════════════════════════════════════════════════
# 6. End-to-end tool call (mocked LLM, real in-memory DB)
# ═══════════════════════════════════════════════════════════════════════════

class TestQuerySalesDatabaseTool:
    """
    Full tool execution against the seeded in-memory database.
    The LLM is mocked to return a known correct SQL query.
    """

    def _run(self, sql: str, tmp_sales_db, question: str = "test"):
        """Patch DB path and LLM, then invoke the tool."""
        import tools.sales_tool as st
        st._SCHEMA_CACHE = None          # reset between tests
        st._SQL_LLM = None

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_response(sql)

        with patch("tools.sales_tool.SALES_DB_PATH", tmp_sales_db), \
             patch("tools.sales_tool._get_sql_llm", return_value=mock_llm):
            return st.query_sales_database.invoke({"question": question})

    def test_total_completed_revenue(self, tmp_sales_db):
        """Seeded DB has $11,999.50 in completed sales (3 × rows, Cancelled excluded)."""
        result = self._run(
            "SELECT SUM(total_amount) as total_revenue FROM sales WHERE status = 'Completed'",
            tmp_sales_db,
        )
        # $5999.80 + $3999.80 + $1999.90 = $11,999.50
        assert "11,999.50" in result

    def test_top_customers_returns_companies(self, tmp_sales_db):
        result = self._run(
            """SELECT c.company, SUM(s.total_amount) as revenue
               FROM sales s
               JOIN customers c ON s.customer_id = c.customer_id
               WHERE s.status = 'Completed'
               GROUP BY c.company
               ORDER BY revenue DESC""",
            tmp_sales_db,
            question="top customers",
        )
        assert "Acme Corp" in result
        assert "TechCo" in result

    def test_scope_error_is_surfaced(self, tmp_sales_db):
        """SCOPE_ERROR from LLM should produce a helpful redirect message."""
        import tools.sales_tool as st
        st._SCHEMA_CACHE = None
        st._SQL_LLM = None

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_response(
            "SCOPE_ERROR: Sales goals are not in the database. Try search_local_docs."
        )

        with patch("tools.sales_tool.SALES_DB_PATH", tmp_sales_db), \
             patch("tools.sales_tool._get_sql_llm", return_value=mock_llm):
            result = st.query_sales_database.invoke({"question": "What is our Q1 target?"})

        assert "search_local_docs" in result
        assert "not in the sales database" in result.lower() or "scope" in result.lower()

    def test_empty_result_handled_gracefully(self, tmp_sales_db):
        result = self._run(
            "SELECT * FROM sales WHERE sale_date = '1900-01-01'",
            tmp_sales_db,
        )
        assert "no results" in result.lower() or "0 row" in result.lower() or "returned no results" in result.lower()
