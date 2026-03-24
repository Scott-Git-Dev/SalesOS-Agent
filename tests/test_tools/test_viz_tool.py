"""
tests/test_tools/test_viz_tool.py

Unit tests for tools/viz_tool.py.

Altair saves HTML to disk; we assert:
  - File exists and is non-empty HTML
  - Tool returns a path string
  - All chart types produce output
  - Bad JSON input is reported cleanly
  - Multi-series tool reshapes data correctly
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError


# ─── Sample data ────────────────────────────────────────────────────────────

SINGLE_SERIES = json.dumps([
    {"month": "Jan", "revenue": 10000},
    {"month": "Feb", "revenue": 15000},
    {"month": "Mar", "revenue": 12000},
])

MULTI_SERIES = json.dumps([
    {"month": "Jan", "revenue": 10000, "costs": 7000},
    {"month": "Feb", "revenue": 15000, "costs": 9000},
    {"month": "Mar", "revenue": 12000, "costs": 8000},
])

DICT_DATA = json.dumps({"Acme Corp": 50000, "TechCo": 30000, "StartX": 10000})


# ─── Helpers ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_charts_dir(tmp_path):
    """Redirect chart output to a temp directory so tests stay isolated."""
    with patch("tools.viz_tool.CHARTS_DIR", tmp_path):
        yield tmp_path


def _call_create_chart(data, chart_type="bar", title="Test Chart",
                       x_label="", y_label="", filename=""):
    from tools.viz_tool import create_chart
    return create_chart.invoke({
        "data": data,
        "chart_type": chart_type,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "filename": filename,
    })


def _call_multi_chart(data, chart_type="line", title="Multi", filename=""):
    from tools.viz_tool import create_multi_series_chart
    return create_multi_series_chart.invoke({
        "data": data,
        "chart_type": chart_type,
        "title": title,
        "filename": filename,
    })


# ═══════════════════════════════════════════════════════════════════════════
# 1. create_chart
# ═══════════════════════════════════════════════════════════════════════════

class TestCreateChart:

    def test_returns_filepath_string(self, tmp_path):
        result = _call_create_chart(SINGLE_SERIES, filename="test_bar.html")
        assert "test_bar.html" in result

    def test_html_file_is_created(self, tmp_path):
        _call_create_chart(SINGLE_SERIES, filename="out.html")
        assert (tmp_path / "out.html").exists()

    def test_html_file_is_non_empty(self, tmp_path):
        _call_create_chart(SINGLE_SERIES, filename="out.html")
        size = (tmp_path / "out.html").stat().st_size
        assert size > 1000, "HTML file seems too small to be a real Altair chart"

    @pytest.mark.parametrize("chart_type", ["bar", "line", "pie", "scatter", "area"])
    def test_all_chart_types_succeed(self, tmp_path, chart_type):
        result = _call_create_chart(SINGLE_SERIES, chart_type=chart_type)
        assert "Error" not in result

    def test_histogram_chart_type(self, tmp_path):
        # Histogram uses the single numeric column differently
        data = json.dumps([{"value": i * 100} for i in range(20)])
        result = _call_create_chart(data, chart_type="histogram")
        assert "Error" not in result

    def test_dict_data_format_accepted(self, tmp_path):
        result = _call_create_chart(DICT_DATA)
        assert "Error" not in result

    def test_custom_title_in_result(self, tmp_path):
        result = _call_create_chart(SINGLE_SERIES, title="Revenue Chart", filename="titled.html")
        # The tool's return string mentions the filename
        assert "titled.html" in result

    def test_invalid_json_returns_error(self, tmp_path):
        result = _call_create_chart("not valid json at all")
        assert "Error" in result

    def test_empty_data_returns_error(self, tmp_path):
        result = _call_create_chart(json.dumps([]))
        assert "Error" in result

    def test_unknown_chart_type_is_rejected_by_schema(self, tmp_path):
        with pytest.raises(ValidationError, match="literal_error"):
            _call_create_chart(SINGLE_SERIES, chart_type="radar")

    def test_auto_generates_filename_if_not_provided(self, tmp_path):
        result = _call_create_chart(SINGLE_SERIES)
        # Result should mention an .html file even without an explicit name
        assert ".html" in result


# ═══════════════════════════════════════════════════════════════════════════
# 2. create_multi_series_chart
# ═══════════════════════════════════════════════════════════════════════════

class TestCreateMultiSeriesChart:

    def test_line_chart_succeeds(self, tmp_path):
        result = _call_multi_chart(MULTI_SERIES, chart_type="line")
        assert "Error" not in result

    def test_bar_chart_succeeds(self, tmp_path):
        result = _call_multi_chart(MULTI_SERIES, chart_type="bar")
        assert "Error" not in result

    def test_reports_series_count(self, tmp_path):
        result = _call_multi_chart(MULTI_SERIES)
        assert "2" in result   # 2 metrics: revenue + costs

    def test_html_file_created(self, tmp_path):
        _call_multi_chart(MULTI_SERIES, filename="multi.html")
        assert (tmp_path / "multi.html").exists()

    def test_insufficient_columns_returns_error(self, tmp_path):
        one_col = json.dumps([{"month": "Jan"}, {"month": "Feb"}])
        result = _call_multi_chart(one_col)
        assert "Error" in result

    def test_invalid_chart_type_is_rejected_by_schema(self, tmp_path):
        with pytest.raises(ValidationError, match="literal_error"):
            _call_multi_chart(MULTI_SERIES, chart_type="pie")

# ════════════════════════════════════════════════════════════════════════════
# tests/test_tools/test_web_tools.py  (appended here for simplicity)
# ════════════════════════════════════════════════════════════════════════════
"""
Unit tests for tools/web_tools.py.

The Wikipedia API is always mocked — we never make real HTTP calls.
"""

class TestWikiSummary:
    """wiki_summary fetches Wikipedia and formats the response."""

    def _call(self, query: str = "Python programming language"):
        from tools.web_tools import wiki_summary
        return wiki_summary.invoke({"query": query})

    def test_returns_summary_on_success(self):
        fake_response = {
            "extract": "Python is a high-level programming language.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Python"}},
        }
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response

        with patch("tools.web_tools.requests.get", return_value=mock_resp):
            result = self._call("Python programming language")

        assert "Python" in result
        assert "https://en.wikipedia.org" in result

    def test_404_returns_helpful_message(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("tools.web_tools.requests.get", return_value=mock_resp):
            result = self._call("xyzzy_nonexistent_1234")

        assert "No Wikipedia page found" in result or "not found" in result.lower()

    def test_timeout_returns_error_message(self):
        import requests
        with patch("tools.web_tools.requests.get", side_effect=requests.Timeout):
            result = self._call("anything")

        assert "timed out" in result.lower()

    def test_generic_exception_is_caught(self):
        with patch("tools.web_tools.requests.get", side_effect=RuntimeError("network down")):
            result = self._call("anything")

        assert "Error" in result

    def test_query_spaces_converted_to_underscores(self):
        """Wikipedia API needs underscores, not spaces."""
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("tools.web_tools.requests.get", return_value=mock_resp) as mock_get:
            self._call("machine learning")

        called_url = mock_get.call_args[0][0]
        assert "machine_learning" in called_url




