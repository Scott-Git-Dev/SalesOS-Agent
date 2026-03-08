"""
Shared pytest fixtures for SalesOS test suite.

Fixtures are organized into three layers:
  1. Infrastructure  – in-memory SQLite DB, temp ChromaDB
  2. Mocks           – LLM mock that never calls the real server
  3. Integration     – real LLM + real DBs (skipped if server not available)

Run with:
  pytest tests/                        # fast, all mocked
  pytest tests/ --run-integration      # also hits local LLM server
"""

import os
import json
import sqlite3
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Generator

# ── tell the app to use our temp paths ─────────────────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "not_a_real_key")


# ═══════════════════════════════════════════════════════════════════════════
# pytest options
# ═══════════════════════════════════════════════════════════════════════════

def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests that call the real local LLM server",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require a running LLM server (deselect with -m 'not integration')",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip = pytest.mark.skip(reason="Pass --run-integration to run this test")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Infrastructure fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def tmp_sales_db(tmp_path_factory) -> str:
    """
    Minimal SQLite sales database mirroring the real schema.
    Seeded with deterministic data so assertions can use exact values.
    """
    db_dir = tmp_path_factory.mktemp("sales_db")
    db_path = db_dir / "sales_data.db"

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE regions (
            region_id INTEGER PRIMARY KEY,
            region_name TEXT NOT NULL,
            country TEXT NOT NULL
        );
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            email TEXT,
            company TEXT,
            region_id INTEGER,
            customer_since DATE,
            customer_tier TEXT
        );
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            unit_price REAL NOT NULL,
            cost REAL NOT NULL
        );
        CREATE TABLE sales (
            sale_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            sale_date DATE NOT NULL,
            total_amount REAL NOT NULL,
            status TEXT NOT NULL,
            sales_rep TEXT
        );
        CREATE TABLE sales_items (
            item_id INTEGER PRIMARY KEY,
            sale_id INTEGER,
            product_id INTEGER,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            discount REAL DEFAULT 0
        );
    """)

    # Seed: 2 regions, 3 customers, 2 products, 4 sales
    c.executemany("INSERT INTO regions VALUES (?,?,?)", [
        (1, "North America", "USA"),
        (2, "Europe",        "UK"),
    ])
    c.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?)", [
        (1, "Alice Smith",  "alice@acme.com",  "Acme Corp",    1, "2023-01-01", "Gold"),
        (2, "Bob Jones",    "bob@techco.com",  "TechCo",       2, "2023-06-01", "Silver"),
        (3, "Carol White",  "carol@startx.io", "StartX",       1, "2024-01-01", "Bronze"),
    ])
    c.executemany("INSERT INTO products VALUES (?,?,?,?,?)", [
        (1, "Enterprise Suite Pro", "Software", 299.99, 50.0),
        (2, "Analytics Dashboard",  "Software", 199.99, 30.0),
    ])

    # 2025-Q1 sales (completed)
    c.executemany("INSERT INTO sales VALUES (?,?,?,?,?,?)", [
        (1, 1, "2025-01-15", 5999.80, "Completed", "Alice Johnson"),
        (2, 2, "2025-02-20", 3999.80, "Completed", "Bob Smith"),
        (3, 3, "2025-03-10", 1999.90, "Completed", "Alice Johnson"),
        (4, 1, "2025-03-28",  999.95, "Cancelled", "Bob Smith"),    # should NOT be counted
    ])
    c.executemany("INSERT INTO sales_items VALUES (?,?,?,?,?,?)", [
        (1, 1, 1, 20, 299.99, 0.0),
        (2, 2, 2, 20, 199.99, 0.0),
        (3, 3, 1,  5, 299.99, 0.0),
        (4, 4, 2,  5, 199.99, 0.0),
    ])
    conn.commit()
    conn.close()
    return db_path.as_posix()


@pytest.fixture(scope="session")
def tmp_chroma_db(tmp_path_factory) -> Path:
    """
    Tiny ChromaDB populated with one document chunk per KB file topic.
    Only created if chromadb is importable.
    """
    try:
        import chromadb
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        pytest.skip("chromadb / langchain-chroma not installed")

    db_dir = tmp_path_factory.mktemp("chroma_db")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=str(db_dir),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="Q1 2025 Total Q1 Target: $15M. New Business Goal: $12M.",
            metadata={"source": "kb/Q1_2025_Sales_Priorities.md"},
        ),
        Document(
            page_content="Discount approval: AE can discount up to 10% with no approval. RSM up to 20%.",
            metadata={"source": "kb/Discount_Approval_Policy.txt"},
        ),
        Document(
            page_content="Enterprise Suite Pro list price $299.99/month. Unlimited users, 99.9% uptime.",
            metadata={"source": "kb/Product_Catalog_2025.md"},
        ),
        Document(
            page_content="Win rate vs WorkSuite Pro: 71%. Implementation speed is our top differentiator.",
            metadata={"source": "kb/Competitive_Battlecard_2025.txt"},
        ),
    ]
    vectorstore.add_documents(documents=docs, ids=[f"seed_{i}" for i in range(len(docs))])
    return str(db_dir)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Mock LLM fixtures
# ═══════════════════════════════════════════════════════════════════════════

def _make_llm_response(content: str):
    """Create a minimal LangChain-compatible mock response."""
    msg = MagicMock()
    msg.content = content
    return msg


@pytest.fixture
def mock_sql_llm():
    """
    Patches the LLM used by sales_tool to return a hardcoded SQL query.
    Tests can override .return_value.content on the returned mock.
    """
    with patch("tools.sales_tool._get_sql_llm") as mock_factory:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(
            "SELECT SUM(total_amount) as total_revenue FROM sales WHERE status = 'Completed'"
        )
        mock_factory.return_value = mock_llm
        yield mock_llm


@pytest.fixture
def mock_agent_llm():
    """
    Patches ChatOpenAI used by agent.create_sales_agent.
    Yields the mock so individual tests can customise .invoke / .stream.
    """
    with patch("agent.ChatOpenAI") as MockChatOpenAI:
        mock_llm = MagicMock()
        MockChatOpenAI.return_value = mock_llm
        yield mock_llm


# ═══════════════════════════════════════════════════════════════════════════
# 3. Sample data helpers
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_CHART_DATA_SINGLE = json.dumps([
    {"month": "Jan", "revenue": 10000},
    {"month": "Feb", "revenue": 15000},
    {"month": "Mar", "revenue": 12000},
])

SAMPLE_CHART_DATA_MULTI = json.dumps([
    {"month": "Jan", "revenue": 10000, "costs": 7000},
    {"month": "Feb", "revenue": 15000, "costs": 9000},
    {"month": "Mar", "revenue": 12000, "costs": 8000},
])
