"""
tests/eval/eval_cases.py

Golden test cases for the SalesOS eval framework.

Each EvalCase defines:
  - question      : the user's input
  - expected_tools: which tool(s) the agent MUST call (in any order)
  - must_contain  : substrings the final answer MUST contain
  - must_not_contain: substrings that indicate a wrong answer
  - judge_criteria: what the LLM judge should evaluate
  - category      : for grouped reporting

Adding a new test case is the only change needed to extend coverage.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalCase:
    id: str
    question: str
    category: str
    expected_tools: List[str]
    must_contain: List[str] = field(default_factory=list)
    must_not_contain: List[str] = field(default_factory=list)
    judge_criteria: str = ""
    notes: str = ""


# ════════════════════════════════════════════════════════════════════════════
# SQL-only questions
# ════════════════════════════════════════════════════════════════════════════

SQL_CASES = [
    EvalCase(
        id="sql_01",
        question="What were total sales last quarter?",
        category="SQL",
        expected_tools=["query_sales_database"],
        must_contain=["$", "revenue", "sales"],
        must_not_contain=["I don't know", "cannot access", "no data"],
        judge_criteria=(
            "The answer must include a dollar amount for total revenue. "
            "It should clearly state the revenue figure for the requested time period. "
            "It should NOT confuse goals/targets with actual revenue."
        ),
    ),
    EvalCase(
        id="sql_02",
        question="Who are our top 5 customers by revenue?",
        category="SQL",
        expected_tools=["query_sales_database"],
        must_contain=["1", "2", "3"],
        must_not_contain=["I cannot", "no customers"],
        judge_criteria=(
            "The answer must list at least 3 customer names with associated revenue figures. "
            "Customers should be ranked in descending order by revenue."
        ),
    ),
    EvalCase(
        id="sql_03",
        question="Show me monthly revenue trends for 2024",
        category="SQL",
        expected_tools=["query_sales_database"],
        must_contain=["2024"],
        judge_criteria=(
            "The answer must show revenue broken down by month for 2024. "
            "Months should be in chronological order."
        ),
    ),
    EvalCase(
        id="sql_04",
        question="How many completed sales did we have in Q1 2025?",
        category="SQL",
        expected_tools=["query_sales_database"],
        must_contain=["Q1", "2025"],
        must_not_contain=["I don't know"],
        judge_criteria=(
            "The answer must include a count of completed sales transactions in Q1 2025."
        ),
    ),
    EvalCase(
        id="sql_05",
        question="Which product category generates the most revenue?",
        category="SQL",
        expected_tools=["query_sales_database"],
        must_contain=["Software", "Hardware", "Services", "Licenses"],
        judge_criteria=(
            "The answer must identify one product category as the top revenue generator "
            "and ideally provide the revenue figures for at least two categories."
        ),
    ),
]


# ════════════════════════════════════════════════════════════════════════════
# RAG-only questions
# ════════════════════════════════════════════════════════════════════════════

RAG_CASES = [
    EvalCase(
        id="rag_01",
        question="What is the Q1 2025 total sales target?",
        category="RAG",
        expected_tools=["search_local_docs"],
        must_contain=["15"],
        must_not_contain=["I don't know", "cannot find"],
        judge_criteria=(
            "The answer must state the $15M Q1 2025 total sales target. "
            "It may also break down the target into new business ($12M) and expansion ($3M)."
        ),
    ),
    EvalCase(
        id="rag_02",
        question="What discount can an Account Executive approve without manager sign-off?",
        category="RAG",
        expected_tools=["search_local_docs"],
        must_contain=["10"],
        must_not_contain=["I don't know"],
        judge_criteria=(
            "The answer must state that AEs can approve up to 10% discount without approval. "
            "Conditions (single-year commitment, <$50K deal) are a bonus."
        ),
    ),
    EvalCase(
        id="rag_03",
        question="What is the price of Enterprise Suite Pro?",
        category="RAG",
        expected_tools=["search_local_docs"],
        must_contain=["299"],
        judge_criteria=(
            "The answer must include the price of $299.99/month for Enterprise Suite Pro."
        ),
    ),
    EvalCase(
        id="rag_04",
        question="How does our win rate compare against WorkSuite Pro?",
        category="RAG",
        expected_tools=["search_local_docs"],
        must_contain=["WorkSuite"],
        judge_criteria=(
            "The answer should mention the win rate against WorkSuite Pro (71%) "
            "and at least one competitive differentiator like implementation speed."
        ),
    ),
    EvalCase(
        id="rag_05",
        question="What are the top reasons for losing deals in the last 90 days?",
        category="RAG",
        expected_tools=["search_local_docs"],
        must_contain=["price", "Price", "cost", "cheap"],
        judge_criteria=(
            "The answer must identify at least 2 of the top loss reasons. "
            "Price should be mentioned as the #1 loss reason."
        ),
    ),
]


# ════════════════════════════════════════════════════════════════════════════
# Multi-tool questions (requires BOTH SQL and RAG)
# ════════════════════════════════════════════════════════════════════════════

MULTI_TOOL_CASES = [
    EvalCase(
        id="multi_01",
        question="Did we hit our Q1 2025 sales targets?",
        category="MULTI_TOOL",
        expected_tools=["query_sales_database", "search_local_docs"],
        must_contain=["$15M", "15"],
        must_not_contain=["I cannot", "insufficient data"],
        judge_criteria=(
            "The answer MUST compare actual Q1 sales (from database) against the $15M target (from docs). "
            "It should give a gap analysis: how much short or over target. "
            "Using only one source is a failure."
        ),
    ),
    EvalCase(
        id="multi_02",
        question="Should we offer a discount to our top customer?",
        category="MULTI_TOOL",
        expected_tools=["query_sales_database", "search_local_docs"],
        must_contain=[],
        judge_criteria=(
            "The answer should identify who the top customer is (from database) AND "
            "reference the discount policy (from docs) to give a recommendation. "
            "An answer with no customer data or no policy reference is incomplete."
        ),
    ),
]


# ════════════════════════════════════════════════════════════════════════════
# Scope boundary tests (questions the SQL tool must decline)
# ════════════════════════════════════════════════════════════════════════════

SCOPE_CASES = [
    EvalCase(
        id="scope_01",
        question="What is our Q1 revenue target?",
        category="SCOPE",
        expected_tools=["search_local_docs"],   # Must NOT use SQL tool
        must_not_contain=["database error", "no such column"],
        judge_criteria=(
            "Goals and targets are NOT in the sales database. "
            "The agent must use search_local_docs for this, not query_sales_database. "
            "If the agent erroneously tries SQL and gets a scope error it must recover gracefully."
        ),
    ),
    EvalCase(
        id="scope_02",
        question="What is the customer satisfaction score for Acme Corp?",
        category="SCOPE",
        expected_tools=[],   # no tool has this data
        must_contain=["Acme"],
        must_not_contain=["9", "8", "7", "satisfaction score is"],
        judge_criteria=(
            "Customer satisfaction scores don't exist in the database or knowledge base. "
            "The agent should admit this gracefully rather than hallucinating a number."
        ),
    ),
]


# ════════════════════════════════════════════════════════════════════════════
# System prompt adherence
# ════════════════════════════════════════════════════════════════════════════

PROMPT_ADHERENCE_CASES = [
    EvalCase(
        id="prompt_01",
        question="show me a bar chart of top 5 customers by revenue",
        category="PROMPT_ADHERENCE",
        expected_tools=["query_sales_database", "create_chart"],
        judge_criteria=(
            "The system prompt specifies a visualization workflow: get data first, THEN call create_chart. "
            "The answer must follow this two-step sequence. "
            "The answer should mention that a chart was created and where to find it."
        ),
    ),
    EvalCase(
        id="prompt_02",
        question="Who is Elon Musk?",
        category="PROMPT_ADHERENCE",
        expected_tools=["wiki_summary"],
        must_contain=["Musk", "Tesla", "SpaceX"],
        judge_criteria=(
            "General knowledge questions should use wiki_summary per the system prompt. "
            "The answer should contain factual biographical information."
        ),
    ),
]


# ════════════════════════════════════════════════════════════════════════════
# All cases bundled
# ════════════════════════════════════════════════════════════════════════════

ALL_CASES: List[EvalCase] = (
    SQL_CASES
    + RAG_CASES
    + MULTI_TOOL_CASES
    + SCOPE_CASES
    + PROMPT_ADHERENCE_CASES
)

CASES_BY_CATEGORY = {
    "SQL":             SQL_CASES,
    "RAG":             RAG_CASES,
    "MULTI_TOOL":      MULTI_TOOL_CASES,
    "SCOPE":           SCOPE_CASES,
    "PROMPT_ADHERENCE": PROMPT_ADHERENCE_CASES,
}
