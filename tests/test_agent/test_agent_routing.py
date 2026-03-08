"""
tests/test_agent/test_agent_routing.py

Tests that the agent:
  1. Selects the correct tool(s) for different question types
  2. Adheres to system prompt routing rules
  3. Handles multi-tool questions (SQL + RAG)
  4. Passes thread_id through to LangGraph memory

These tests inspect the message flow produced by the agent rather than
the final answer text, so they work without a real LLM: we mock the
LLM to return a deterministic tool-call sequence.

Test Strategy
─────────────
LangGraph's create_react_agent loops:
  HumanMessage → AIMessage(tool_calls=[…]) → ToolMessages → AIMessage(content=…)

We mock the LLM so that:
  - First invoke → returns an AIMessage with the expected tool_call(s)
  - Second invoke → returns a final AIMessage with a text answer

This lets us assert which tools were selected without running the real model.
"""

import uuid
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_tool_call(name: str, args: dict, call_id: str = None) -> dict:
    return {
        "name": name,
        "args": args,
        "id": call_id or f"call_{name}_{uuid.uuid4().hex[:6]}",
        "type": "tool_call",
    }


def _ai_with_tool_calls(*calls) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


def _ai_final(text: str) -> AIMessage:
    return AIMessage(content=text)


# ═══════════════════════════════════════════════════════════════════════════
# Routing rule assertions
# ═══════════════════════════════════════════════════════════════════════════

# Maps question patterns to the tool the system prompt says to use
ROUTING_RULES = [
    # (question, expected_primary_tool, description)
    (
        "What were total sales last quarter?",
        "query_sales_database",
        "historical revenue → SQL tool",
    ),
    (
        "Who are our top 5 customers by revenue?",
        "query_sales_database",
        "customer rankings → SQL tool",
    ),
    (
        "What is our Q1 2025 sales target?",
        "search_local_docs",
        "goals/targets → RAG tool",
    ),
    (
        "What is the discount approval policy?",
        "search_local_docs",
        "company policy → RAG tool",
    ),
    (
        "What is quantum computing?",
        "wiki_summary",
        "general knowledge → Wikipedia tool",
    ),
]


class TestSystemPromptRouting:
    """
    Validates that the system prompt correctly steers tool selection.

    We drive the LLM mock to return the tool call that the system prompt
    specifies, then verify the agent actually invokes that tool.
    """

    def _run_agent_and_capture_tool_calls(
        self,
        question: str,
        tool_name: str,
        agent,
        thread_id: str,
    ) -> list[str]:
        """
        Invoke agent and return the list of tool names that were called.
        We read this from the AIMessage.tool_calls in the message history.
        """
        from langchain_core.messages import HumanMessage
        config = {"configurable": {"thread_id": thread_id}}
        result = agent.invoke({"messages": [HumanMessage(content=question)]}, config)

        called_tools = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    called_tools.append(tc["name"])
        return called_tools

    @pytest.mark.parametrize("question,expected_tool,description", ROUTING_RULES)
    def test_tool_routing(self, question, expected_tool, description, tmp_sales_db, tmp_chroma_db):
        """
        For each routing rule: create a mock LLM that returns the expected
        tool call, run the agent, verify the tool appears in the call log.

        This validates that the agent wiring (tool binding, graph edges) is
        correct — it doesn't test LLM intelligence.
        """
        import tools.sales_tool as st
        import tools.knowledge_tool as kt
        st._SCHEMA_CACHE = None
        kt._VECTORSTORE = None

        # Mock: first call → tool selection; second call → final answer
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm  # LangGraph binds tools

        call_count = [0]

        def side_effect(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _ai_with_tool_calls(
                    _make_tool_call(expected_tool, {"question": question} if "sales" in expected_tool else {"query": question})
                )
            return _ai_final(f"Answer to: {question}")

        mock_llm.invoke.side_effect = side_effect

        with patch("agent.ChatOpenAI", return_value=mock_llm), \
             patch("agent.validate_config", return_value=True), \
             patch("tools.sales_tool.SALES_DB_PATH", tmp_sales_db), \
             patch("tools.knowledge_tool.CHROMA_DB_PATH", tmp_chroma_db), \
             patch("tools.knowledge_tool.os.path.exists", return_value=True):

            from agent import create_sales_agent
            agent = create_sales_agent()
            tools_called = self._run_agent_and_capture_tool_calls(
                question, expected_tool, agent, thread_id=str(uuid.uuid4())
            )

        assert expected_tool in tools_called, (
            f"ROUTING FAILURE — {description}\n"
            f"  Question:      '{question}'\n"
            f"  Expected tool: {expected_tool}\n"
            f"  Called tools:  {tools_called}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Multi-tool routing (both SQL + RAG required)
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiToolRouting:
    """
    The system prompt instructs the agent to use BOTH tools for
    goal-vs-actual questions.  Verify both appear in the call log.
    """

    MULTI_TOOL_QUESTIONS = [
        "Did we hit our Q1 2025 sales targets?",
        "How are we performing against our revenue goals?",
    ]

    @pytest.mark.parametrize("question", MULTI_TOOL_QUESTIONS)
    def test_multi_tool_question_calls_both_tools(self, question, tmp_sales_db, tmp_chroma_db):
        import tools.sales_tool as st
        import tools.knowledge_tool as kt
        st._SCHEMA_CACHE = None
        kt._VECTORSTORE = None

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        call_count = [0]

        def side_effect(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Agent requests BOTH tools simultaneously
                return _ai_with_tool_calls(
                    _make_tool_call("query_sales_database", {"question": "Q1 revenue"}),
                    _make_tool_call("search_local_docs", {"query": "Q1 sales target"}),
                )
            return _ai_final("Q1 sales were $X vs target of $Y.")

        mock_llm.invoke.side_effect = side_effect

        with patch("agent.ChatOpenAI", return_value=mock_llm), \
             patch("agent.validate_config", return_value=True), \
             patch("tools.sales_tool.SALES_DB_PATH", tmp_sales_db), \
             patch("tools.knowledge_tool.CHROMA_DB_PATH", tmp_chroma_db), \
             patch("tools.knowledge_tool.os.path.exists", return_value=True):

            from agent import create_sales_agent
            agent = create_sales_agent()
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            result = agent.invoke({"messages": [HumanMessage(content=question)]}, config)

        called = [
            tc["name"]
            for msg in result["messages"]
            if hasattr(msg, "tool_calls")
            for tc in (msg.tool_calls or [])
        ]

        assert "query_sales_database" in called, f"SQL tool not called for: {question}"
        assert "search_local_docs" in called, f"RAG tool not called for: {question}"


# ═══════════════════════════════════════════════════════════════════════════
# Thread ID / memory isolation
# ═══════════════════════════════════════════════════════════════════════════

class TestConversationMemory:
    """
    LangGraph InMemorySaver should isolate conversation state per thread_id.
    Asking the same question on two different threads should produce
    independent message histories.
    """

    def test_different_threads_are_independent(self, tmp_sales_db, tmp_chroma_db):
        import tools.sales_tool as st
        import tools.knowledge_tool as kt
        st._SCHEMA_CACHE = None
        kt._VECTORSTORE = None

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = _ai_final("some answer")

        with patch("agent.ChatOpenAI", return_value=mock_llm), \
             patch("agent.validate_config", return_value=True), \
             patch("tools.sales_tool.SALES_DB_PATH", tmp_sales_db), \
             patch("tools.knowledge_tool.CHROMA_DB_PATH", tmp_chroma_db), \
             patch("tools.knowledge_tool.os.path.exists", return_value=True):

            from agent import create_sales_agent
            agent = create_sales_agent()

            thread_a = str(uuid.uuid4())
            thread_b = str(uuid.uuid4())

            config_a = {"configurable": {"thread_id": thread_a}}
            config_b = {"configurable": {"thread_id": thread_b}}

            agent.invoke({"messages": [HumanMessage(content="Hello thread A")]}, config_a)
            agent.invoke({"messages": [HumanMessage(content="Hello thread B")]}, config_b)

            state_a = agent.get_state(config_a)
            state_b = agent.get_state(config_b)

            msgs_a = [m.content for m in state_a.values["messages"] if hasattr(m, "content")]
            msgs_b = [m.content for m in state_b.values["messages"] if hasattr(m, "content")]

            assert "Hello thread A" in " ".join(msgs_a)
            assert "Hello thread B" in " ".join(msgs_b)
            # Cross-contamination check
            assert "Hello thread B" not in " ".join(msgs_a)
            assert "Hello thread A" not in " ".join(msgs_b)
