"""
tests/eval/eval_judge.py

LLM-as-Judge evaluator for SalesOS.

Uses the same local LLM configured in config.py to score agent responses.
This is a standard technique in LLM evaluation (LMSYS, MT-Bench, etc.)
adapted for a RAG + agent setting.

Score dimensions
────────────────
  correctness  (1-5)   factually right based on judge_criteria
  completeness (1-5)   answers all parts of the question
  tool_use     (1-5)   used the right tools; didn't hallucinate tool results
  grounded     (1-5)   claims are supported by data, not invented
  total        (0-20)  sum of above

Design notes
────────────
- We ask the judge to return structured JSON so scores are machine-readable.
- We include the eval case's judge_criteria in the prompt so the judge
  knows what a good answer looks like.
- We strip the judge's thinking (if any) and parse only the JSON block.
- If the judge call fails, we return a GradeResult with is_valid=False
  so the runner can still report a partial result.
"""

import json
import re
import time
from dataclasses import dataclass, field


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# ════════════════════════════════════════════════════════════════════════════
# Data types
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class GradeResult:
    correctness: int   = 0   # 1-5
    completeness: int  = 0   # 1-5
    tool_use: int      = 0   # 1-5
    grounded: int      = 0   # 1-5
    reasoning: str     = ""
    is_valid: bool     = True
    error: str         = ""

    @property
    def total(self) -> int:
        return self.correctness + self.completeness + self.tool_use + self.grounded

    @property
    def percentage(self) -> float:
        return round(self.total / 20 * 100, 1)

    def to_dict(self) -> dict:
        return {
            "correctness":  self.correctness,
            "completeness": self.completeness,
            "tool_use":     self.tool_use,
            "grounded":     self.grounded,
            "total":        self.total,
            "percentage":   self.percentage,
            "reasoning":    self.reasoning,
            "is_valid":     self.is_valid,
            "error":        self.error,
        }


# ════════════════════════════════════════════════════════════════════════════
# Judge prompt
# ════════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for an AI sales assistant called SalesOS.
Your job is to grade how well the assistant answered a user question.

The assistant has access to three types of information:
1. A SQL database with historical sales data (actual revenue, customers, products)
2. A knowledge base with company documents (goals, targets, policies, playbooks)
3. Wikipedia for general knowledge

You will be given:
- The user's question
- The assistant's answer
- Which tools were called (tool call log)
- Evaluation criteria specific to this question

Grade on these four dimensions (1=poor, 5=excellent):

CORRECTNESS (1-5)
  5: Factually accurate, numbers are correct
  3: Mostly correct but minor errors or imprecision
  1: Wrong facts, wrong numbers, or confuses targets with actuals

COMPLETENESS (1-5)
  5: Fully answers the question, addresses all parts
  3: Answers the core question but misses some aspects
  1: Partial answer, misses the main point

TOOL_USE (1-5)
  5: Used exactly the right tools; didn't hallucinate results
  3: Used mostly right tools but minor issues (e.g., unnecessary call)
  1: Wrong tool, skipped a needed tool, or hallucinated data not from a tool

GROUNDED (1-5)
  5: All claims trace back to tool outputs or stated assumptions
  3: Most claims grounded, minor unsupported statements
  1: Answers include invented data not from any tool

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{
  "correctness": <1-5>,
  "completeness": <1-5>,
  "tool_use": <1-5>,
  "grounded": <1-5>,
  "reasoning": "<one paragraph explaining your scores>"
}"""


JUDGE_USER_TEMPLATE = """USER QUESTION:
{question}

TOOLS CALLED (in order):
{tools_called}

ASSISTANT'S ANSWER:
{answer}

EVALUATION CRITERIA FOR THIS QUESTION:
{judge_criteria}

Grade the assistant's answer now:"""


# ════════════════════════════════════════════════════════════════════════════
# Judge class
# ════════════════════════════════════════════════════════════════════════════

class LLMJudge:
    """
    Calls the local LLM to score an agent response.

    Parameters
    ──────────
    model_name   : LLM model (from config)
    base_url     : LLM server URL (from config)
    temperature  : Low temperature for deterministic scoring
    max_retries  : Retry if JSON parse fails
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        temperature: float = 0.1,
        max_retries: int = 2,
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            max_tokens=8000,
        )
        self.max_retries = max_retries

    def grade(
        self,
        question: str,
        answer: str,
        tools_called: list[str],
        judge_criteria: str,
    ) -> GradeResult:
        """
        Grade a single answer.  Returns GradeResult.
        On failure, returns an invalid GradeResult with error message.
        """
        tools_str = (
            "\n".join(f"  {i+1}. {t}" for i, t in enumerate(tools_called))
            if tools_called
            else "  (no tools called)"
        )

        user_message = JUDGE_USER_TEMPLATE.format(
            question=question,
            tools_called=tools_str,
            answer=answer[:3000],  # truncate very long answers
            judge_criteria=judge_criteria,
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm.invoke([
                    SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                    HumanMessage(content=user_message),
                ])

                raw = response.content.strip()

                # Extract JSON block if wrapped in markdown
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    raw = json_match.group(0)

                data = json.loads(raw)

                return GradeResult(
                    correctness=int(data.get("correctness", 0)),
                    completeness=int(data.get("completeness", 0)),
                    tool_use=int(data.get("tool_use", 0)),
                    grounded=int(data.get("grounded", 0)),
                    reasoning=data.get("reasoning", ""),
                    is_valid=True,
                )

            except json.JSONDecodeError as e:
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue
                return GradeResult(
                    is_valid=False,
                    error=f"JSON parse error after {self.max_retries+1} attempts: {e}\nRaw: {raw[:200]}",
                )
            except Exception as e:
                return GradeResult(
                    is_valid=False,
                    error=f"Judge call failed: {str(e)}",
                )

        return GradeResult(is_valid=False, error="Max retries exceeded")


# ════════════════════════════════════════════════════════════════════════════
# Heuristic checks (no LLM needed)
# ════════════════════════════════════════════════════════════════════════════

def check_must_contain(answer: str, must_contain: list[str]) -> tuple[bool, list[str]]:
    """Returns (passed, list of missing substrings)."""
    missing = [s for s in must_contain if s not in answer]
    return len(missing) == 0, missing


def check_must_not_contain(answer: str, must_not_contain: list[str]) -> tuple[bool, list[str]]:
    """Returns (passed, list of found forbidden substrings)."""
    found = [s for s in must_not_contain if s.lower() in answer.lower()]
    return len(found) == 0, found


def check_tool_calls(
    actual_tools: list[str],
    expected_tools: list[str],
) -> tuple[bool, list[str]]:
    """Returns (all expected tools were called, list of missing tools)."""
    missing = [t for t in expected_tools if t not in actual_tools]
    return len(missing) == 0, missing
