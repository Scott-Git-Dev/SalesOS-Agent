"""
tests/eval/eval_runner.py

Main evaluation runner for SalesOS.

Runs the full agent against every EvalCase in eval_cases.py,
scores each response with heuristics + LLM judge, and writes
results to a JSON file for the report generator.

Usage:
    # From project root:
    python tests/eval/eval_runner.py

    # Run only one category:
    python tests/eval/eval_runner.py --category SQL

    # Skip LLM judge (faster, heuristics only):
    python tests/eval/eval_runner.py --no-judge

    # Run a single case by ID:
    python tests/eval/eval_runner.py --case rag_01
"""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import MODEL_NAME, LLAMA_SERVER_URL
from agent import create_sales_agent, ask_agent
from langchain_core.messages import AIMessage

from tests.eval.eval_cases import ALL_CASES, CASES_BY_CATEGORY, EvalCase
from tests.eval.eval_judge import (
    LLMJudge,
    GradeResult,
    check_must_contain,
    check_must_not_contain,
    check_tool_calls,
)


# ════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ════════════════════════════════════════════════════════════════════════════

class EvalResult:
    def __init__(self, case: EvalCase):
        self.case_id       = case.id
        self.question      = case.question
        self.category      = case.category
        self.answer        = ""
        self.tools_called  : list[str] = []
        self.latency_s     : float = 0.0

        # Heuristic checks
        self.must_contain_pass   : bool = True
        self.must_contain_missing: list[str] = []
        self.must_not_contain_pass: bool = True
        self.must_not_contain_found: list[str] = []
        self.tool_call_pass      : bool = True
        self.tool_call_missing   : list[str] = []

        # LLM judge scores
        self.grade: Optional[GradeResult] = None

        # Overall
        self.error: str = ""

    @property
    def heuristic_pass(self) -> bool:
        return (
            self.must_contain_pass
            and self.must_not_contain_pass
            and self.tool_call_pass
        )

    def to_dict(self) -> dict:
        return {
            "case_id":              self.case_id,
            "question":             self.question,
            "category":             self.category,
            "answer":               self.answer[:1000],
            "tools_called":         self.tools_called,
            "latency_s":            round(self.latency_s, 2),
            "heuristic_pass":       self.heuristic_pass,
            "must_contain_pass":    self.must_contain_pass,
            "must_contain_missing": self.must_contain_missing,
            "must_not_contain_pass":      self.must_not_contain_pass,
            "must_not_contain_found":     self.must_not_contain_found,
            "tool_call_pass":       self.tool_call_pass,
            "tool_call_missing":    self.tool_call_missing,
            "grade":                self.grade.to_dict() if self.grade else None,
            "error":                self.error,
        }


# ════════════════════════════════════════════════════════════════════════════
# Tool-call extraction
# ════════════════════════════════════════════════════════════════════════════

def _extract_tools_called(agent, thread_id: str) -> list[str]:
    """Read which tools were called from LangGraph's in-memory state."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = agent.get_state(config)
        messages = state.values.get("messages", [])
        tools = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools.append(tc["name"])
        return tools
    except Exception:
        return []


# ════════════════════════════════════════════════════════════════════════════
# Single case runner
# ════════════════════════════════════════════════════════════════════════════

def run_case(
    case: EvalCase,
    agent,
    judge: Optional[LLMJudge],
    verbose: bool = True,
) -> EvalResult:

    result = EvalResult(case)
    thread_id = str(uuid.uuid4())

    if verbose:
        print(f"\n{'─'*60}")
        print(f"[{case.id}] {case.category} | {case.question[:80]}")

    # ── 1. Run the agent ───────────────────────────────────────────────────
    try:
        t0 = time.time()
        answer = ask_agent(agent, case.question, thread_id=thread_id, verbose=False)
        result.latency_s = time.time() - t0
        result.answer = answer or ""
        result.tools_called = _extract_tools_called(agent, thread_id)

        if verbose:
            print(f"  Tools called : {result.tools_called}")
            print(f"  Latency      : {result.latency_s:.1f}s")
            print(f"  Answer (200c): {result.answer[:200]}")

    except Exception as e:
        result.error = str(e)
        if verbose:
            print(f"  ❌ Agent error: {e}")
        return result

    # ── 2. Heuristic checks ────────────────────────────────────────────────
    result.must_contain_pass, result.must_contain_missing = check_must_contain(
        result.answer, case.must_contain
    )
    result.must_not_contain_pass, result.must_not_contain_found = check_must_not_contain(
        result.answer, case.must_not_contain
    )
    result.tool_call_pass, result.tool_call_missing = check_tool_calls(
        result.tools_called, case.expected_tools
    )

    heuristic_icon = "✅" if result.heuristic_pass else "⚠️ "
    if verbose:
        print(f"  Heuristics   : {heuristic_icon} (contain={result.must_contain_pass}, "
              f"not-contain={result.must_not_contain_pass}, "
              f"tools={result.tool_call_pass})")
        if result.tool_call_missing:
            print(f"    Missing tools: {result.tool_call_missing}")
        if result.must_contain_missing:
            print(f"    Missing text:  {result.must_contain_missing}")

    # ── 3. LLM judge ──────────────────────────────────────────────────────
    if judge and case.judge_criteria:
        try:
            result.grade = judge.grade(
                question=case.question,
                answer=result.answer,
                tools_called=result.tools_called,
                judge_criteria=case.judge_criteria,
            )
            if result.grade.is_valid:
                judge_icon = "✅" if result.grade.total >= 14 else ("⚠️ " if result.grade.total >= 10 else "❌")
                if verbose:
                    print(f"  Judge score  : {judge_icon} {result.grade.total}/20 "
                          f"({result.grade.percentage}%)")
                    print(f"    Reasoning  : {result.grade.reasoning[:150]}")
            else:
                if verbose:
                    print(f"  Judge        : ❌ invalid — {result.grade.error[:100]}")
        except Exception as e:
            if verbose:
                print(f"  Judge        : ❌ exception — {e}")

    return result


# ════════════════════════════════════════════════════════════════════════════
# Full eval run
# ════════════════════════════════════════════════════════════════════════════

def run_eval(
    cases: list[EvalCase],
    use_judge: bool = True,
    verbose: bool = True,
    output_path: Optional[Path] = None,
) -> list[EvalResult]:

    print("\n" + "═"*60)
    print("  SalesOS Evaluation Run")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Cases: {len(cases)}  |  Judge: {'enabled' if use_judge else 'disabled'}")
    print("═"*60)

    # ── Init agent ─────────────────────────────────────────────────────────
    print("\n🔧 Initialising agent …")
    try:
        agent = create_sales_agent()
        print("✅ Agent ready")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        sys.exit(1)

    # ── Init judge ─────────────────────────────────────────────────────────
    judge = None
    if use_judge:
        print("🔧 Initialising LLM judge …")
        try:
            judge = LLMJudge(model_name=MODEL_NAME, base_url=LLAMA_SERVER_URL)
            print("✅ Judge ready")
        except Exception as e:
            print(f"⚠️  Judge init failed: {e} — running heuristics only")

    # ── Run all cases ──────────────────────────────────────────────────────
    results = []
    for case in cases:
        result = run_case(case, agent, judge, verbose=verbose)
        results.append(result)

    # ── Summary ─────────────────────────────────────────────────────────────
    _print_summary(results)

    # ── Persist results ────────────────────────────────────────────────────
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = ROOT / "tests" / "eval" / f"results_{ts}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "run_timestamp": datetime.now().isoformat(),
                "model": MODEL_NAME,
                "total_cases": len(results),
                "results": [r.to_dict() for r in results],
            },
            f,
            indent=2,
        )
    print(f"\n💾 Results saved to: {output_path}")
    return results


def _print_summary(results: list[EvalResult]):
    print("\n" + "═"*60)
    print("  SUMMARY")
    print("═"*60)

    total = len(results)
    heuristic_pass = sum(1 for r in results if r.heuristic_pass)
    graded = [r for r in results if r.grade and r.grade.is_valid]

    print(f"\nHeuristic checks : {heuristic_pass}/{total} passed "
          f"({heuristic_pass/total*100:.0f}%)")

    if graded:
        avg_score = sum(r.grade.total for r in graded) / len(graded)
        avg_pct   = sum(r.grade.percentage for r in graded) / len(graded)
        print(f"LLM Judge avg    : {avg_score:.1f}/20 ({avg_pct:.1f}%)")

    # Per-category breakdown
    from tests.eval.eval_cases import CASES_BY_CATEGORY
    cats = set(r.category for r in results)
    if len(cats) > 1:
        print("\nBy category:")
        for cat in sorted(cats):
            cat_results = [r for r in results if r.category == cat]
            cat_pass    = sum(1 for r in cat_results if r.heuristic_pass)
            cat_graded  = [r for r in cat_results if r.grade and r.grade.is_valid]
            judge_str   = ""
            if cat_graded:
                judge_str = f"  judge avg {sum(r.grade.total for r in cat_graded)/len(cat_graded):.1f}/20"
            print(f"  {cat:<18} {cat_pass}/{len(cat_results)} heuristic{judge_str}")

    # Failures
    failures = [r for r in results if not r.heuristic_pass or r.error]
    if failures:
        print("\nFailed cases:")
        for r in failures:
            issues = []
            if r.error:
                issues.append(f"error: {r.error[:60]}")
            if r.tool_call_missing:
                issues.append(f"missing tools: {r.tool_call_missing}")
            if r.must_contain_missing:
                issues.append(f"missing text: {r.must_contain_missing}")
            print(f"  [{r.case_id}] {', '.join(issues)}")

    print()


# ════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SalesOS Evaluation Runner")
    parser.add_argument("--category", choices=list(CASES_BY_CATEGORY.keys()),
                        help="Run only this category")
    parser.add_argument("--case", help="Run a single case by ID (e.g. rag_01)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM-as-judge, run heuristics only")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-case output")
    parser.add_argument("--output", type=Path,
                        help="Path to save JSON results")
    args = parser.parse_args()

    # Select cases
    if args.case:
        cases = [c for c in ALL_CASES if c.id == args.case]
        if not cases:
            print(f"❌ No case found with id '{args.case}'")
            print(f"   Available: {[c.id for c in ALL_CASES]}")
            sys.exit(1)
    elif args.category:
        cases = CASES_BY_CATEGORY[args.category]
    else:
        cases = ALL_CASES

    run_eval(
        cases=cases,
        use_judge=not args.no_judge,
        verbose=not args.quiet,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
