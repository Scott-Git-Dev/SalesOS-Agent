#!/usr/bin/env python
"""
run_tests.py — SalesOS test & evaluation runner

Runs the full test suite in the right order and prints a combined summary.

Usage
─────
  python run_tests.py                  # unit tests only (fast, no LLM server needed)
  python run_tests.py --integration    # unit + integration tests (needs LLM server)
  python run_tests.py --eval           # unit tests + full eval run (needs LLM server)
  python run_tests.py --eval-only      # skip pytest, run eval only
  python run_tests.py --eval --no-judge # eval with heuristics only (no LLM judge)
  python run_tests.py --category SQL   # eval only SQL cases
  python run_tests.py --case rag_01    # eval a single case

Layers
──────
  Layer 1  Unit tests (pytest)
           Fast, fully mocked, no LLM server required.
           Tests individual functions in tools/, agent.py, etc.

  Layer 2  Integration tests (pytest --run-integration)
           Requires a running local LLM server.
           Tests the real agent end-to-end with actual LLM calls.

  Layer 3  Eval framework (eval_runner.py)
           Requires a running local LLM server.
           Runs golden-set questions, scores with heuristics + LLM judge,
           writes JSON results, and generates an HTML report.

Quick start
───────────
  1. Start your LLM server:
       ollama serve   OR   ./llama-server --model your-model.gguf

  2. Set up data (first time only):
       python setup_sales_db.py
       python setup_knowledge_base.py

  3. Run everything:
       python run_tests.py --eval
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run_pytest(integration: bool = False) -> int:
    """Run pytest and return exit code."""
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    if integration:
        cmd.append("--run-integration")
    print("\n" + "═" * 60)
    print(f"  Running pytest {'(+ integration)' if integration else '(unit only)'}")
    print("═" * 60)
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def run_eval(
    category: str = None,
    case: str = None,
    no_judge: bool = False,
    output: Path = None,
) -> int:
    """Run the eval framework and return exit code."""
    cmd = [sys.executable, "tests/eval/eval_runner.py"]
    if category:
        cmd += ["--category", category]
    if case:
        cmd += ["--case", case]
    if no_judge:
        cmd.append("--no-judge")
    if output:
        cmd += ["--output", str(output)]

    print("\n" + "═" * 60)
    print("  Running eval framework")
    print("═" * 60)
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="SalesOS unified test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--integration", action="store_true",
                        help="Also run integration tests (requires LLM server)")
    parser.add_argument("--eval", action="store_true",
                        help="Also run the eval framework (requires LLM server)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip pytest, run only the eval framework")
    parser.add_argument("--no-judge", action="store_true",
                        help="Eval with heuristics only, skip LLM judge")
    parser.add_argument("--category",
                        choices=["SQL", "RAG", "MULTI_TOOL", "SCOPE", "PROMPT_ADHERENCE"],
                        help="Eval: run only this category")
    parser.add_argument("--case", help="Eval: run a single case by ID (e.g. rag_01)")
    args = parser.parse_args()

    exit_codes = []

    # ── pytest ────────────────────────────────────────────────────────────
    if not args.eval_only:
        code = run_pytest(integration=args.integration)
        exit_codes.append(code)

    # ── eval ──────────────────────────────────────────────────────────────
    if args.eval or args.eval_only:
        code = run_eval(
            category=args.category,
            case=args.case,
            no_judge=args.no_judge,
        )
        exit_codes.append(code)

    if not exit_codes:
        # Default: just unit tests
        exit_codes.append(run_pytest(integration=False))

    # ── report ────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    if all(c == 0 for c in exit_codes):
        print("  ✅ All checks passed")
    else:
        print(f"  ❌ Some checks failed (exit codes: {exit_codes})")
    print("═" * 60)

    sys.exit(max(exit_codes))


if __name__ == "__main__":
    main()
