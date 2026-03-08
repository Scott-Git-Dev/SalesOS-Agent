"""
tests/eval/eval_report.py

Generates a self-contained HTML report from eval_runner.py JSON output.

Usage:
    python tests/eval/eval_report.py tests/eval/results_20250101_120000.json

Opens the report automatically in your default browser.
No external dependencies beyond the standard library.
"""

import json
import sys
import webbrowser
import tempfile
from pathlib import Path
from datetime import datetime


# ════════════════════════════════════════════════════════════════════════════
# Score helpers
# ════════════════════════════════════════════════════════════════════════════

def _score_color(pct: float) -> str:
    if pct >= 80: return "#10b981"   # green
    if pct >= 60: return "#f59e0b"   # amber
    return "#ef4444"                  # red

def _bool_badge(passed: bool) -> str:
    if passed:
        return '<span style="color:#10b981;font-weight:700">✅ PASS</span>'
    return '<span style="color:#ef4444;font-weight:700">❌ FAIL</span>'


# ════════════════════════════════════════════════════════════════════════════
# HTML builder
# ════════════════════════════════════════════════════════════════════════════

def generate_report(data: dict) -> str:
    results   = data["results"]
    ts        = data.get("run_timestamp", "")
    model     = data.get("model", "unknown")
    total     = len(results)

    heuristic_pass = sum(1 for r in results if r["heuristic_pass"])
    graded         = [r for r in results if r.get("grade") and r["grade"].get("is_valid")]
    avg_judge      = (
        sum(r["grade"]["total"] for r in graded) / len(graded)
        if graded else None
    )

    # ── category breakdown ─────────────────────────────────────────────────
    cats: dict[str, list] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)

    cat_rows_html = ""
    for cat, cat_results in cats.items():
        cat_pass   = sum(1 for r in cat_results if r["heuristic_pass"])
        cat_graded = [r for r in cat_results if r.get("grade") and r["grade"].get("is_valid")]
        judge_cell = (
            f"{sum(r['grade']['total'] for r in cat_graded)/len(cat_graded):.1f}/20"
            if cat_graded else "—"
        )
        cat_pct = cat_pass / len(cat_results) * 100
        cat_rows_html += f"""
        <tr>
          <td><b>{cat}</b></td>
          <td>{cat_pass}/{len(cat_results)}</td>
          <td style="color:{_score_color(cat_pct)}">{cat_pct:.0f}%</td>
          <td>{judge_cell}</td>
        </tr>"""

    # ── individual case rows ────────────────────────────────────────────────
    case_rows_html = ""
    for r in results:
        grade    = r.get("grade") or {}
        j_total  = grade.get("total", "—") if grade.get("is_valid") else "err"
        j_pct    = f"{grade.get('percentage', 0):.0f}%" if grade.get("is_valid") else ""
        j_color  = _score_color(grade.get("percentage", 0)) if grade.get("is_valid") else "#94a3b8"
        j_reason = grade.get("reasoning", "")[:180] if grade else ""

        tools_str = ", ".join(r.get("tools_called", [])) or "(none)"

        issues = []
        if r.get("tool_call_missing"):
            issues.append(f"missing tools: {r['tool_call_missing']}")
        if r.get("must_contain_missing"):
            issues.append(f"missing text: {r['must_contain_missing']}")
        if r.get("must_not_contain_found"):
            issues.append(f"bad text: {r['must_not_contain_found']}")
        if r.get("error"):
            issues.append(f"error: {r['error'][:80]}")
        issues_html = "".join(f'<div class="issue">⚠ {i}</div>' for i in issues)

        answer_preview = (r.get("answer") or "")[:300]

        case_rows_html += f"""
        <tr class="{'fail-row' if not r['heuristic_pass'] else ''}">
          <td><code>{r['case_id']}</code></td>
          <td><span class="cat-badge">{r['category']}</span></td>
          <td class="question-cell">{r['question']}</td>
          <td>{tools_str}</td>
          <td>{_bool_badge(r['heuristic_pass'])}</td>
          <td style="color:{j_color};font-weight:700">{j_total} {j_pct}</td>
          <td>{r.get('latency_s', 0):.1f}s</td>
          <td>
            <details>
              <summary>View</summary>
              <div class="answer-box">{answer_preview}</div>
              {issues_html}
              {'<div class="reasoning">Judge: ' + j_reason + '</div>' if j_reason else ''}
            </details>
          </td>
        </tr>"""

    judge_summary_html = (
        f"<div class='stat'><span>LLM Judge avg</span><b style='color:{_score_color(avg_judge/20*100)}'>"
        f"{avg_judge:.1f}/20 ({avg_judge/20*100:.0f}%)</b></div>"
        if avg_judge is not None else
        "<div class='stat'><span>LLM Judge</span><b>not run</b></div>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SalesOS Eval Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #0f172a; color: #e2e8f0; padding: 24px; }}
  h1   {{ font-size: 24px; margin-bottom: 4px; }}
  .meta {{ font-size: 13px; color: #94a3b8; margin-bottom: 32px; }}
  .stats {{ display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 32px; }}
  .stat {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px;
           padding: 16px 24px; min-width: 180px; }}
  .stat span {{ display: block; font-size: 12px; color: #94a3b8; margin-bottom: 4px; }}
  .stat b {{ font-size: 22px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 24px; font-size: 13px; }}
  th  {{ background: #1e293b; padding: 10px 12px; text-align: left;
         border-bottom: 1px solid #334155; color: #94a3b8; font-weight: 600; }}
  td  {{ padding: 10px 12px; border-bottom: 1px solid #1e293b; vertical-align: top; }}
  tr:hover td {{ background: rgba(255,255,255,0.03); }}
  .fail-row td {{ background: rgba(239,68,68,0.06); }}
  .question-cell {{ max-width: 260px; }}
  .cat-badge {{ background: #334155; border-radius: 4px; padding: 2px 6px;
                font-size: 11px; font-weight: 600; }}
  .answer-box {{ background: #0f172a; border: 1px solid #334155; border-radius: 6px;
                 padding: 10px; margin-top: 8px; font-size: 12px; white-space: pre-wrap;
                 max-height: 200px; overflow-y: auto; }}
  .issue {{ color: #f59e0b; font-size: 12px; margin-top: 4px; }}
  .reasoning {{ font-size: 12px; color: #94a3b8; margin-top: 6px; font-style: italic; }}
  details summary {{ cursor: pointer; color: #3b82f6; font-size: 12px; }}
  h2 {{ font-size: 16px; color: #94a3b8; margin-top: 32px; margin-bottom: 8px; }}
</style>
</head>
<body>
  <h1>📊 SalesOS Evaluation Report</h1>
  <div class="meta">Model: {model} &nbsp;|&nbsp; Run: {ts} &nbsp;|&nbsp; {total} cases</div>

  <div class="stats">
    <div class="stat">
      <span>Heuristic pass rate</span>
      <b style="color:{_score_color(heuristic_pass/total*100)}">{heuristic_pass}/{total} ({heuristic_pass/total*100:.0f}%)</b>
    </div>
    {judge_summary_html}
    <div class="stat">
      <span>Cases with errors</span>
      <b style="color:#ef4444">{sum(1 for r in results if r.get('error'))}</b>
    </div>
    <div class="stat">
      <span>Avg latency</span>
      <b>{sum(r.get('latency_s',0) for r in results)/total:.1f}s</b>
    </div>
  </div>

  <h2>By Category</h2>
  <table style="width:auto">
    <tr><th>Category</th><th>Heuristic</th><th>Pass %</th><th>Judge avg</th></tr>
    {cat_rows_html}
  </table>

  <h2>All Cases</h2>
  <table>
    <tr>
      <th>ID</th>
      <th>Cat</th>
      <th>Question</th>
      <th>Tools called</th>
      <th>Heuristic</th>
      <th>Judge</th>
      <th>Latency</th>
      <th>Details</th>
    </tr>
    {case_rows_html}
  </table>
</body>
</html>"""


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_report.py <results_json_file>")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        sys.exit(1)

    data    = json.loads(json_path.read_text())
    html    = generate_report(data)

    out_path = json_path.with_suffix(".html")
    out_path.write_text(html, encoding="utf-8")

    print(f"✅ Report written to: {out_path}")
    webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
