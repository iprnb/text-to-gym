"""
HTML formatting helpers for every display surface in the Gradio UI.

All functions return an HTML string ready to be set on a ``gr.HTML``
component.  Keeping them here prevents the pipeline modules from being
cluttered with presentation logic.

Theme-compatibility notes
-------------------------
Gradio's Soft theme exposes CSS custom properties (e.g. --color-accent,
--background-fill-primary) but these are not reliably available inside
gr.HTML snippets.  We therefore use a small set of CSS classes injected
via a single <style> block (``_THEME_CSS``) that defines light-mode
values and overrides them inside a ``[data-testid]`` dark-mode selector.
All helper functions include ``_THEME_CSS`` in their output so the styles
are always present regardless of which component renders first.
"""

from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Theme-aware CSS
# ---------------------------------------------------------------------------

_THEME_CSS = """
<style>
  /* ── Layout ─────────────────────────────────────────────────────── */
  .ttg-card         { padding: 18px; border-radius: 10px; margin-bottom: 10px; }
  /* Light-mode card backgrounds — all have WCAG AA contrast for body text */
  .ttg-card-info    { background: #dbeafe; border: 1px solid #93c5fd; color: #1e3a5f; }
  .ttg-card-success { background: #dcfce7; border: 1px solid #86efac; color: #14532d; }
  .ttg-card-warning { background: #fef9c3; border: 1px solid #fde047; color: #713f12; }
  .ttg-card-error   { background: #fee2e2; border: 1px solid #fca5a5; color: #7f1d1d; }
  .ttg-card-neutral { background: var(--background-fill-secondary, #f3f4f6);
                      border: 1px solid var(--border-color-primary, #d1d5db); color: inherit; }

  /* ── Typography ──────────────────────────────────────────────────── */
  .ttg-h3  { font-size: 1.1rem; font-weight: 700; margin: 0 0 8px 0; }
  .ttg-h4  { font-size: 0.95rem; font-weight: 700; margin: 12px 0 4px 0; }
  /* ttg-p intentionally has no color — inherits from the card (contrast safe) */
  .ttg-p   { margin: 4px 0; line-height: 1.55; }
  .ttg-em  { font-style: italic; opacity: 0.85; }
  .ttg-muted { opacity: 0.72; }

  /* ── Semantic colour overrides (only used when NOT inside a coloured card) ── */
  .ttg-accent  { color: #1d4ed8; }   /* blue-700  — 7:1 on white */
  .ttg-success { color: #15803d; }   /* green-700 — 5.6:1 on white */
  .ttg-warning { color: #92400e; }   /* amber-800 — 7:1 on white */
  .ttg-error   { color: #991b1b; }   /* red-800   — 7:1 on white */

  /* ── Question cards ──────────────────────────────────────────────── */
  .ttg-q-card   { margin: 12px 0; padding: 14px; border-radius: 6px;
                  background: var(--background-fill-primary, #fff);
                  border-left: 4px solid #3b82f6;
                  color: inherit; }
  .ttg-q-high   { border-left-color: #dc2626 !important; }
  .ttg-q-medium { border-left-color: #d97706 !important; }
  .ttg-q-low    { border-left-color: #0d9488 !important; }

  /* ── Mismatch cards ──────────────────────────────────────────────── */
  .ttg-mismatch          { margin: 10px 0; padding: 14px; border-radius: 6px; border-left: 4px solid; }
  .ttg-mismatch-critical { background: #fee2e2; border-left-color: #b91c1c; color: #7f1d1d; }
  .ttg-mismatch-warning  { background: #fef9c3; border-left-color: #b45309; color: #713f12; }
  .ttg-mismatch-info     { background: #dbeafe; border-left-color: #1d4ed8; color: #1e3a5f; }

  /* ── Badges ──────────────────────────────────────────────────────── */
  .ttg-badge-critical { color: #b91c1c; font-weight: 700; }
  .ttg-badge-warning  { color: #b45309; font-weight: 700; }
  .ttg-badge-info     { color: #1d4ed8; font-weight: 700; }

  /* ── Misc ────────────────────────────────────────────────────────── */
  .ttg-pre  { background: var(--background-fill-secondary, #f3f4f6);
              color: inherit;
              padding: 8px; border-radius: 4px; font-size: 0.85em;
              overflow-x: auto; white-space: pre-wrap; }
  .ttg-code { background: var(--background-fill-secondary, #f3f4f6);
              color: inherit;
              padding: 2px 6px; border-radius: 3px; font-family: monospace; font-size: 0.9em; }
  .ttg-hr   { border: none; border-top: 1px solid var(--border-color-primary, #d1d5db); margin: 12px 0; }

  /* ── Dark-mode overrides ─────────────────────────────────────────── */
  /* Cards: dark enough backgrounds, light text — all WCAG AA */
  .dark .ttg-card-info    { background: #1e3a5f; border-color: #2d5a8e; color: #bfdbfe; }
  .dark .ttg-card-success { background: #14532d; border-color: #166534; color: #bbf7d0; }
  .dark .ttg-card-warning { background: #451a03; border-color: #78350f; color: #fde68a; }
  .dark .ttg-card-error   { background: #7f1d1d; border-color: #991b1b; color: #fecaca; }
  .dark .ttg-card-neutral { background: var(--background-fill-secondary, #1f2937); color: #f3f4f6; }

  .dark .ttg-q-card { background: var(--background-fill-primary, #111827); color: #f3f4f6; }

  .dark .ttg-mismatch-critical { background: #7f1d1d; color: #fecaca; }
  .dark .ttg-mismatch-warning  { background: #451a03; color: #fde68a; }
  .dark .ttg-mismatch-info     { background: #1e3a5f; color: #bfdbfe; }

  /* Semantic colours on dark backgrounds */
  .dark .ttg-accent  { color: #93c5fd; }
  .dark .ttg-success { color: #86efac; }
  .dark .ttg-warning { color: #fde68a; }
  .dark .ttg-error   { color: #fca5a5; }

  .dark .ttg-badge-critical { color: #fca5a5; }
  .dark .ttg-badge-warning  { color: #fde68a; }
  .dark .ttg-badge-info     { color: #93c5fd; }

  .dark .ttg-pre, .dark .ttg-code { background: var(--background-fill-secondary, #1f2937); color: #f3f4f6; }

  /* ── Pipeline progress bar (used in merged Step 5) ───────────────── */
  .ttg-progress { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
  .ttg-step-badge {
    padding: 4px 12px; border-radius: 20px; font-size: 0.82rem; font-weight: 600;
    background: #e5e7eb; color: #374151;
  }
  .ttg-step-badge.done    { background: #dcfce7; color: #14532d; }
  .ttg-step-badge.running { background: #dbeafe; color: #1e3a5f; }
  .ttg-step-badge.error   { background: #fee2e2; color: #7f1d1d; }
  .dark .ttg-step-badge           { background: #374151; color: #d1d5db; }
  .dark .ttg-step-badge.done      { background: #14532d; color: #bbf7d0; }
  .dark .ttg-step-badge.running   { background: #1e3a5f; color: #bfdbfe; }
  .dark .ttg-step-badge.error     { background: #7f1d1d; color: #fecaca; }
</style>
"""


# ---------------------------------------------------------------------------
# Step 2 – Clarifying questions
# ---------------------------------------------------------------------------

def format_questions(questions_data: Optional[Dict]) -> str:
    """Render the LLM-generated clarifying questions as readable HTML."""
    if not questions_data:
        return _error_box("Could not parse LLM response. Check the Debug tab for raw output.")

    html = _THEME_CSS
    html += "<div class='ttg-card ttg-card-neutral'>"

    analysis = questions_data.get("analysis", {})
    if analysis:
        html += "<h3 class='ttg-h3 ttg-accent'>📊 Analysis</h3>"
        html += _p(f"<strong>Scope:</strong> {analysis.get('scope_assessment', 'N/A')}")
        html += _p(f"<strong>Decision-Maker:</strong> {analysis.get('decision_maker_assessment', 'N/A')}")

        for label, key in [
            ("Critical Ambiguities", "critical_ambiguities"),
            ("Missing Information", "missing_information"),
            ("Contradictions Found", "contradictions"),
        ]:
            items = analysis.get(key)
            if items:
                html += _p(f"<strong>{label}:</strong>") + _ul(items)

    questions = questions_data.get("questions", [])
    if questions:
        html += "<hr class='ttg-hr'><h3 class='ttg-h3 ttg-accent'>❓ Clarifying Questions</h3>"
        priority_class = {"high": "ttg-q-high", "medium": "ttg-q-medium", "low": "ttg-q-low"}

        for i, q in enumerate(questions, 1):
            pclass = priority_class.get(q.get("priority", "medium"), "ttg-q-medium")
            html += f"<div class='ttg-q-card {pclass}'>"
            html += _p(f"<strong>Q{i}. [{q.get('category','other').upper()}]</strong> {q.get('question_text','N/A')}")

            fmt = q.get("format", "free_form")
            if fmt == "multiple_choice" and q.get("options"):
                html += _ul(q["options"])
            elif fmt == "yes_no":
                html += _p("<em class='ttg-em'>Answer: Yes / No</em>")
                if q.get("follow_up_question"):
                    cond = q.get("follow_up_condition", "yes")
                    html += _p(f"<em class='ttg-em'>If {cond}: {q['follow_up_question']}</em>")
            else:
                html += _p("<em class='ttg-em'>[Free-form text answer]</em>")

            html += "</div>"
    else:
        html += _p("<em class='ttg-em'>No questions generated.</em>")

    if questions_data.get("trajectory_needed"):
        html += "<div class='ttg-card ttg-card-warning'>"
        html += _p("<strong>📝 Trajectory Example Requested:</strong>")
        html += _p(questions_data.get("trajectory_request", "Please provide an example scenario."))
        html += "</div>"

    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# Step 3 – Answers summary
# ---------------------------------------------------------------------------

def format_answers_summary(answers: dict, new_count: int) -> str:
    """Render a colour-coded summary of all collected Q&A pairs."""
    html = _THEME_CSS
    html += "<div class='ttg-card ttg-card-success'>"
    html += "<h3 class='ttg-h3 ttg-success'>✅ Answers Updated</h3>"
    html += _p(f"Added {new_count} new answers. Total answers: {len(answers)}")
    html += "<hr class='ttg-hr'>"

    for key, value in answers.items():
        label_class = "ttg-accent" if key.startswith("FQ") else "ttg-success"
        round_label = f" (Round {value.get('round', 1)})" if key.startswith("FQ") else ""
        html += f"<p class='ttg-p'><strong class='{label_class}'>{key}{round_label}:</strong> {value.get('question','N/A')}</p>"
        html += f"<p class='ttg-p ttg-muted' style='margin-left:18px;'><em>Answer: {value.get('answer','N/A')}</em></p>"

    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# Step 4 – Specification display
# ---------------------------------------------------------------------------

def format_specification(validation_data: Dict) -> str:
    """Render a complete, validated specification."""
    spec = validation_data.get("complete_specification", {})

    html = _THEME_CSS
    html += "<div class='ttg-card ttg-card-success'>"
    html += "<h3 class='ttg-h3 ttg-success'>✅ Specification Complete!</h3>"
    html += _p(validation_data.get("final_confirmation", ""))
    html += "<hr class='ttg-hr'>"

    html += "<h4 class='ttg-h4 ttg-accent'>📋 Domain Summary</h4>"
    html += _p(spec.get("domain_summary", "N/A"))

    for title, emoji, key in [
        ("Action Space", "🎮", "action_space"),
        ("Observation Space", "👁️", "observation_space"),
        ("Reward Structure", "🎯", "reward_structure"),
    ]:
        section = spec.get(key, {})
        html += f"<h4 class='ttg-h4 ttg-accent'>{emoji} {title}</h4>"
        html += _p(f"<strong>Type:</strong> {section.get('type','N/A')}")
        html += _p(section.get("description", "N/A"))

    rh = validation_data.get("reward_hacking_check", {})
    if rh.get("potential_exploits"):
        html += "<div class='ttg-card ttg-card-warning'>"
        html += "<h4 class='ttg-h4 ttg-warning'>⚠️ Reward Hacking Check</h4>"
        html += _p("<strong>Potential Exploits:</strong>") + _ul(rh.get("potential_exploits", []))
        html += _p("<strong>Suggested Mitigations:</strong>") + _ul(rh.get("mitigations", []))
        html += "</div>"

    html += "<hr class='ttg-hr'>"
    html += "<p class='ttg-p ttg-success'><strong>✨ Ready to generate environment code! Go to Step 5.</strong></p>"
    html += "</div>"
    return html


def format_validation_issues(validation_data: Dict) -> str:
    """Render follow-up questions when the spec is not yet complete."""
    html = _THEME_CSS
    html += "<div class='ttg-card ttg-card-warning'>"
    html += "<h3 class='ttg-h3 ttg-warning'>⚠️ Additional Clarification Needed</h3>"

    issues = validation_data.get("issues", {})
    if issues.get("ambiguities"):
        html += "<h4 class='ttg-h4'>Ambiguities:</h4>" + _ul(issues["ambiguities"])
    if issues.get("missing"):
        html += "<h4 class='ttg-h4'>Missing Information:</h4>" + _ul(issues["missing"])

    follow_ups = validation_data.get("follow_up_questions", [])
    if follow_ups:
        html += "<hr class='ttg-hr'>"
        html += "<h4 class='ttg-h4'>Follow-up Questions:</h4>"
        for i, fq in enumerate(follow_ups, 1):
            html += "<div class='ttg-q-card ttg-q-medium'>"
            html += _p(f"<strong>FQ{i}.</strong> {fq.get('question','N/A')}")
            html += _p(f"<em class='ttg-em'>Reason: {fq.get('reason','N/A')}</em>")
            html += "</div>"

    html += "<hr class='ttg-hr'>"
    html += _p("<strong>Please answer the follow-up questions below and click Submit.</strong>")
    html += "</div>"
    return html


def format_round_limit() -> str:
    """Message shown when the 2-round validation limit is reached."""
    return _THEME_CSS + """
<div class='ttg-card ttg-card-info'>
    <h3 class='ttg-h3 ttg-accent'>ℹ️ Validation Round Limit Reached</h3>
    <p class='ttg-p'>We have completed 2 rounds of clarification. Proceeding to generate the environment with the current understanding.</p>
    <p class='ttg-p'>Some parameters may use reasonable defaults where information was not fully provided.</p>
    <p class='ttg-p ttg-success'><strong>✨ You can now go to Step 5 to generate the environment code.</strong></p>
</div>
"""


# ---------------------------------------------------------------------------
# Merged Step 5 – pipeline progress card
# ---------------------------------------------------------------------------

def format_pipeline_progress(
    gen_status: str,      # "pending" | "running" | "done" | "error"
    check_status: str,
    test_status: str,
    message: str = "",
    fixes_applied: int = 0,
    test_rounds: int = 0,
    test_passed: bool = False,
) -> str:
    """
    Render a live progress card for the merged Generate → Check → Test pipeline.
    ``*_status`` is one of "pending" | "running" | "done" | "error".
    """
    def badge(label: str, state: str) -> str:
        return f"<span class='ttg-step-badge {state}'>{label}</span>"

    overall_class = (
        "ttg-card-success" if test_passed
        else "ttg-card-error" if "error" in (gen_status, check_status, test_status)
        else "ttg-card-info"
    )

    html = _THEME_CSS
    html += f"<div class='ttg-card {overall_class}'>"
    html += "<div class='ttg-progress'>"
    html += badge("⚙️ Generate", gen_status)
    html += badge("🔬 Spec Check", check_status)
    html += badge("🧪 Runtime Test", test_status)
    html += "</div>"
    if message:
        html += _p(f"<strong>{message}</strong>")
    if fixes_applied:
        html += _p(f"🔧 Auto-applied {fixes_applied} spec fix(es)")
    if test_rounds:
        html += _p(f"🔄 Debug rounds used: {test_rounds}")
    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# Step 5 – Code generation status
# ---------------------------------------------------------------------------

def format_code_status(class_name: str, char_count: int, line_count: int, syntax_ok: bool, syntax_msg: str) -> str:
    """Status card shown after environment code is generated."""
    if syntax_ok:
        return _THEME_CSS + f"""
<div class='ttg-card ttg-card-success'>
    <h3 class='ttg-h3 ttg-success'>✅ Environment Code Generated!</h3>
    <p class='ttg-p'>{syntax_msg}</p>
    <p class='ttg-p'><strong>Stats:</strong> {char_count:,} characters, {line_count:,} lines</p>
    <p class='ttg-p'><strong>Class name:</strong> <code class='ttg-code'>{class_name}</code></p>
    <hr class='ttg-hr'>
    <p class='ttg-p'><strong>Next steps:</strong></p>
    <ol>
        <li class='ttg-p'>Run <strong>Step 5.5</strong> to validate code against the spec</li>
        <li class='ttg-p'>Run <strong>Step 5.7</strong> to test it with SB3</li>
        <li class='ttg-p'>Train in <strong>Step 6</strong> once tests pass</li>
    </ol>
</div>
"""
    return _THEME_CSS + f"""
<div class='ttg-card ttg-card-error'>
    <h3 class='ttg-h3 ttg-error'>⚠️ Code Generated with Syntax Issues</h3>
    <p class='ttg-p'>{syntax_msg}</p>
    <p class='ttg-p'>The code may need manual fixes. Please review and correct errors.</p>
</div>
"""


# ---------------------------------------------------------------------------
# Step 5.5 – Code-vs-spec validation report
# ---------------------------------------------------------------------------

def format_code_validation_report(result: dict) -> str:
    """Render the code-vs-spec mismatch report."""
    if not result:
        return _error_box("Could not generate validation report.")

    mismatches = result.get("mismatches", [])
    critical = result.get("critical_count", 0)
    warning = result.get("warning_count", 0)
    info = result.get("info_count", 0)

    header_class = "ttg-card-success" if critical == 0 else "ttg-card-error"
    header_icon = "✅" if critical == 0 else "⚠️"

    html = _THEME_CSS
    html += f"<div class='ttg-card {header_class}'>"
    html += f"<h3 class='ttg-h3'>{header_icon} Code Validation Report</h3>"
    html += _p(result.get("summary", ""))
    html += (
        f"<p class='ttg-p'>"
        f"<span class='ttg-badge-critical'>Critical: {critical}</span> &nbsp;|&nbsp; "
        f"<span class='ttg-badge-warning'>Warnings: {warning}</span> &nbsp;|&nbsp; "
        f"<span class='ttg-badge-info'>Info: {info}</span>"
        f"</p></div>"
    )

    if not mismatches:
        html += "<div class='ttg-card ttg-card-success'>"
        html += "<p class='ttg-p ttg-success'>✅ No mismatches found! Code matches specification.</p>"
        html += "</div>"
        return html

    html += "<div style='margin-top:12px;'><h4 class='ttg-h4'>Mismatches Found:</h4>"

    severity_classes = {
        "critical": ("ttg-mismatch-critical", "ttg-badge-critical", "🔴"),
        "warning":  ("ttg-mismatch-warning",  "ttg-badge-warning",  "🟡"),
        "info":     ("ttg-mismatch-info",      "ttg-badge-info",     "🔵"),
    }

    for m in mismatches:
        sev = m.get("severity", "info")
        mismatch_class, badge_class, icon = severity_classes.get(sev, ("", "", "⚪"))
        mid = m.get("id", "unknown")
        html += f"<div class='ttg-mismatch {mismatch_class}'>"
        html += (
            f"<p class='ttg-p'>"
            f"<span class='{badge_class}'>{icon} ID: "
            f"<code class='ttg-code'>{mid}</code></span> "
            f"<strong>[{m.get('category','').upper()}] {m.get('description','')}</strong></p>"
        )
        html += _p(f"📋 <strong>Spec says:</strong> {m.get('spec_value','N/A')}")
        html += _p(f"💻 <strong>Code has:</strong> {m.get('code_value','N/A')}")
        html += _p(f"📍 <em class='ttg-em'>Location: {m.get('location','unknown')}</em>")
        html += _p(f"🔧 <strong>Fix:</strong> {m.get('fix_suggestion','')}")
        html += "</div>"

    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# Step 5.7 – Runtime test report
# ---------------------------------------------------------------------------

def format_runtime_test_report(rounds: list, test_steps: int) -> str:
    """Render the iterative runtime-test report."""
    if not rounds:
        return _error_box("No test rounds completed.")

    last_round = rounds[-1]
    final_success = last_round.get("success", False)
    header_class = "ttg-card-success" if final_success else "ttg-card-error"
    header_icon = "✅" if final_success else "❌"
    header_msg = (
        f"Code passed full SB3 compatibility test ({test_steps:,} training steps)!"
        if final_success
        else f"Code still has runtime errors after {len(rounds)} round(s)"
    )

    html = _THEME_CSS
    html += f"<div class='ttg-card {header_class}'>"
    html += f"<h3 class='ttg-h3'>{header_icon} Runtime Test Report</h3>"
    html += _p(header_msg)
    html += _p(f"<strong>Tests:</strong> Instantiation → Reset → Manual steps → SB3 env_checker → SB3 model creation → SB3 training ({test_steps:,} steps)")
    html += _p(f"<strong>Rounds:</strong> {len(rounds)}")
    html += "</div>"

    for r in rounds:
        round_num = r["round"]
        success = r.get("success", False)
        if success:
            html += "<div class='ttg-card ttg-card-success'>"
            html += f"<p class='ttg-p ttg-success'><strong>Round {round_num}:</strong> ✅ Passed all tests including SB3 training</p>"
            html += "</div>"
        else:
            error_snippet = r.get("error", "")[:300]
            diagnosis = r.get("diagnosis", "")
            html += "<div class='ttg-card ttg-card-error'>"
            html += f"<p class='ttg-p ttg-error'><strong>Round {round_num}:</strong> ❌ Runtime error</p>"
            html += _p("<strong>Error:</strong>")
            html += f"<pre class='ttg-pre ttg-error'>{error_snippet}...</pre>"
            html += _p(f"<strong>Diagnosis:</strong> {diagnosis}")
            html += "</div>"

    footer_class = "ttg-card-info" if final_success else "ttg-card-warning"
    footer_msg = (
        "✨ Code is SB3-compatible and ready for training!"
        if final_success
        else "⚠️ Code still has issues. You can try again or proceed to the playground where you can re-test."
    )
    html += f"<div class='ttg-card {footer_class}'>"
    html += f"<p class='ttg-p'><strong>{footer_msg}</strong></p>"
    html += "</div>"

    return html


# ---------------------------------------------------------------------------
# Step 6 – Training status
# ---------------------------------------------------------------------------

def format_training_status(state: str, message: str) -> str:
    """Render a colour-coded training status card."""
    configs = {
        "running":  ("ttg-card-info",    "ttg-accent",   "⏳"),
        "complete": ("ttg-card-success",  "ttg-success",  "✅"),
        "error":    ("ttg-card-error",    "ttg-error",    "❌"),
        "stopped":  ("ttg-card-warning",  "ttg-warning",  "⏹️"),
        "idle":     ("ttg-card-neutral",  "ttg-muted",    "💤"),
    }
    card_class, text_class, icon = configs.get(state, configs["idle"])
    return (
        _THEME_CSS
        + f"<div class='ttg-card {card_class}'>"
        f"<p class='ttg-p {text_class}' style='margin:0;'>"
        f"<strong>{icon} {message}</strong>"
        f"</p></div>"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _p(text: str, **kwargs) -> str:
    """Render a paragraph. Extra kwargs are ignored (legacy compat)."""
    return f"<p class='ttg-p'>{text}</p>"


def _ul(items: list) -> str:
    li = "".join(f"<li class='ttg-p'>{item}</li>" for item in items)
    return f"<ul style='margin:4px 0 8px 18px;'>{li}</ul>"


def _error_box(message: str) -> str:
    return (
        _THEME_CSS
        + f"<div class='ttg-card ttg-card-error'>"
        f"<p class='ttg-p ttg-error'><strong>⚠️ Error:</strong> {message}</p>"
        f"</div>"
    )
