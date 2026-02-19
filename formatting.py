"""
HTML formatting helpers for every display surface in the Gradio UI.

All functions return an HTML string ready to be set on a ``gr.HTML``
component.  Keeping them here prevents the pipeline modules from being
cluttered with presentation logic.
"""

from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Step 2 – Clarifying questions
# ---------------------------------------------------------------------------

def format_questions(questions_data: Optional[Dict]) -> str:
    """Render the LLM-generated clarifying questions as readable HTML."""
    if not questions_data:
        return _error_box("Could not parse LLM response. Check the Debug tab for raw output.")

    html = "<div style='padding: 20px; background-color: #f5f5f5; border-radius: 8px;'>"

    analysis = questions_data.get("analysis", {})
    if analysis:
        html += "<h3 style='color: #1976d2 !important;'>📊 Analysis</h3>"
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
        html += "<hr style='border-color: #ccc;'><h3 style='color: #1976d2 !important;'>❓ Clarifying Questions</h3>"
        priority_color = {"high": "#ff6b6b", "medium": "#ffa500", "low": "#4ecdc4"}

        for i, q in enumerate(questions, 1):
            color = priority_color.get(q.get("priority", "medium"), "#666")
            html += f"<div style='margin:15px 0;padding:15px;background:white;border-left:4px solid {color};border-radius:4px;'>"
            html += _p(f"<strong>Q{i}. [{q.get('category','other').upper()}]</strong> {q.get('question_text','N/A')}", margin=0)

            fmt = q.get("format", "free_form")
            if fmt == "multiple_choice" and q.get("options"):
                html += _ul(q["options"])
            elif fmt == "yes_no":
                html += _p("<em>Answer: Yes / No</em>", color="#333333")
                if q.get("follow_up_question"):
                    cond = q.get("follow_up_condition", "yes")
                    html += f"<p style='margin-left:20px;color:#555555 !important;'><em>If {cond}: {q['follow_up_question']}</em></p>"
            else:
                html += _p("<em>[Free-form text answer]</em>", color="#333333")

            html += "</div>"
    else:
        html += _p("<em>No questions generated.</em>")

    if questions_data.get("trajectory_needed"):
        html += "<div style='margin:15px 0;padding:15px;background:#fff3cd;border-radius:4px;'>"
        html += _p("<strong>📝 Trajectory Example Requested:</strong>")
        html += _p(questions_data.get("trajectory_request", "Please provide an example scenario."), color="#333333")
        html += "</div>"

    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# Step 3 – Answers summary
# ---------------------------------------------------------------------------

def format_answers_summary(answers: dict, new_count: int) -> str:
    """Render a colour-coded summary of all collected Q&A pairs."""
    html = "<div style='padding:20px;background-color:#e8f5e9;border-radius:8px;'>"
    html += "<h3 style='color:#2e7d32 !important;'>✅ Answers Updated</h3>"
    html += _p(f"Added {new_count} new answers. Total answers: {len(answers)}")
    html += "<hr style='border-color:#81c784;'>"

    for key, value in answers.items():
        color = "#1976d2" if key.startswith("FQ") else "#2e7d32"
        round_label = f" (Round {value.get('round', 1)})" if key.startswith("FQ") else ""
        html += f"<p style='color:#000000 !important;'><strong style='color:{color} !important;'>{key}{round_label}:</strong> {value.get('question','N/A')}</p>"
        html += f"<p style='margin-left:20px;color:#424242 !important;'><em>Answer: {value.get('answer','N/A')}</em></p>"

    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# Step 4 – Specification display
# ---------------------------------------------------------------------------

def format_specification(validation_data: Dict) -> str:
    """Render a complete, validated specification."""
    spec = validation_data.get("complete_specification", {})

    html = "<div style='padding:20px;background-color:#e8f5e9;border-radius:8px;'>"
    html += "<h3 style='color:#2e7d32 !important;'>✅ Specification Complete!</h3>"
    html += _p(validation_data.get("final_confirmation", ""))
    html += "<hr style='border-color:#81c784;'>"

    html += "<h4 style='color:#1976d2 !important;'>📋 Domain Summary</h4>"
    html += _p(spec.get("domain_summary", "N/A"))

    for title, emoji, key in [
        ("Action Space", "🎮", "action_space"),
        ("Observation Space", "👁️", "observation_space"),
        ("Reward Structure", "🎯", "reward_structure"),
    ]:
        section = spec.get(key, {})
        html += f"<h4 style='color:#1976d2 !important;'>{emoji} {title}</h4>"
        html += _p(f"<strong>Type:</strong> {section.get('type','N/A')}")
        html += _p(section.get("description", "N/A"))

    rh = validation_data.get("reward_hacking_check", {})
    if rh.get("potential_exploits"):
        html += "<div style='margin:15px 0;padding:15px;background:#fff3cd;border-radius:4px;'>"
        html += "<h4 style='color:#f57c00 !important;'>⚠️ Reward Hacking Check</h4>"
        html += _p("<strong>Potential Exploits:</strong>") + _ul(rh.get("potential_exploits", []))
        html += _p("<strong>Suggested Mitigations:</strong>") + _ul(rh.get("mitigations", []))
        html += "</div>"

    html += "<hr style='border-color:#81c784;'>"
    html += "<p style='color:#2e7d32 !important;'><strong>✨ Ready to generate environment code! Go to Step 5.</strong></p>"
    html += "</div>"
    return html


def format_validation_issues(validation_data: Dict) -> str:
    """Render follow-up questions when the spec is not yet complete."""
    html = "<div style='padding:20px;background-color:#fff3cd;border-radius:8px;'>"
    html += "<h3 style='color:#f57c00 !important;'>⚠️ Additional Clarification Needed</h3>"

    issues = validation_data.get("issues", {})
    if issues.get("ambiguities"):
        html += "<h4 style='color:#000000 !important;'>Ambiguities:</h4>" + _ul(issues["ambiguities"])
    if issues.get("missing"):
        html += "<h4 style='color:#000000 !important;'>Missing Information:</h4>" + _ul(issues["missing"])

    follow_ups = validation_data.get("follow_up_questions", [])
    if follow_ups:
        html += "<hr style='border-color:#ffb74d;'>"
        html += "<h4 style='color:#000000 !important;'>Follow-up Questions:</h4>"
        for i, fq in enumerate(follow_ups, 1):
            html += "<div style='margin:10px 0;padding:12px;background:white;border-left:4px solid #ffa500;border-radius:4px;'>"
            html += _p(f"<strong>FQ{i}.</strong> {fq.get('question','N/A')}")
            html += f"<p style='margin-left:20px;color:#666666 !important;'><em>Reason: {fq.get('reason','N/A')}</em></p>"
            html += "</div>"

    html += "<hr style='border-color:#ffb74d;'>"
    html += _p("<strong>Please answer the follow-up questions below and click Submit.</strong>")
    html += "</div>"
    return html


def format_round_limit() -> str:
    """Message shown when the 2-round validation limit is reached."""
    return """
<div style='padding:20px;background-color:#e3f2fd;border-radius:8px;'>
    <h3 style='color:#1976d2 !important;'>ℹ️ Validation Round Limit Reached</h3>
    <p style='color:#000000 !important;'>We have completed 2 rounds of clarification. Proceeding to generate the environment with the current understanding.</p>
    <p style='color:#000000 !important;'>Some parameters may use reasonable defaults where information was not fully provided.</p>
    <p style='color:#2e7d32 !important;'><strong>✨ You can now go to Step 5 to generate the environment code.</strong></p>
</div>
"""


# ---------------------------------------------------------------------------
# Step 5 – Code generation status
# ---------------------------------------------------------------------------

def format_code_status(class_name: str, char_count: int, line_count: int, syntax_ok: bool, syntax_msg: str) -> str:
    """Status card shown after environment code is generated."""
    if syntax_ok:
        return f"""
<div style='padding:20px;background-color:#e8f5e9;border-radius:8px;'>
    <h3 style='color:#2e7d32 !important;'>✅ Environment Code Generated!</h3>
    <p style='color:#000000 !important;'>{syntax_msg}</p>
    <p style='color:#000000 !important;'><strong>Stats:</strong> {char_count} characters, {line_count} lines</p>
    <p style='color:#000000 !important;'><strong>Class name:</strong> <code style='background:#f5f5f5;padding:2px 6px;border-radius:3px;'>{class_name}</code></p>
    <hr style='border-color:#81c784;'>
    <p style='color:#000000 !important;'><strong>Next steps:</strong></p>
    <ol>
        <li style='color:#000000 !important;'>Save the code as <code style='background:#f5f5f5;padding:2px 6px;'>environment.py</code></li>
        <li style='color:#000000 !important;'>Install dependencies: <code style='background:#f5f5f5;padding:2px 6px;'>pip install gymnasium numpy</code></li>
        <li style='color:#000000 !important;'>Test: <code style='background:#f5f5f5;padding:2px 6px;'>python -c "import environment; env = environment.{class_name}(); env.reset()"</code></li>
    </ol>
</div>
"""
    return f"""
<div style='padding:20px;background-color:#ffe6e6;border-radius:8px;'>
    <h3 style='color:#d32f2f !important;'>⚠️ Code Generated with Syntax Issues</h3>
    <p style='color:#000000 !important;'>{syntax_msg}</p>
    <p style='color:#000000 !important;'>The code may need manual fixes. Please review and correct errors.</p>
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

    header_bg = "#e8f5e9" if critical == 0 else "#ffe6e6"
    header_icon = "✅" if critical == 0 else "⚠️"

    html = f"<div style='padding:20px;background:{header_bg};border-radius:8px;'>"
    html += f"<h3 style='color:#000000 !important;'>{header_icon} Code Validation Report</h3>"
    html += _p(result.get("summary", ""))
    html += (
        f"<p style='color:#000000 !important;'>"
        f"<strong style='color:#d32f2f !important;'>Critical: {critical}</strong> &nbsp;|&nbsp; "
        f"<strong style='color:#f57c00 !important;'>Warnings: {warning}</strong> &nbsp;|&nbsp; "
        f"<strong style='color:#1976d2 !important;'>Info: {info}</strong>"
        f"</p></div>"
    )

    if not mismatches:
        html += "<div style='padding:15px;background:#e8f5e9;border-radius:8px;margin-top:10px;'>"
        html += "<p style='color:#2e7d32 !important;'>✅ No mismatches found! Code matches specification.</p>"
        html += "</div>"
        return html

    html += "<div style='margin-top:15px;'><h4 style='color:#000000 !important;'>Mismatches Found:</h4>"

    severity_styles = {
        "critical": ("#ffe6e6", "#d32f2f", "🔴"),
        "warning":  ("#fff3cd", "#f57c00", "🟡"),
        "info":     ("#e3f2fd", "#1976d2", "🔵"),
    }

    for m in mismatches:
        sev = m.get("severity", "info")
        bg, col, icon = severity_styles.get(sev, ("#f5f5f5", "#000000", "⚪"))
        mid = m.get("id", "unknown")
        html += f"<div style='margin:10px 0;padding:15px;background:{bg};border-radius:6px;border-left:4px solid {col};'>"
        html += (
            f"<p style='color:#000000 !important;margin:0 0 6px 0;'>"
            f"<strong style='color:{col} !important;'>{icon} ID: "
            f"<code style='background:#fff;padding:2px 6px;border-radius:3px;'>{mid}</code></strong> "
            f"<strong>[{m.get('category','').upper()}] {m.get('description','')}</strong></p>"
        )
        html += _p(f"📋 <strong>Spec says:</strong> {m.get('spec_value','N/A')}")
        html += _p(f"💻 <strong>Code has:</strong> {m.get('code_value','N/A')}")
        html += f"<p style='color:#555555 !important;margin:3px 0;'>📍 <em>Location: {m.get('location','unknown')}</em></p>"
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
    header_bg = "#e8f5e9" if final_success else "#ffe6e6"
    header_icon = "✅" if final_success else "❌"
    header_msg = (
        f"Code passed full SB3 compatibility test ({test_steps} training steps)!"
        if final_success
        else f"Code still has runtime errors after {len(rounds)} round(s)"
    )

    html = f"<div style='padding:20px;background:{header_bg};border-radius:8px;'>"
    html += f"<h3 style='color:#000000 !important;'>{header_icon} Runtime Test Report</h3>"
    html += _p(header_msg)
    html += _p(f"<strong>Tests:</strong> Instantiation → Reset → Manual steps → SB3 env_checker → SB3 model creation → SB3 training ({test_steps} steps)")
    html += _p(f"<strong>Rounds:</strong> {len(rounds)}")
    html += "</div>"

    for r in rounds:
        round_num = r["round"]
        success = r.get("success", False)
        if success:
            html += f"<div style='margin:10px 0;padding:12px;background:#e8f5e9;border-radius:6px;'>"
            html += f"<p style='color:#2e7d32 !important;'><strong>Round {round_num}:</strong> ✅ Passed all tests including SB3 training</p>"
            html += "</div>"
        else:
            error_snippet = r.get("error", "")[:300]
            diagnosis = r.get("diagnosis", "")
            html += f"<div style='margin:10px 0;padding:12px;background:#ffe6e6;border-radius:6px;'>"
            html += f"<p style='color:#d32f2f !important;'><strong>Round {round_num}:</strong> ❌ Runtime error</p>"
            html += _p("<strong>Error:</strong>")
            html += f"<pre style='background:#fff;padding:8px;border-radius:4px;color:#d32f2f !important;font-size:0.85em;overflow-x:auto;white-space:pre-wrap;'>{error_snippet}...</pre>"
            html += _p(f"<strong>Diagnosis:</strong> {diagnosis}")
            html += "</div>"

    footer_bg = "#e3f2fd" if final_success else "#fff3cd"
    footer_col = "#1976d2" if final_success else "#f57c00"
    footer_msg = (
        "✨ Code is SB3-compatible and ready for training! Proceed to Step 6."
        if final_success
        else "⚠️ Code still has issues. You can try again or proceed to the playground where you can re-test."
    )
    html += f"<div style='margin-top:12px;padding:12px;background:{footer_bg};border-radius:6px;'>"
    html += f"<p style='color:{footer_col} !important;'><strong>{footer_msg}</strong></p>"
    html += "</div>"

    return html


# ---------------------------------------------------------------------------
# Step 6 – Training status
# ---------------------------------------------------------------------------

def format_training_status(state: str, message: str) -> str:
    """Render a colour-coded training status card."""
    configs = {
        "running":  ("#e3f2fd", "#1976d2", "⏳"),
        "complete": ("#e8f5e9", "#2e7d32", "✅"),
        "error":    ("#ffe6e6", "#d32f2f", "❌"),
        "stopped":  ("#fff3cd", "#f57c00", "⏹️"),
        "idle":     ("#f5f5f5", "#666666", "💤"),
    }
    bg, col, icon = configs.get(state, configs["idle"])
    return (
        f"<div style='padding:15px;background:{bg};border-radius:8px;'>"
        f"<p style='color:{col} !important;margin:0;'>"
        f"<strong style='color:{col} !important;'>{icon} {message}</strong>"
        f"</p></div>"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _p(text: str, color: str = "#000000", margin: int = None) -> str:
    margin_style = "" if margin is None else f"margin:{margin}px;"
    return f"<p style='color:{color} !important;{margin_style}'>{text}</p>"


def _ul(items: list) -> str:
    li = "".join(f"<li style='color:#000000 !important;'>{item}</li>" for item in items)
    return f"<ul>{li}</ul>"


def _error_box(message: str) -> str:
    return (
        f"<div style='padding:20px;background-color:#ffe6e6;border-radius:8px;'>"
        f"<p style='color:#d32f2f !important;'><strong>⚠️ Error:</strong> {message}</p>"
        f"</div>"
    )
