"""
Specification pipeline: domain description → clarifying questions → spec.

Covers Steps 1–4 of the UI workflow:
  1. Build and send the clarifying-questions prompt
  2. Process user answers
  3. Maintain an accumulated "environment understanding"
  4. Validate the spec (up to 2 rounds of follow-up questions)
"""

import json
import traceback

from llm import call_llm, parse_llm_response
from prompts import (
    CLARIFYING_QUESTIONS_SYSTEM,
    CLARIFYING_QUESTIONS_USER_TEMPLATE,
    UPDATE_UNDERSTANDING_SYSTEM,
    VALIDATION_SYSTEM,
)
from formatting import (
    format_questions,
    format_answers_summary,
    format_specification,
    format_validation_issues,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value: str) -> str:
    """Return the stripped value, or '[Not provided]' if empty."""
    return (value or "").strip() or "[Not provided]"


# ---------------------------------------------------------------------------
# Step 1 – Submit description and generate clarifying questions
# ---------------------------------------------------------------------------

def submit_description(
    provider, part_a_desc, part_a_example,
    b1_decision_maker, b1_alternative,
    b2_actions, b2_restrictions,
    b3_observations, b3_hidden,
    b4_response, b4_independent, b4_variability,
    b5_desirable, b5_measurement, b5_safety,
    b6_duration, b6_reset, b6_frequency,
    b7_starting, b7_variability,
    b8_scope, b8_fixed, b9_additional,
):
    """
    Build the clarifying-questions prompt, call the LLM, and return
    ``(formatted_html, raw_json_string)``.
    """
    try:
        if not part_a_desc or not part_a_desc.strip():
            err = "<div style='padding:20px;background-color:#ffe6e6;border-radius:8px;'><p style='color:#d32f2f !important;'><strong>⚠️ Error:</strong> Please provide Part A: Overall Description (required)</p></div>"
            return err, "Error: Part A is required"

        example_text = (
            f"\n## Optional Example\n{part_a_example}"
            if part_a_example and part_a_example.strip()
            else ""
        )

        user_prompt = CLARIFYING_QUESTIONS_USER_TEMPLATE.format(
            part_a_description=part_a_desc,
            part_a_example=example_text,
            b1_decision_maker=_fmt(b1_decision_maker),
            b1_alternative=_fmt(b1_alternative),
            b2_actions=_fmt(b2_actions),
            b2_restrictions=_fmt(b2_restrictions),
            b3_observations=_fmt(b3_observations),
            b3_hidden=_fmt(b3_hidden),
            b4_response=_fmt(b4_response),
            b4_independent=_fmt(b4_independent),
            b4_variability=_fmt(b4_variability),
            b5_desirable=_fmt(b5_desirable),
            b5_measurement=_fmt(b5_measurement),
            b5_safety=_fmt(b5_safety),
            b6_duration=_fmt(b6_duration),
            b6_reset=_fmt(b6_reset),
            b6_frequency=_fmt(b6_frequency),
            b7_starting=_fmt(b7_starting),
            b7_variability=_fmt(b7_variability),
            b8_scope=_fmt(b8_scope),
            b8_fixed=_fmt(b8_fixed),
            b9_additional=_fmt(b9_additional),
        )

        print(f"\n{'='*60}\nBUILDING PROMPT | Provider: {provider} | Length: {len(user_prompt)}\n{'='*60}\n")
        llm_response = call_llm(CLARIFYING_QUESTIONS_SYSTEM, user_prompt, provider=provider)

        if llm_response.startswith("Error:"):
            err = f"<div style='padding:20px;background-color:#ffe6e6;border-radius:8px;'><p style='color:#d32f2f !important;'><strong>⚠️ {llm_response}</strong></p></div>"
            return err, llm_response

        questions_data = parse_llm_response(llm_response)
        formatted_output = format_questions(questions_data)
        raw_json = (
            json.dumps(questions_data, indent=2)
            if questions_data
            else f"=== RAW RESPONSE ===\n{llm_response}\n\n=== PARSING FAILED ==="
        )
        return formatted_output, raw_json

    except Exception as e:
        error_trace = traceback.format_exc()
        err = f"<div style='padding:20px;background-color:#ffe6e6;border-radius:8px;'><p style='color:#d32f2f !important;'><strong>⚠️ Unexpected Error:</strong> {e}</p></div>"
        print(error_trace)
        return err, error_trace


# ---------------------------------------------------------------------------
# Step 3 – Process user answers
# ---------------------------------------------------------------------------

def process_answers(questions_json: str, *answers):
    """
    Map user answers to question texts and return
    ``(summary_html, answers_json_string)``.
    """
    try:
        questions_data = json.loads(questions_json) if questions_json else {}
        questions = questions_data.get("questions", [])

        answers_dict = {}
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if answer and answer.strip():
                answers_dict[f"Q{i + 1}"] = {
                    "question": question.get("question_text"),
                    "answer": answer,
                }

        summary = "<div style='padding:20px;background-color:#e8f5e9;border-radius:8px;'>"
        summary += "<h3 style='color:#2e7d32 !important;'>✅ Answers Collected</h3>"
        summary += f"<p style='color:#000000 !important;'>You answered {len(answers_dict)} out of {len(questions)} questions.</p>"
        summary += "<hr style='border-color:#81c784;'>"
        for key, value in answers_dict.items():
            summary += f"<p style='color:#000000 !important;'><strong>{key}:</strong> {value['question']}</p>"
            summary += f"<p style='margin-left:20px;color:#424242 !important;'><em>Answer: {value['answer']}</em></p>"
        summary += "</div>"

        return summary, json.dumps(answers_dict, indent=2)

    except Exception as e:
        err = f"<div style='padding:20px;background-color:#ffe6e6;border-radius:8px;'><p style='color:#d32f2f !important;'><strong>Error:</strong> {e}</p></div>"
        return err, str(e)


# ---------------------------------------------------------------------------
# Step 4 – Environment understanding management
# ---------------------------------------------------------------------------

def update_environment_understanding(
    current_understanding: dict,
    new_answers: dict,
    original_description: str,
    provider: str,
    round_number: int,
) -> dict:
    """
    Ask the LLM to merge *new_answers* into *current_understanding*.
    Never erases existing confirmed fields – only adds or modifies.
    Returns the updated understanding dict (falls back to current on failure).
    """
    try:
        update_prompt = (
            f"## Current Understanding (Round {round_number})\n"
            + (
                json.dumps(current_understanding, indent=2)
                if current_understanding
                else "No understanding yet - this is the first round."
            )
            + f"\n\n## Original Domain Description\n{original_description}"
            + f"\n\n## New Answers to Incorporate\n{json.dumps(new_answers, indent=2)}"
            + f"\n\n---\n\nUpdate the understanding by incorporating the new answers.\n"
            f"Remember:\n"
            f"- Only ADD or MODIFY, never ERASE existing confirmed information\n"
            f"- Mark fields as confirmed when the user has explicitly answered them\n"
            f"- Update the change_log with what changed\n"
            f"- Set round_number to {round_number}\n"
            f"- Update still_unclear to reflect what remains unknown after these answers"
        )

        print(f"\n{'='*60}\nUPDATING ENVIRONMENT UNDERSTANDING (Round {round_number})...\n{'='*60}")
        response = call_llm(UPDATE_UNDERSTANDING_SYSTEM, update_prompt, provider=provider)
        updated = parse_llm_response(response)

        if not updated:
            print("❌ Failed to parse understanding update – keeping previous")
            return current_understanding

        print(f"✅ Understanding updated! Changes: {updated.get('change_log', [])}")
        return updated

    except Exception as e:
        print(f"Error updating understanding: {e}")
        return current_understanding


def _validate_with_understanding(
    understanding: dict,
    original_description: str,
    all_answers: dict,
    provider: str,
    qa_history: list = None,
) -> tuple:
    """
    Ask the LLM whether we have enough information to build the environment.

    ``qa_history`` is a flat list of ``{"question": ..., "answer": ...}`` dicts
    covering every question asked and answered across all rounds.  It is injected
    into the prompt so the LLM can avoid repeating previously asked questions.

    Returns ``(validation_data_dict | None, raw_json_string)``.
    """
    try:
        history_section = ""
        if qa_history:
            history_lines = []
            for i, pair in enumerate(qa_history, 1):
                history_lines.append(
                    f"  Q{i}: {pair.get('question', '(unknown question)')}\n"
                    f"  A{i}: {pair.get('answer', '(no answer)')}"
                )
            history_section = (
                "\n\n## Full Q&A History (ALL rounds – do NOT repeat any of these questions)\n"
                + "\n\n".join(history_lines)
            )

        validation_prompt = (
            f"## Current Environment Understanding\n{json.dumps(understanding, indent=2)}"
            + f"\n\n## Original Domain Description\n{original_description}"
            + history_section
            + f"\n\n## Structured Answer Store\n{json.dumps(all_answers, indent=2)}"
            + "\n\n---\n\nBased on the current understanding, determine if we have enough information "
            "to generate a complete Gymnasium environment.\n\n"
            "IMPORTANT:\n"
            "- Only ask follow-up questions about fields still marked as unclear or unconfirmed\n"
            "- Do NOT ask about anything already confirmed\n"
            "- Do NOT repeat or rephrase any question already present in the Q&A History above\n"
            "- If understanding is mostly complete, set status to 'complete' and fill in reasonable defaults for minor unknowns"
        )

        print(f"\n{'='*60}\nVALIDATING WITH FULL UNDERSTANDING...\n{'='*60}")
        response = call_llm(VALIDATION_SYSTEM, validation_prompt, provider=provider)
        validation_data = parse_llm_response(response)

        if not validation_data:
            return None, response

        print(f"✅ Validation complete! Status: {validation_data.get('status')}")
        return validation_data, json.dumps(validation_data, indent=2)

    except Exception as e:
        print(f"Error in validation: {e}")
        return None, str(e)


def validate_specification(
    part_a_desc: str,
    part_a_example: str,
    questions_json: str,
    answers_json: str,
    provider: str,
    current_understanding: dict = None,
    round_number: int = 1,
    qa_history: list = None,
) -> tuple:
    """
    Full Step 4 validation flow:
      1. Update the accumulated understanding with all answers so far.
      2. Ask the LLM to validate completeness.

    ``qa_history`` is a flat list of ``{"question": ..., "answer": ...}`` dicts
    spanning every round; it is forwarded to the LLM to prevent duplicate questions.

    Returns ``(display_html, raw_json, validation_data | None, updated_understanding)``.
    """
    try:
        answers_data = json.loads(answers_json) if answers_json else {}

        updated_understanding = update_environment_understanding(
            current_understanding or {},
            answers_data,
            part_a_desc,
            provider,
            round_number,
        )

        validation_data, validation_json = _validate_with_understanding(
            updated_understanding,
            part_a_desc,
            answers_data,
            provider,
            qa_history=qa_history or [],
        )

        if not validation_data:
            err = "<div style='padding:20px;background-color:#ffe6e6;border-radius:8px;'><p style='color:#d32f2f !important;'><strong>⚠️ Error:</strong> Could not parse validation response.</p></div>"
            return err, validation_json, None, updated_understanding

        status = validation_data.get("status", "needs_clarification")
        if status == "complete":
            html = format_specification(validation_data)
        else:
            html = format_validation_issues(validation_data)

        return html, validation_json, validation_data if status == "complete" else None, updated_understanding

    except Exception as e:
        err = f"<div style='padding:20px;background-color:#ffe6e6;border-radius:8px;'><p style='color:#d32f2f !important;'><strong>Error:</strong> {e}</p></div>"
        return err, traceback.format_exc(), None, current_understanding or {}