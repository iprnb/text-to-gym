"""
Gradio UI – assembles all steps into a single tabbed interface.

Builds the full multi-tab Gradio app covering:
  Step 1   – Domain description input (Part A free-form + Part B structured)
  Step 2   – Review LLM-generated clarifying questions
  Step 3   – Answer clarifying questions
  Step 4   – Validate specification (up to 2 rounds of follow-up)
  Step 5   – Generate Gymnasium environment code
  Step 5.5 – Validate generated code against the spec
  Step 7   – Runtime test with SB3 and auto-debug
  Step 6   – Training playground (PPO / SAC with live reward curve)

Import ``demo`` from this module and call ``demo.launch()`` to run the app.
"""

import json
import traceback

import gradio as gr

from config import MODEL_PROVIDER, OLLAMA_API_KEY, OLLAMA_MODEL, OPENAI_API_KEY, OPENAI_MODEL
from spec_pipeline import submit_description, process_answers, validate_specification
from code_pipeline import generate_environment_code, validate_code_against_spec, apply_fixes, run_runtime_testing
from training import run_training, stop_training
from formatting import (
    format_answers_summary,
    format_round_limit,
    format_code_validation_report,
    format_training_status,
)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="RL Environment Design Pipeline", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# 🤖 RL Environment Design Pipeline")
        gr.Markdown("### Transform your domain description into a working RL environment")

        with gr.Row():
            provider_dropdown = gr.Dropdown(
                choices=["ollama", "openai"],
                value=MODEL_PROVIDER,
                label="🔧 Model Provider",
                info="Select which LLM to use.",
            )
            with gr.Column():
                gr.Markdown(
                    f"**Current Settings:**\n"
                    f"- **Ollama:** `{OLLAMA_MODEL}` {'✅' if OLLAMA_API_KEY else '❌ (API key not set)'}\n"
                    f"- **OpenAI:** `{OPENAI_MODEL}` {'✅' if OPENAI_API_KEY else '❌ (API key not set)'}"
                )

        # ----- Shared state -----
        questions_json_state    = gr.State("")
        part_a_desc_state       = gr.State("")
        part_a_example_state    = gr.State("")
        specification_state     = gr.State("")
        validation_status_state = gr.State("")
        understanding_state     = gr.State({})
        validation_round_state  = gr.State(0)
        code_state              = gr.State("")   # source-of-truth for generated code
        # Flat list of {"question": ..., "answer": ...} dicts across ALL rounds.
        # Passed to the LLM at each validation step so it never repeats a question.
        qa_history_state        = gr.State([])
        # Texts of the follow-up questions shown to the user in the last validation
        # round, so we can record the exact wording in the history.
        followup_questions_state = gr.State([])
        with gr.Tabs():

            # ================================================================
            # Step 1 – Describe domain
            # ================================================================
            with gr.Tab("📝 Step 1: Describe Your Domain"):
                gr.Markdown("## Part A: Overall System Description (Required)")
                gr.Markdown("Describe your problem in your own words.")

                part_a_desc = gr.Textbox(
                    label="Your Domain Description",
                    placeholder=(
                        "Example: We have a data center cooling system. "
                        "Servers generate heat, and we need to keep temperature stable "
                        "while minimising energy costs..."
                    ),
                    lines=8,
                )

                gr.Markdown("### Optional: Example Scenario")
                part_a_example = gr.Textbox(
                    label="Example Scenario (Optional)",
                    placeholder="Describe a short sequence: what situation starts, what decisions get made, what changes, what's the outcome...",
                    lines=4,
                )

                gr.Markdown("---")
                gr.Markdown("## Part B: Detailed Questions (Optional but Recommended)")

                with gr.Accordion("Show Example (Warehouse Robot)", open=False):
                    gr.Markdown("""
**Decision-maker:** A warehouse robot
**Choices:** Move to shelf A, B, or C; Pick item; Return to station
**Information available:** Current location, items needed, shelf distances, battery level
**How things change:** Battery drains with movement, items disappear when picked, new orders arrive randomly
**Success measure:** Fulfil orders quickly while conserving battery
**Time structure:** 8-hour work shift, then reset
**Starting conditions:** Robot at charging station, random initial orders
""")

                gr.Markdown("### 1. Decision-Maker")
                b1_decision_maker = gr.Textbox(label="Who or what is making decisions?", placeholder="e.g., a robot, algorithm, control system")
                b1_alternative    = gr.Textbox(label="Could there be a better decision-maker?", placeholder="Sometimes the obvious choice isn't the most effective one")

                gr.Markdown("### 2. Available Choices (Actions)")
                b2_actions      = gr.Textbox(label="What options are available at each decision point?", lines=3)
                b2_restrictions = gr.Textbox(label="Are there restrictions on when certain actions can be taken?", placeholder="e.g., time-based, state-dependent, resource-limited")

                gr.Markdown("### 3. Available Information (Observations)")
                b3_observations = gr.Textbox(label="What can be measured or observed when making decisions?", lines=3)
                b3_hidden       = gr.Textbox(label="Is any information hidden, uncertain, or noisy?", lines=2)

                gr.Markdown("### 4. How Things Change (Dynamics)")
                b4_response    = gr.Textbox(label="How does the situation respond to different actions?", lines=3)
                b4_independent = gr.Textbox(label="What changes on its own, independent of decisions?", placeholder="e.g., time passing, external events, natural drift", lines=2)
                b4_variability = gr.Textbox(label="Sources of variability or gradual changes over time?", placeholder="Random events? Wear-and-tear? Seasonal patterns?", lines=3)

                gr.Markdown("### 5. What Matters (Success Criteria)")
                b5_desirable   = gr.Textbox(label="What outcomes are desirable vs. undesirable?", lines=2)
                b5_measurement = gr.Textbox(label="How do you currently measure success?", lines=2)
                b5_safety      = gr.Textbox(label="Critical safety constraints or failure conditions?", placeholder="Things that absolutely must be avoided", lines=2)

                gr.Markdown("### 6. Time Structure")
                b6_duration  = gr.Textbox(label="How long does a typical decision sequence last?")
                b6_reset     = gr.Textbox(label="Does the scenario naturally reset, or is it continuous/ongoing?")
                b6_frequency = gr.Textbox(label="How often are decisions made?", placeholder="e.g., every second, once per hour, whenever an event occurs")

                gr.Markdown("### 7. Starting Conditions")
                b7_starting    = gr.Textbox(label="How does each scenario begin?", lines=2)
                b7_variability = gr.Textbox(label="Is the starting situation always the same, or does it vary?", placeholder='"Always start empty" vs. "Start with random initial state"')

                gr.Markdown("### 8. Scope Check")
                b8_scope = gr.Textbox(label="Is this problem too broad? Should we focus on a sub-problem first?", lines=2)
                b8_fixed = gr.Textbox(label="Parts of the system to treat as 'given' vs. controlled?", lines=2)

                gr.Markdown("### 9. Additional Context")
                b9_additional = gr.Textbox(label="Anything else important we should know?", placeholder="Constraints, rare events, domain-specific details, etc.", lines=3)

                submit_btn = gr.Button("🚀 Submit & Generate Questions", variant="primary", size="lg")

            # ================================================================
            # Step 2 – Review generated questions
            # ================================================================
            with gr.Tab("❓ Step 2: Clarifying Questions"):
                gr.Markdown("## Generated Clarifying Questions")
                gr.Markdown("Based on your description, here are the questions we need answered to build your environment.")
                questions_display = gr.HTML()
                with gr.Accordion("🔍 Debug: Raw JSON Response", open=False):
                    raw_json_output = gr.Textbox(label="Raw LLM Output", lines=20)

            # ================================================================
            # Step 3 – Answer questions
            # ================================================================
            with gr.Tab("📋 Step 3: Answer Questions"):
                gr.Markdown("## Answer the Clarifying Questions")
                gr.Markdown("Match the question numbers from Step 2 (Q1, Q2, etc.). Leave blank if not applicable.")

                answer_fields = []
                for i in range(1, 11):
                    with gr.Accordion(f"Question {i}", open=(i <= 3)):
                        ans = gr.Textbox(
                            label=f"Answer to Q{i}",
                            placeholder="Your answer here (leave empty to skip)",
                            lines=2,
                        )
                        answer_fields.append(ans)

                submit_answers_btn = gr.Button("✅ Submit Answers", variant="primary", size="lg")

            # ================================================================
            # Step 4 – Review & validate spec
            # ================================================================
            with gr.Tab("📊 Step 4: Review & Validate"):
                gr.Markdown("## Review Your Answers & Validate Specification")
                gr.Markdown("Submit your answers first, then click Validate. You have **2 rounds** of clarification.")

                answers_summary = gr.HTML()
                with gr.Accordion("🔍 Debug: Answers JSON", open=False):
                    answers_json_output = gr.Textbox(label="Answers JSON", lines=10)

                gr.Markdown("---")
                validate_btn = gr.Button("🔍 Validate Specification", variant="secondary", size="lg")
                validation_display = gr.HTML()

                with gr.Accordion("🔍 Debug: Specification JSON", open=False):
                    specification_json_output = gr.Textbox(label="Specification JSON", lines=20)
                with gr.Accordion("🧠 Current Environment Understanding", open=False):
                    understanding_display = gr.JSON(label="Accumulated Understanding")

                gr.Markdown("---")
                gr.Markdown("### 📝 Follow-up Questions (If Needed)")
                gr.Markdown("If the validation above shows follow-up questions, answer them below and re-validate:")

                followup_fields = []
                for i in range(1, 6):
                    fq = gr.Textbox(
                        label=f"Follow-up Answer {i}",
                        placeholder=f"Answer to follow-up question {i} shown above (leave empty if no question {i})",
                        lines=3,
                    )
                    followup_fields.append(fq)

                submit_followup_btn = gr.Button("✅ Submit Follow-up Answers & Re-validate", variant="primary", size="lg")

            # ================================================================
            # Step 5 – Generate code
            # ================================================================
            with gr.Tab("🎯 Step 5: Generate Environment"):
                gr.Markdown("## Generate Gymnasium Environment Code")
                gr.Markdown("Once your specification is validated in Step 4, click below to generate the environment code.")
                generate_env_btn = gr.Button("🚀 Generate Environment Code", variant="primary", size="lg")
                code_status = gr.HTML()
                code_output = gr.Code(label="Generated Environment Code", language="python", lines=30)

            # ================================================================
            # Step 5.5 – Validate code against spec
            # ================================================================
            with gr.Tab("🔬 Step 5.5: Validate Code"):
                gr.Markdown("## Validate Code Against Specification")
                gr.Markdown("Check that the generated code actually matches what you described. Catches hardcoded values, wrong ranges, missing reward components, etc.")

                validate_code_btn = gr.Button("🔍 Check Code Against Spec", variant="secondary", size="lg")
                validation_report_display = gr.HTML()

                gr.Markdown("### 🛠️ Apply Fixes")
                gr.Markdown("Enter the mismatch IDs you want to fix (comma-separated, e.g. `unique_id_1, unique_id_2`). IDs are shown in the report above.")
                fix_ids_input = gr.Textbox(label="Mismatch IDs to Fix", placeholder="e.g. unique_id_1, unique_id_3", lines=1)
                apply_fixes_btn = gr.Button("🔧 Apply Selected Fixes & Update Code", variant="primary")
                fix_status = gr.HTML()

                with gr.Accordion("🔍 Debug: Raw Validation JSON", open=False):
                    validation_raw_json = gr.Textbox(label="Raw JSON", lines=15)

                mismatches_state = gr.State([])

            # ================================================================
            # Step 5.7 – Runtime test
            # ================================================================
            with gr.Tab("🧪 Step 5.7: Runtime Test"):
                gr.Markdown("## Runtime Testing & Auto-Debug")
                gr.Markdown(
                    "**Comprehensive SB3 compatibility test:** Runs the environment through "
                    "instantiation → reset → manual steps → SB3's `check_env()` → model creation → actual training. "
                    "If errors occur, the LLM debugs and fixes them automatically (up to 5 rounds)."
                )

                with gr.Row():
                    test_steps_slider = gr.Slider(minimum=100, maximum=5000, value=1000, step=100,
                                                  label="Test Steps", info="How many environment steps to test")
                    test_max_rounds_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1,
                                                       label="Max Debug Rounds", info="Maximum fix attempts")

                test_code_btn = gr.Button("🧪 Test & Auto-Fix Code", variant="primary", size="lg")
                test_report_display = gr.HTML()
                with gr.Accordion("📋 Test Log", open=False):
                    test_log_output = gr.Textbox(label="Round-by-round test log", lines=15, interactive=False)

            # ================================================================
            # Step 6 – Playground
            # ================================================================
            with gr.Tab("🎮 Step 6: Playground"):
                gr.Markdown("## Train & Visualise")
                gr.Markdown("Train your environment with PPO or SAC and watch the reward curve live. Make sure Step 5.7 passes before training.")

                with gr.Row():
                    algo_dropdown = gr.Dropdown(choices=["PPO", "SAC"], value="PPO", label="Algorithm",
                                                info="PPO works for most environments. SAC requires continuous action spaces.")
                    timesteps_slider = gr.Slider(minimum=10_000, maximum=500_000, value=50_000, step=10_000,
                                                 label="Total Timesteps", info="More timesteps = longer training = better results")

                with gr.Row():
                    start_training_btn = gr.Button("▶️ Start Training", variant="primary", size="lg")
                    stop_training_btn  = gr.Button("⏹️ Stop Training", variant="stop", size="lg")
                    retest_btn         = gr.Button("🔄 Re-test & Fix Code", variant="secondary", size="lg")

                training_status = gr.HTML(value=format_training_status("idle", "Ready. Click Start Training to begin."))
                reward_plot = gr.Plot(label="Reward Curve")

                with gr.Accordion("📋 Training Logs", open=False):
                    training_logs = gr.Textbox(label="Last 50 log lines", lines=15, interactive=False)
                with gr.Accordion("🔧 Re-test Results", open=False):
                    retest_report = gr.HTML()
                    retest_log    = gr.Textbox(label="Re-test log", lines=10, interactive=False)

        # --------------------------------------------------------------------
        # About section
        # --------------------------------------------------------------------
        gr.Markdown("""
---
### 📚 About This Tool
This pipeline helps domain experts create working RL environments from natural language descriptions.
No RL expertise required – just describe your problem and the tool handles the technical details.

### 🎯 Complete Workflow
1. **Step 1:** Describe your domain in natural language
2. **Step 2:** Review generated clarifying questions
3. **Step 3:** Answer the questions
4. **Step 4:** Validate specification (max 2 rounds of follow-ups)
5. **Step 5:** Generate environment code
6. **Step 5.5:** Validate code against spec, apply fixes if needed
7. **Step 5.7:** Runtime test (runs 1 000 steps, auto-debugs errors)
8. **Step 6:** Train with PPO/SAC, watch live curve

### 🔧 Setup
```bash
export OLLAMA_API_KEY="your-key"      # For Ollama Cloud
export OPENAI_API_KEY="sk-..."        # For OpenAI
export OLLAMA_MODEL="qwen3-next:80b"  # Change model
```
""")

        # ====================================================================
        # Event handlers
        # ====================================================================

        # --- Step 1 submit ---
        def on_submit(provider, desc, example, *part_b_args):
            """Step 1 handler: build the clarifying-questions prompt and call the LLM.

            Passes Part A and Part B fields to the spec pipeline, stores the returned
            questions JSON in state, and navigates the user toward Step 3.
            """
            formatted, raw_json = submit_description(provider, desc, example, *part_b_args)
            if not raw_json.startswith("Error") and not raw_json.startswith("==="):
                formatted += (
                    "<div style='padding:15px;background:#e3f2fd;border-radius:8px;margin-top:20px;'>"
                    "<p style='color:#000000 !important;'>✅ <strong>Questions generated!</strong> "
                    "Please go to <strong>Step 3</strong> to provide your answers.</p></div>"
                )
            return formatted, raw_json, raw_json, desc, example

        submit_btn.click(
            fn=on_submit,
            inputs=[
                provider_dropdown,
                part_a_desc, part_a_example,
                b1_decision_maker, b1_alternative,
                b2_actions, b2_restrictions,
                b3_observations, b3_hidden,
                b4_response, b4_independent, b4_variability,
                b5_desirable, b5_measurement, b5_safety,
                b6_duration, b6_reset, b6_frequency,
                b7_starting, b7_variability,
                b8_scope, b8_fixed, b9_additional,
            ],
            outputs=[questions_display, raw_json_output, questions_json_state, part_a_desc_state, part_a_example_state],
        )

        # --- Step 3 collect answers ---
        # Wrap process_answers so we can also build the initial qa_history from Q&A pairs.
        def on_process_answers(questions_json, *answers):
            """Step 3 handler: map raw answer strings to their question texts.

            Builds the answers dict and also constructs the initial qa_history list
            (flat list of question/answer pairs) that is forwarded to the LLM at each
            validation round to prevent repeated questions.
            """
            summary, answers_json_str = process_answers(questions_json, *answers)

            # Build flat history list from the questions that were answered
            try:
                questions_data = json.loads(questions_json) if questions_json else {}
                questions = questions_data.get("questions", [])
                history = []
                for question, answer in zip(questions, answers):
                    if answer and answer.strip():
                        history.append({
                            "question": question.get("question_text", ""),
                            "answer": answer.strip(),
                        })
            except Exception:
                history = []

            return summary, answers_json_str, history

        submit_answers_btn.click(
            fn=on_process_answers,
            inputs=[questions_json_state] + answer_fields,
            outputs=[answers_summary, answers_json_output, qa_history_state],
        )

        # --- Step 4 validate ---
        def on_validate(desc, example, provider, questions_json, answers_json, understanding, round_num, qa_history):
            """Step 4 handler: run one round of specification validation.

            Updates the accumulated environment understanding, asks the LLM whether
            the spec is complete, and either advances to 'complete' status or returns
            follow-up questions. Enforces a hard cap of 2 validation rounds.
            """
            round_num += 1
            print(f"\n{'='*60}\nVALIDATION ROUND {round_num}/2\n{'='*60}\n")

            if round_num > 2:
                limit_html = format_round_limit()
                spec_json  = json.dumps(understanding, indent=2)
                return limit_html, spec_json, spec_json, understanding, understanding, round_num, "complete", qa_history, []

            html, val_json, val_data, updated = validate_specification(
                desc, example, questions_json, answers_json, provider,
                current_understanding=understanding, round_number=round_num,
                qa_history=qa_history,
            )

            # Extract the follow-up question texts so the next round can record them precisely
            followup_texts = []
            try:
                parsed = json.loads(val_json) if val_json else {}
                followup_texts = [fq.get("question", "") for fq in parsed.get("follow_up_questions", [])]
            except Exception:
                pass

            if val_data and val_data.get("status") == "complete":
                spec_json = json.dumps(val_data.get("complete_specification", {}), indent=2)
                return html, val_json, spec_json, updated, updated, round_num, "complete", qa_history, followup_texts
            return html, val_json, "", updated, updated, round_num, "needs_clarification", qa_history, followup_texts

        validate_btn.click(
            fn=on_validate,
            inputs=[part_a_desc_state, part_a_example_state, provider_dropdown,
                    questions_json_state, answers_json_output,
                    understanding_state, validation_round_state, qa_history_state],
            outputs=[validation_display, specification_json_output, specification_state,
                     understanding_state, understanding_display,
                     validation_round_state, validation_status_state, qa_history_state, followup_questions_state],
        )

        # --- Step 4 follow-up submit ---
        def on_submit_followup(desc, example, provider, questions_json,
                               original_answers_json, understanding, round_num,
                               qa_history, followup_questions,
                               *followup_answers):
            """Step 4 follow-up handler: merge follow-up answers and re-validate.

            Appends follow-up Q&A pairs to qa_history (using the stored question texts
            so the LLM sees exact wording), increments the round counter, and calls
            validate_specification again. If the round limit is reached, proceeds with
            the current understanding using reasonable defaults.
            """
            try:
                original_answers = json.loads(original_answers_json) if original_answers_json else {}
                followup_count = 0
                new_history_pairs = []

                for i, answer in enumerate(followup_answers, 1):
                    if answer and answer.strip():
                        followup_count += 1
                        key = f"FQ{i}_round{round_num}"
                        # Use the actual question text if we stored it, else fall back to a label
                        q_text = (
                            followup_questions[i - 1]
                            if followup_questions and i - 1 < len(followup_questions)
                            else f"Follow-up question {i} (round {round_num})"
                        )
                        original_answers[key] = {
                            "question": q_text,
                            "answer": answer,
                            "round": round_num,
                        }
                        new_history_pairs.append({"question": q_text, "answer": answer.strip()})

                updated_history = list(qa_history or []) + new_history_pairs
                updated_answers_json = json.dumps(original_answers, indent=2)
                round_num += 1
                print(f"\n{'='*60}\nFOLLOW-UP VALIDATION ROUND {round_num}/2\n{'='*60}\n")

                if round_num > 2:
                    limit_html = format_round_limit()
                    spec_json  = json.dumps(understanding, indent=2)
                    summary    = format_answers_summary(original_answers, followup_count)
                    return (summary, updated_answers_json,
                            limit_html, spec_json, spec_json,
                            understanding, understanding, round_num, "complete", updated_history, [])

                html, val_json, val_data, updated = validate_specification(
                    desc, example, questions_json, updated_answers_json, provider,
                    current_understanding=understanding, round_number=round_num,
                    qa_history=updated_history,
                )
                summary = format_answers_summary(original_answers, followup_count)

                # Extract next round's follow-up question texts
                next_followup_texts = []
                try:
                    parsed = json.loads(val_json) if val_json else {}
                    next_followup_texts = [fq.get("question", "") for fq in parsed.get("follow_up_questions", [])]
                except Exception:
                    pass

                if val_data and val_data.get("status") == "complete":
                    spec_json = json.dumps(val_data.get("complete_specification", {}), indent=2)
                    return (summary, updated_answers_json,
                            html, val_json, spec_json,
                            updated, updated, round_num, "complete", updated_history, next_followup_texts)

                return (summary, updated_answers_json,
                        html, val_json, "",
                        updated, updated, round_num, "needs_clarification", updated_history, next_followup_texts)

            except Exception as e:
                error = f"<div style='padding:20px;background-color:#ffe6e6;border-radius:8px;'><p style='color:#d32f2f !important;'><strong>Error:</strong> {e}</p></div>"
                return (error, original_answers_json,
                        error, traceback.format_exc(), "",
                        understanding, understanding, round_num, "error", qa_history, followup_questions)

        submit_followup_btn.click(
            fn=on_submit_followup,
            inputs=[part_a_desc_state, part_a_example_state, provider_dropdown,
                    questions_json_state, answers_json_output,
                    understanding_state, validation_round_state,
                    qa_history_state, followup_questions_state] + followup_fields,
            outputs=[answers_summary, answers_json_output,
                     validation_display, specification_json_output, specification_state,
                     understanding_state, understanding_display,
                     validation_round_state, validation_status_state,
                     qa_history_state, followup_questions_state],
        )

        # --- Step 5 generate code ---
        def on_generate_code(spec_json, provider):
            """Step 5 handler: generate Gymnasium environment code from the validated spec.

            Guards against missing specification and stores the generated code in
            code_state, which is the single source of truth for all downstream steps.
            """
            if not spec_json:
                err = "<div style='padding:20px;background-color:#ffe6e6;border-radius:8px;'><p style='color:#d32f2f !important;'><strong>Error:</strong> Please validate specification first in Step 4!</p></div>"
                return err, "", ""
            status_html, code = generate_environment_code(spec_json, provider)
            return status_html, code, code

        generate_env_btn.click(
            fn=on_generate_code,
            inputs=[specification_state, provider_dropdown],
            outputs=[code_status, code_output, code_state],
        )

        # --- Step 5.7 runtime test ---
        def on_test_code(code, test_steps, max_rounds, provider):
            """Step 5.7 handler: run the environment through a full SB3 compatibility test.

            Executes the code in a subprocess, runs up to max_rounds of LLM-assisted
            auto-debugging if errors are found, and updates code_state with the fixed
            version so Step 6 always trains on the latest code.
            """
            if not code or not code.strip():
                err = "<div style='padding:20px;background:#ffe6e6;border-radius:8px;'><p style='color:#d32f2f !important;'>Please generate code in Step 5 first.</p></div>"
                return err, "", code, code
            final_code, report_html, log_text = run_runtime_testing(code, provider, max_rounds=int(max_rounds), test_steps=int(test_steps))
            return report_html, log_text, final_code, final_code

        test_code_btn.click(
            fn=on_test_code,
            inputs=[code_state, test_steps_slider, test_max_rounds_slider, provider_dropdown],
            outputs=[test_report_display, test_log_output, code_output, code_state],
        )

        # --- Step 5.5 validate code ---
        def on_validate_code(spec_json, code, provider):
            """Step 5.5 handler: ask the LLM to find mismatches between code and spec.

            Returns a formatted mismatch report and stores the raw mismatches list in
            mismatches_state so the apply-fixes handler can reference them by ID.
            """
            if not spec_json:
                err = "<div style='padding:20px;background:#ffe6e6;border-radius:8px;'><p style='color:#d32f2f !important;'>Please complete Step 4 (validation) first.</p></div>"
                return err, "[]", err, []
            if not code or not code.strip():
                err = "<div style='padding:20px;background:#ffe6e6;border-radius:8px;'><p style='color:#d32f2f !important;'>Please generate code in Step 5 first.</p></div>"
                return err, "[]", err, []

            result, raw_json = validate_code_against_spec(spec_json, code, provider)
            if not result:
                err = "<div style='padding:20px;background:#ffe6e6;border-radius:8px;'><p style='color:#d32f2f !important;'>Could not parse validation response. See debug output.</p></div>"
                return err, raw_json, err, []

            report_html = format_code_validation_report(result)
            return report_html, raw_json, "", result.get("mismatches", [])

        validate_code_btn.click(
            fn=on_validate_code,
            inputs=[specification_state, code_state, provider_dropdown],
            outputs=[validation_report_display, validation_raw_json, fix_status, mismatches_state],
        )

        # --- Step 5.5 apply fixes ---
        def on_apply_fixes(spec_json, code, fix_ids_str, mismatches, provider):
            """Step 5.5 fix handler: apply a user-selected subset of mismatches to the code.

            Parses the comma-separated mismatch IDs, asks the LLM to apply only those
            specific fixes, validates the result parses as valid Python, and updates
            both code_output and code_state.
            """
            if not fix_ids_str or not fix_ids_str.strip():
                return (
                    "<div style='padding:15px;background:#fff3cd;border-radius:8px;'><p style='color:#f57c00 !important;'>No IDs entered. Please enter mismatch IDs to fix.</p></div>",
                    code, code,
                )
            selected_ids = [x.strip() for x in fix_ids_str.split(",") if x.strip()]
            fixed_code   = apply_fixes(spec_json, code, selected_ids, mismatches, provider)

            if fixed_code == code:
                status = "<div style='padding:15px;background:#fff3cd;border-radius:8px;'><p style='color:#f57c00 !important;'>⚠️ No changes made. IDs may not match or fixes were identical to existing code.</p></div>"
            else:
                status = f"<div style='padding:15px;background:#e8f5e9;border-radius:8px;'><p style='color:#2e7d32 !important;'>✅ Applied fixes for: {', '.join(selected_ids)}. Code updated in Step 5.</p></div>"

            return status, fixed_code, fixed_code

        apply_fixes_btn.click(
            fn=on_apply_fixes,
            inputs=[specification_state, code_state, fix_ids_input, mismatches_state, provider_dropdown],
            outputs=[fix_status, code_output, code_state],
        )

        # --- Step 6 training ---
        start_training_btn.click(
            fn=run_training,
            inputs=[code_state, algo_dropdown, timesteps_slider, provider_dropdown],
            outputs=[training_status, reward_plot, training_logs],
        )

        stop_training_btn.click(
            fn=stop_training,
            inputs=[],
            outputs=[training_status],
        )

        def on_retest_playground(code, provider):
            """Step 6 re-test handler: re-run runtime testing from the training playground.

            Convenience wrapper around run_runtime_testing so users can fix and re-test
            without switching tabs. Updates code_state with any auto-fixed version.
            """
            if not code or not code.strip():
                err = "<div style='padding:20px;background:#ffe6e6;border-radius:8px;'><p style='color:#d32f2f !important;'>No code to test.</p></div>"
                return err, "", code, code
            final_code, report_html, log_text = run_runtime_testing(code, provider, max_rounds=5, test_steps=1000)
            return report_html, log_text, final_code, final_code

        retest_btn.click(
            fn=on_retest_playground,
            inputs=[code_state, provider_dropdown],
            outputs=[retest_report, retest_log, code_output, code_state],
        )

    return demo


demo = build_demo()