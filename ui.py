"""
Gradio UI – assembles all steps into a single tabbed interface.

Tab layout:
  Step 1 – Domain description (Part A + Part B)
  Step 2 – Clarifying questions WITH inline answer boxes (Q1/A1/Q2/A2… interleaved)
  Step 3 – Validate specification (auto-triggered, single Re-validate button, inline follow-up Q&A)
  Step 4 – Generate environment (auto generate → spec-check → runtime test in one click)
  Step 5 – Training playground
"""

import json
import traceback

import gradio as gr

from config import (
    MODEL_PROVIDER, OLLAMA_API_KEY, OLLAMA_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
    ALGORITHM_NAMES, ALGORITHM_ACTION_SPACE_SUPPORT,
)
from spec_pipeline import submit_description, process_answers, validate_specification
from code_pipeline import generate_full_pipeline, run_runtime_testing
from training import run_training, stop_training
from formatting import (
    format_answers_summary,
    format_round_limit,
    format_training_status,
    _error_box,
    _THEME_CSS,
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

_GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, .gradio-container, input, textarea, select, button, .prose,
label, .label-wrap span {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}
h1, h2, h3, h4, h5 {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    font-weight: 600; letter-spacing: -0.01em;
}
/* Ensure Gradio textbox text is always readable */
textarea, input[type="text"] { color: var(--body-text-color) !important; }
/* Tab labels */
.tab-nav button { font-size: 0.88rem; font-weight: 500; }

/* ── Loading spinner ──────────────────────────────────────────────────── */
@keyframes ttg-spin {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes ttg-pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}
.ttg-spinner {
    display: inline-block;
    width: 22px; height: 22px;
    border: 3px solid rgba(59,130,246,0.25);
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: ttg-spin 0.8s linear infinite;
    vertical-align: middle;
    margin-right: 10px;
}
.dark .ttg-spinner {
    border-color: rgba(147,197,253,0.25);
    border-top-color: #93c5fd;
}
.ttg-loading-row {
    display: flex; align-items: center; gap: 10px;
    padding: 14px 18px; border-radius: 10px;
    background: var(--background-fill-secondary, #f0f4ff);
    border: 1px solid #93c5fd;
    animation: ttg-pulse 1.8s ease-in-out infinite;
    color: #1e3a5f;
    font-weight: 500;
}
.dark .ttg-loading-row {
    background: #1e3a5f;
    border-color: #2d5a8e;
    color: #bfdbfe;
}
"""


def _algo_info(algo: str) -> str:
    support = ALGORITHM_ACTION_SPACE_SUPPORT.get(algo, "both")
    if support == "continuous":
        return f"⚠️ **{algo}** requires a **continuous** (Box) action space. Use PPO or A2C for discrete spaces."
    if support == "discrete":
        return f"⚠️ **{algo}** requires a **discrete** action space."
    return f"✅ **{algo}** works with both discrete and continuous action spaces."


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_LOADING_HTML = (
    _THEME_CSS
    + "<div class='ttg-loading-row'>"
    "<span class='ttg-spinner'></span>"
    "<span>Thinking… this usually takes a few seconds.</span>"
    "</div>"
)

_LOADING_HTML_PIPELINE = (
    _THEME_CSS
    + "<div class='ttg-loading-row'>"
    "<span class='ttg-spinner'></span>"
    "<span>Running pipeline… generating, checking, and testing your environment.</span>"
    "</div>"
)


def _question_card_html(idx: int, q: dict, prefix: str = "Q") -> str:
    """Render a single question card (without CSS — caller adds _THEME_CSS once)."""
    pclass_map = {"high": "ttg-q-high", "medium": "ttg-q-medium", "low": "ttg-q-low"}
    pclass = pclass_map.get(q.get("priority", "medium"), "ttg-q-medium")
    fmt = q.get("format", "free_form")

    html = f"<div class='ttg-q-card {pclass}'>"
    html += (
        f"<p class='ttg-p'>"
        f"<strong>{prefix}{idx}. [{q.get('category','other').upper()}]</strong> "
        f"{q.get('question_text','N/A')}</p>"
    )
    if fmt == "multiple_choice" and q.get("options"):
        html += "<ul style='margin:4px 0 4px 18px;'>"
        for opt in q["options"]:
            html += f"<li class='ttg-p'>{opt}</li>"
        html += "</ul>"
    elif fmt == "yes_no":
        html += "<p class='ttg-p ttg-em'>Answer: Yes / No</p>"
        if q.get("follow_up_question"):
            cond = q.get("follow_up_condition", "yes")
            html += f"<p class='ttg-p ttg-em' style='margin-left:16px;'>If {cond}: {q['follow_up_question']}</p>"
    else:
        html += "<p class='ttg-p ttg-em'>[Free-form answer]</p>"
    html += "</div>"
    return html


def _followup_card_html(idx: int, q_text: str) -> str:
    """Render a single follow-up question card (without CSS)."""
    return (
        f"<div class='ttg-q-card ttg-q-medium'>"
        f"<p class='ttg-p'><strong>FQ{idx}.</strong> {q_text}</p>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Text to Gym",
        theme=gr.themes.Soft(
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        ),
        css=_GLOBAL_CSS,
    ) as demo:

        gr.Markdown("# 🤖 Text to Gym")
        gr.Markdown("### Transform a plain-language description into a working RL environment")

        _available_providers = [p for p in ["ollama", "openai"]
                                if (p == "ollama" and OLLAMA_API_KEY) or (p == "openai" and OPENAI_API_KEY)]
        _default_provider = MODEL_PROVIDER if MODEL_PROVIDER in _available_providers else (_available_providers[0] if _available_providers else "ollama")
        with gr.Row():
            provider_dropdown = gr.Dropdown(
                choices=_available_providers or ["ollama", "openai"],
                value=_default_provider,
                label="🔧 Model Provider",
                info="Only providers with a configured API key are shown.",
                scale=1,
            )
            gr.Column(scale=3)  # spacer

        # ── Shared state ─────────────────────────────────────────────────────
        questions_json_state     = gr.State("")
        part_a_desc_state        = gr.State("")
        part_a_example_state     = gr.State("")
        specification_state      = gr.State("")
        understanding_state      = gr.State({})
        validation_round_state   = gr.State(0)
        qa_history_state         = gr.State([])
        followup_questions_state = gr.State([])
        # Pending HTML values — set by the slow LLM call, applied by the fast follow-up .then()
        # so that gr.HTML components are NOT in the slow call's outputs (prevents loading spinners)
        _pending_intro_state     = gr.State("")
        _pending_display_state   = gr.State("")
        _pending_section_state   = gr.State("")
        # Step 5 state copies — decouples training inputs from UI components
        # to avoid Gradio 6 event cross-firing when components are shared across handlers
        algo_state               = gr.State("PPO")
        timesteps_state          = gr.State(50_000)

        with gr.Tabs() as tabs:

            # ================================================================
            # Step 1 – Describe domain
            # ================================================================
            with gr.Tab("📝 Step 1: Describe", id="tab1"):
                gr.Markdown("## Part A: Overall Description *(required)*")
                part_a_desc = gr.Textbox(
                    label="Your Domain Description",
                    placeholder=(
                        "Example: We have a data center cooling system. "
                        "Servers generate heat, and we need to keep temperature stable "
                        "while minimising energy costs…"
                    ),
                    lines=8,
                )
                gr.Markdown("### Optional: Example Scenario")
                part_a_example = gr.Textbox(
                    label="Example Scenario (Optional)",
                    placeholder="Describe a short sequence: what starts, what decisions are made, what changes, what's the outcome…",
                    lines=4,
                )
                gr.Markdown("---")
                gr.Markdown("## Part B: Detailed Questions *(optional but recommended)*")

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
                step1_loading = gr.HTML(value="", visible=False)

            # ================================================================
            # Step 2 – Questions + Answers (Q1/A1/Q2/A2 interleaved)
            # ================================================================
            with gr.Tab("❓ Step 2: Questions & Answers", id="tab2"):

                step2_intro = gr.HTML(
                    value=(
                        _THEME_CSS
                        + "<div class='ttg-card ttg-card-info'>"
                        "<p class='ttg-p'>Your questions will appear here once you submit your description.</p>"
                        "</div>"
                    )
                )

                # Interleaved: question card HTML + answer textbox, 10 pairs
                q_cards = []   # gr.HTML — one per question slot
                answer_fields = []  # gr.Textbox — one per question slot
                for i in range(10):
                    qc = gr.HTML(value="", visible=False)
                    af = gr.Textbox(
                        label=f"Your answer",
                        placeholder="Type your answer here (leave empty to skip)",
                        lines=2,
                        visible=False,
                    )
                    q_cards.append(qc)
                    answer_fields.append(af)

                step2_loading = gr.HTML(value="", visible=False)

                with gr.Accordion("🔍 Debug: Raw LLM Response", open=False):
                    raw_json_output = gr.Textbox(label="Raw LLM Output", lines=20)

                submit_answers_btn = gr.Button("✅ Submit Answers", variant="primary", size="lg", visible=False)

                answers_json_state = gr.State("")

            # ================================================================
            # Step 3 – Validate spec
            # ================================================================
            with gr.Tab("📊 Step 3: Validate Spec", id="tab3"):

                step3_intro = gr.HTML(
                    value=(
                        _THEME_CSS
                        + "<div class='ttg-card ttg-card-info'>"
                        "<p class='ttg-p'>Validation results will appear here.</p>"
                        "</div>"
                    )
                )

                step3_loading = gr.HTML(value="", visible=False)
                validation_display = gr.HTML(value="", visible=False)

                # Follow-up section — interleaved FQ cards + answer boxes
                followup_section = gr.HTML(value="", visible=False)  # header card

                followup_cards = []   # gr.HTML per FQ slot
                followup_fields = []  # gr.Textbox per FQ slot
                for i in range(5):
                    fqc = gr.HTML(value="", visible=False)
                    fqf = gr.Textbox(
                        label="Your answer",
                        placeholder=f"Answer to follow-up question {i+1}",
                        lines=3,
                        visible=False,
                    )
                    followup_cards.append(fqc)
                    followup_fields.append(fqf)

                validate_btn = gr.Button("🔄 Re-validate", variant="primary", size="lg", visible=False)
                
                with gr.Accordion("🔍 Debug: Specification JSON", open=False):
                    specification_json_output = gr.Textbox(label="Specification JSON", lines=20)
                with gr.Accordion("🧠 Accumulated Understanding", open=False):
                    understanding_display = gr.JSON(label="Understanding")

            # ================================================================
            # Step 4 – Generate + check + test (fully automatic)
            # ================================================================
            with gr.Tab("🚀 Step 4: Generate & Test", id="tab4"):
                gr.Markdown("## Generate Environment")
                gr.Markdown(
                    "One click runs the full pipeline automatically:\n"
                    "1. **Generate** Gymnasium code from your spec\n"
                    "2. **Spec-check** — auto-applies all critical & warning fixes\n"
                    "3. **Runtime test** with SB3 — auto-debugs errors (up to 5 rounds)\n\n"
                    "When done, head to **Step 5** to train."
                )

                with gr.Row():
                    test_steps_slider = gr.Slider(
                        minimum=100, maximum=5000, value=1000, step=100,
                        label="Test Steps",
                        info="SB3 training steps used during runtime test",
                    )
                    max_debug_rounds_slider = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="Max Debug Rounds",
                        info="Max auto-fix attempts if runtime test fails",
                    )

                generate_env_btn = gr.Button("⚙️ Generate & Test Environment", variant="primary", size="lg")

                pipeline_status = gr.HTML(
                    value=_THEME_CSS + "<div class='ttg-card ttg-card-neutral'>"
                    "<p class='ttg-p'>Click the button above to start.</p></div>"
                )

                # Compact code viewer inside accordion
                with gr.Accordion("📄 View Generated Code", open=False):
                    code_output = gr.Code(label="Generated Environment Code", language="python", lines=20)

                with gr.Accordion("🧪 Runtime Test Report", open=False):
                    runtime_report = gr.HTML()
                with gr.Accordion("📋 Pipeline Log", open=False):
                    pipeline_log = gr.Textbox(label="Log", lines=15, interactive=False)

                code_state = gr.State("")

            # ================================================================
            # Step 5 – Training playground
            # ================================================================
            with gr.Tab("🎮 Step 5: Train", id="tab5"):
                gr.Markdown("## Train & Visualise")
                gr.Markdown(
                    "Train your environment with an SB3 algorithm and watch the reward curve live. "
                    "Make sure **Step 4** completed successfully before training."
                )

                with gr.Row():
                    algo_dropdown = gr.Dropdown(
                        choices=ALGORITHM_NAMES,
                        value="PPO",
                        label="Algorithm",
                        info="SAC and TD3 require a continuous (Box) action space.",
                    )
                    timesteps_slider = gr.Slider(
                        minimum=10_000, maximum=500_000, value=50_000, step=10_000,
                        label="Total Timesteps",
                        info="More timesteps = longer training = potentially better results",
                    )

                algo_info_display = gr.Markdown(_algo_info("PPO"))

                start_training_btn = gr.Button("▶️ Start Training", variant="primary", size="lg", elem_id="start-training-btn")
                with gr.Row():
                    stop_training_btn  = gr.Button("⏹️ Stop Training", variant="stop", size="lg", elem_id="stop-training-btn")
                    retest_btn         = gr.Button("🔄 Re-test & Fix Code", variant="secondary", size="lg", elem_id="retest-btn")

                training_status = gr.HTML(value=format_training_status("idle", "Ready. Click Start Training to begin."))
                reward_plot = gr.Plot(label="Reward Curve")

                with gr.Accordion("📋 Training Logs", open=False):
                    training_logs = gr.Textbox(label="Last 50 log lines", lines=15, interactive=False)
                with gr.Accordion("🔧 Re-test Results", open=False):
                    retest_report = gr.HTML()
                    retest_log    = gr.Textbox(label="Re-test log", lines=10, interactive=False)

        # --------------------------------------------------------------------
        # About
        # --------------------------------------------------------------------
        gr.Markdown("""
---
### 📚 About
Convert any decision-making problem description into a working Gymnasium environment — no RL expertise needed.

**Workflow:** Step 1 → Step 2 → Step 3 → Step 4 → Step 5

### 🔧 Setup
```bash
export OLLAMA_API_KEY="your-key"
export OPENAI_API_KEY="sk-..."
export OLLAMA_MODEL="gpt-oss:120b"
```
""")

        # ====================================================================
        # Event handlers
        # ====================================================================

        # ── Step 1: submit description ──────────────────────────────────────
        def on_submit_loading():
            """Show spinner immediately before the LLM call."""
            return gr.update(value=_LOADING_HTML, visible=True)

        def on_submit(provider, desc, example, *part_b_args):
            _, raw_json = submit_description(provider, desc, example, *part_b_args)

            try:
                data = json.loads(raw_json) if not raw_json.startswith(("Error", "===")) else {}
                questions = data.get("questions", [])
                num_q = len(questions)
            except Exception:
                questions = []
                num_q = 0

            # Warm intro card shown at top of Step 2
            if num_q > 0:
                intro_html = (
                    _THEME_CSS
                    + "<div class='ttg-card ttg-card-success'>"
                    "<p class='ttg-p'>Got it! Here are some follow-up questions to help me better understand your system. "
                    "Answer as many as you can — leave any that don't apply empty.</p>"
                    "</div>"
                )
            else:
                intro_html = (
                    _THEME_CSS
                    + "<div class='ttg-card ttg-card-warning'>"
                    "<p class='ttg-p'>Hmm, I couldn't generate questions from that description. "
                    "Try adding more detail in Part A and resubmitting.</p>"
                    "</div>"
                )

            # Build interleaved question card updates (10 slots)
            q_card_updates = []
            ans_updates = []
            for i in range(10):
                if i < num_q:
                    card_html = _THEME_CSS + _question_card_html(i + 1, questions[i])
                    q_card_updates.append(gr.update(value=card_html, visible=True))
                    ans_updates.append(gr.update(visible=True, value=""))
                else:
                    q_card_updates.append(gr.update(value="", visible=False))
                    ans_updates.append(gr.update(visible=False, value=""))

            submit_btn_update = gr.update(visible=(num_q > 0))

            return (
                [raw_json, desc, example, intro_html, raw_json]
                + q_card_updates
                + ans_updates
                + [submit_btn_update, gr.update(visible=False), gr.update(selected="tab2")]
            )

        # Show loading spinner first, then run the real work
        submit_btn.click(
            fn=on_submit_loading,
            inputs=[],
            outputs=[step1_loading],
        ).then(
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
            outputs=(
                [questions_json_state, part_a_desc_state, part_a_example_state,
                 step2_intro, raw_json_output]
                + q_cards
                + answer_fields
                + [submit_answers_btn, step1_loading, tabs]
            ),
        )

        # ── Step 2: submit answers → auto-trigger validation ─────────────────
        def on_submit_loading_step2():
            """Show spinner in Step 2 while processing."""
            return gr.update(value=_LOADING_HTML, visible=True)

        def on_process_and_validate(provider, questions_json, answers_json_in, understanding,
                                    round_num, qa_history, followup_questions, *answers):
            """Process answers from Step 2, then immediately run the first validation round."""
            # ── Part 1: process answers ────────────────────────────────────
            _, answers_json_str = process_answers(questions_json, *answers)

            try:
                questions_data = json.loads(questions_json) if questions_json else {}
                questions = questions_data.get("questions", [])
                history = [
                    {"question": q.get("question_text", ""), "answer": a.strip()}
                    for q, a in zip(questions, answers)
                    if a and a.strip()
                ]
            except Exception:
                history = []

            # ── Part 2: run first validation round ────────────────────────
            round_num = 1  # always round 1 when coming from Step 2

            try:
                html, val_json, val_data, updated = validate_specification(
                    "",  # desc stored in state — pass empty; spec_pipeline handles missing desc gracefully
                    "",  # example — same
                    questions_json, answers_json_str, provider,
                    current_understanding={}, round_number=round_num,
                    qa_history=history,
                )
            except Exception as e:
                html = _error_box(f"Validation error: {e}")
                val_json = ""
                val_data = None
                updated = {}

            fq_texts = []
            try:
                parsed = json.loads(val_json) if val_json else {}
                fq_texts = [fq.get("question", "") for fq in parsed.get("follow_up_questions", [])]
            except Exception:
                pass

            # Build follow-up card + textbox updates (5 slots)
            if fq_texts:
                fq_header = (
                    _THEME_CSS
                    + "<div class='ttg-card ttg-card-info'>"
                    "<p class='ttg-p'>Great progress! Before we move on, I need a bit more info to make sure my understanding "
                    "of the system matches yours. Please answer the questions below.</p>"
                    "</div>"
                )
            else:
                fq_header = ""

            validate_btn_visible = bool(fq_texts)

            if val_data and val_data.get("status") == "complete":
                spec_json = json.dumps(val_data.get("complete_specification", {}), indent=2)
                tab_target = gr.update(selected="tab4")
            else:
                spec_json = ""
                tab_target = gr.update(selected="tab3")

            step3_intro_html = (
                _THEME_CSS
                + "<div class='ttg-card ttg-card-success'>"
                "<p class='ttg-p'>Answers received! I've reviewed your responses below.</p>"
                "</div>"
            )
            return (
                answers_json_str, history,
                step3_intro_html,                          # _pending_intro_state
                html,                                      # _pending_display_state
                fq_header,                                 # _pending_section_state
                gr.update(visible=False),                  # step3_loading hidden
                val_json, spec_json,
                updated, updated,
                round_num,
                history,                                   # qa_history_state
                fq_texts,                                  # followup_questions_state
                gr.update(visible=validate_btn_visible),   # validate_btn
                tab_target,
            )

        def on_step2_loading():
            return gr.update(value=_LOADING_HTML, visible=True)

        def on_step3_loading():
            return gr.update(value=_LOADING_HTML, visible=True)

        # Instant helper — renders FQ card/field visibility from state.
        # Defined here so it can be used in both submit_answers_btn and validate_btn chains.
        def apply_fq_updates(fq_texts):
            updates = []
            for i in range(5):
                if i < len(fq_texts):
                    card_html = _THEME_CSS + _followup_card_html(i + 1, fq_texts[i])
                    updates.append(gr.update(value=card_html, visible=True))
                    updates.append(gr.update(visible=True, value=""))
                else:
                    updates.append(gr.update(value="", visible=False))
                    updates.append(gr.update(visible=False, value=""))
            return updates

        # Interleaved output list: [card0, field0, card1, field1, ...]
        _fq_interleaved = []
        for _c, _f in zip(followup_cards, followup_fields):
            _fq_interleaved.extend([_c, _f])

        def apply_all_display_updates(intro_html, display_html, section_html, fq_texts):
            """Fast: applies all pending HTML + FQ card/field visibility. No LLM call."""
            updates = [
                gr.update(value=intro_html),
                gr.update(value=display_html, visible=bool(display_html)),
                gr.update(value=section_html, visible=bool(section_html)),
            ]
            updates += apply_fq_updates(fq_texts)
            return updates

        submit_answers_btn.click(
            fn=on_step2_loading,
            inputs=[],
            outputs=[step2_loading],
        ).then(
            fn=on_process_and_validate,
            inputs=(
                [provider_dropdown,
                 questions_json_state, answers_json_state,
                 understanding_state, validation_round_state,
                 qa_history_state, followup_questions_state]
                + answer_fields
            ),
            outputs=[
                answers_json_state, qa_history_state,
                _pending_intro_state,
                _pending_display_state,
                _pending_section_state,
                step3_loading,
                specification_json_output, specification_state,
                understanding_state, understanding_display,
                validation_round_state,
                qa_history_state,
                followup_questions_state,
                validate_btn,
                tabs,
            ],
        ).then(
            fn=apply_all_display_updates,
            inputs=[_pending_intro_state, _pending_display_state,
                    _pending_section_state, followup_questions_state],
            outputs=[step3_intro, validation_display, followup_section] + _fq_interleaved,
        )

        # ── Step 3: re-validate (follow-up answers) ───────────────────────────
        def on_validate(provider, desc, example, questions_json,
                        answers_json, understanding, round_num, qa_history,
                        followup_questions, *followup_answers):
            """Re-validate after the user answers follow-up questions in Step 3."""

            def _fq_hide():
                cards = [gr.update(value="", visible=False) for _ in followup_cards]
                fields = [gr.update(visible=False, value="") for _ in followup_fields]
                return cards + fields

            try:
                original_answers = json.loads(answers_json) if answers_json else {}
                new_history_pairs = []
                for i, answer in enumerate(followup_answers, 1):
                    if answer and answer.strip():
                        key = f"FQ{i}_round{round_num}"
                        q_text = (
                            followup_questions[i - 1]
                            if i - 1 < len(followup_questions)
                            else f"Follow-up question {i}"
                        )
                        original_answers[key] = {"question": q_text, "answer": answer, "round": round_num}
                        new_history_pairs.append({"question": q_text, "answer": answer.strip()})

                updated_history = list(qa_history or []) + new_history_pairs
                updated_answers_json = json.dumps(original_answers, indent=2)
                round_num += 1

                if round_num > 2:
                    html = format_round_limit()
                    spec_json = json.dumps(understanding, indent=2)
                    return (
                        updated_answers_json,
                        html,                              # _pending_display_state
                        "",                                # _pending_section_state (empty)
                        gr.update(visible=False),          # step3_loading hidden
                        spec_json, spec_json,
                        understanding, understanding, round_num,
                        updated_history, [],               # clear followup_questions_state
                        gr.update(visible=False),
                        gr.update(selected="tab4"),
                    )

                html, val_json, val_data, updated = validate_specification(
                    desc, example, questions_json, updated_answers_json, provider,
                    current_understanding=understanding, round_number=round_num,
                    qa_history=updated_history,
                )

                next_fq_texts = []
                try:
                    parsed = json.loads(val_json) if val_json else {}
                    next_fq_texts = [fq.get("question", "") for fq in parsed.get("follow_up_questions", [])]
                except Exception:
                    pass

                if next_fq_texts:
                    fq_header = (
                        _THEME_CSS
                        + "<div class='ttg-card ttg-card-info'>"
                        "<p class='ttg-p'>Almost there! Just a couple more things to clarify.</p>"
                        "</div>"
                    )
                else:
                    fq_header = ""

                if val_data and val_data.get("status") == "complete":
                    spec_json = json.dumps(val_data.get("complete_specification", {}), indent=2)
                    tab_target = gr.update(selected="tab4")
                else:
                    spec_json = ""
                    tab_target = gr.update(selected="tab3")

                return (
                    updated_answers_json,
                    html,                              # _pending_display_state
                    fq_header,                         # _pending_section_state
                    gr.update(visible=False),          # step3_loading hidden
                    val_json, spec_json,
                    updated, updated, round_num,
                    updated_history, next_fq_texts,
                    gr.update(visible=bool(next_fq_texts)),
                    tab_target,
                )

            except Exception as e:
                error = _error_box(f"Error: {e}\n{traceback.format_exc()}")
                return (
                    answers_json,
                    error,                             # _pending_display_state
                    "",                                # _pending_section_state
                    gr.update(visible=False),          # step3_loading hidden
                    "", "",
                    understanding, understanding, round_num,
                    qa_history, [],
                    gr.update(visible=False),
                    gr.update(selected="tab3"),
                )

        validate_btn.click(
            fn=on_step3_loading,
            inputs=[],
            outputs=[step3_loading],
        ).then(
            fn=on_validate,
            inputs=[
                provider_dropdown,
                part_a_desc_state, part_a_example_state,
                questions_json_state, answers_json_state,
                understanding_state, validation_round_state,
                qa_history_state, followup_questions_state,
            ] + followup_fields,
            outputs=[
                answers_json_state,
                _pending_display_state,
                _pending_section_state,
                step3_loading,
                specification_json_output, specification_state,
                understanding_state, understanding_display,
                validation_round_state,
                qa_history_state, followup_questions_state,
                validate_btn,
                tabs,
            ],
        ).then(
            fn=apply_all_display_updates,
            inputs=[_pending_intro_state, _pending_display_state,
                    _pending_section_state, followup_questions_state],
            outputs=[step3_intro, validation_display, followup_section] + _fq_interleaved,
        )

        # ── Step 4: generate pipeline (generator) ────────────────────────────
        def on_generate_pipeline(spec_json, provider, test_steps, max_rounds):
            if not spec_json or not spec_json.strip():
                err = _error_box("Please complete Step 3 (validate spec) first!")
                yield err, "", "", "", ""
                return

            # Show loading state immediately
            yield _LOADING_HTML_PIPELINE, "", "", "", ""

            for progress_html, code, report_html, log_text in generate_full_pipeline(
                spec_json, provider,
                test_steps=int(test_steps),
                max_debug_rounds=int(max_rounds),
            ):
                yield progress_html, code, report_html, log_text, code

        generate_env_btn.click(
            fn=on_generate_pipeline,
            inputs=[specification_state, provider_dropdown,
                    test_steps_slider, max_debug_rounds_slider],
            outputs=[pipeline_status, code_output, runtime_report, pipeline_log, code_state],
        )

        # ── Algorithm info + sync algo/timesteps into state ─────────────────
        def _on_algo_change(algo):
            return _algo_info(algo), algo

        algo_dropdown.change(
            fn=_on_algo_change,
            inputs=[algo_dropdown],
            outputs=[algo_info_display, algo_state],
        )

        timesteps_slider.change(
            fn=lambda v: v,
            inputs=[timesteps_slider],
            outputs=[timesteps_state],
        )

        # ── Step 5: training ─────────────────────────────────────────────────
        # Use state copies of algo/timesteps to avoid Gradio 6 cross-firing
        # when components are shared as inputs across multiple event handlers.
        start_training_btn.click(
            fn=run_training,
            inputs=[code_state, algo_state, timesteps_state, provider_dropdown],
            outputs=[training_status, reward_plot, training_logs],
        )

        stop_training_btn.click(
            fn=stop_training,
            inputs=[],
            outputs=[training_status],
        )

        def on_retest(code, provider):
            if not code or not code.strip():
                return _error_box("No code to test."), "", code
            final_code, report_html, log_text = run_runtime_testing(
                code, provider, max_rounds=5, test_steps=1000
            )
            return report_html, log_text, final_code

        retest_btn.click(
            fn=on_retest,
            inputs=[code_state, provider_dropdown],
            outputs=[retest_report, retest_log, code_state],
        )

    return demo


demo = build_demo()
