# Changelog

## UI Overhaul & Bug Fixes (2026-03-09)

### Bug Fixes

**`ui.py`**
- Fixed Gradio 6 event cross-firing: `algo_dropdown` and `timesteps_slider` are now synced into `algo_state` / `timesteps_state` (`gr.State`) and `start_training_btn` reads from state — preventing Gradio from routing dropdown-change events into the training handler.
- Moved `start_training_btn` out of the shared `gr.Row` with Stop/Retest buttons; Gradio 6 misroutes clicks when multiple buttons with overlapping input/output components share a row.
- Removed `api_name=False` from all event handlers (it caused handler-ID shifts in Gradio 6's internal registry, breaking routing).
- Fixed `code_state` not being populated after pipeline generation: added `code_state` directly to `generate_env_btn.click` outputs so it is set atomically with `code_output` on every generator yield — previously relied on `code_output.change()` which does not fire on programmatic updates.
- Removed leftover `code_output.change(fn=lambda c: c, ...)` sync handler (now redundant).

**`code_pipeline.py`**
- Added `isinstance(x, dict)` guards in `generate_environment_code`, `validate_code_against_spec`, and `generate_full_pipeline` so they accept either a JSON string or an already-parsed dict as the spec argument (prevents `TypeError: expected string or bytes-like object, got 'dict'`).

**`training.py`**
- Fixed error message: "Please generate code in Step 5 first" → "Please generate code in Step 4 first".

### UI / UX Improvements

**`ui.py` (major restructure)**
- **Step 2 — interleaved Q&A**: Questions and answer boxes are now displayed as interleaved pairs (Q1 → A1 → Q2 → A2 …) instead of a questions block followed by a separate answers block. Each question card sits directly above its answer textbox.
- **Step 3 — auto-validation**: Clicking "Submit Answers" in Step 2 now automatically triggers the first validation round and switches to Step 3 — no separate button click required.
- **Step 3 — inline follow-up Q&A**: Follow-up questions and their answer boxes are also interleaved (FQ1 → A1 … FQ5 → A5). A warm friendly intro card ("Great progress!") appears above them.
- **Step 3 — warm intro message**: "Answers received! I've reviewed your responses below." card appears at the top when arriving from Step 2.
- **Step 2 — warm intro message**: After questions load, a friendly card ("Got it! Here are some follow-up questions…") replaces the placeholder text.
- **Step 4 — compact code viewer**: Generated code is now inside a collapsed `gr.Accordion("📄 View Generated Code")` — the tab is not dominated by a large code block.
- **Loading spinners**: Custom CSS animated spinner (`@keyframes ttg-spin` + `@keyframes ttg-pulse`) shown in Steps 1, 2, 3, and 4 during LLM/pipeline calls. Spinner components are kept out of slow call outputs to prevent Gradio loading indicators appearing on visible cards.
- **Gradio loading-indicator suppression**: Introduced three `gr.State` buffers (`_pending_intro_state`, `_pending_display_state`, `_pending_section_state`) so HTML content is computed inside the slow LLM call but applied by a fast instant `.then()` — display components are never in the slow call's outputs list.
- **Auto tab-switching**: App switches to Step 2 after Step 1 submit, Step 3 after Step 2 submit, and Step 4 when validation is complete.
- **Step 5 layout**: Start Training is a full-width standalone button; Stop and Retest share a row below it. All three have unique `elem_id` values.

### Refactor & Improvements (2026-03-09)

### Bug Fixes

**`llm.py`**
- `call_llm` now correctly forwards the `model` parameter to both `call_ollama` and `call_openai`. Previously the model kwarg was ignored, so overriding `OLLAMA_MODEL` via env var had no effect.

**`code_pipeline.py`**
- `_class_name_from_spec` now derives a PascalCase class name from the domain summary instead of always returning the hardcoded `"CustomEnv"`. Falls back to `"CustomEnv"` when the summary is empty.

**`training.py`**
- `matplotlib.use("Agg")` is now called once at module import time instead of inside `_build_reward_plot()` on every call, preventing runtime warnings when matplotlib is already imported.

**`spec_pipeline.py` / `code_pipeline.py`**
- Removed 8+ copies of inline error-box HTML strings. All error rendering now goes through `formatting._error_box()` for consistency and theme-compatibility.

### Code Optimisation

**`formatting.py` (complete rewrite)**
- Replaced every hardcoded hex colour (`#000000`, `#1976d2`, `#ffe6e6`, etc.) with CSS class-based styling using a single `_THEME_CSS` style block.
- CSS classes (`.ttg-card`, `.ttg-h3`, `.ttg-accent`, `.ttg-success`, `.ttg-error`, etc.) provide correct colour values in light mode and override them under `.dark` for dark-mode compatibility.
- Removed duplicate `!important` overrides scattered across every element — consolidated into the style block.
- `_p()` helper simplified: no longer accepts `color`/`margin` kwargs that callers were not using consistently.

**`spec_pipeline.py`**
- Imports `_error_box` from `formatting` and uses it throughout, eliminating duplicated inline HTML.

**`code_pipeline.py`**
- Same `_error_box` consolidation.

### New Features

**`config.py`**
- Added **algorithm registry**: `ALGORITHM_ACTION_SPACE_SUPPORT` maps each algorithm name to `"both"`, `"continuous"`, or `"discrete"` so the UI and training script can enforce compatibility.
- Added `A2C` and `TD3` to `ALGORITHM_HYPERPARAMS` and `ALGORITHM_ACTION_SPACE_SUPPORT`.
- Added `ALGORITHM_NAMES` list (derived from the registry keys) so the UI dropdown picks up new algorithms automatically — no UI changes needed to add an algorithm.
- To add a new algorithm: add one entry to `ALGORITHM_HYPERPARAMS` and one to `ALGORITHM_ACTION_SPACE_SUPPORT`. It will appear in the dropdown immediately.

**`training.py`**
- Training subprocess now validates action-space compatibility before creating the model. If a continuous-only algorithm (SAC, TD3) is used with a discrete action space, a clear `ERROR_ACTION_SPACE` message is printed and the process exits, surfacing as a readable error in the UI log rather than a cryptic SB3 traceback.
- Added guard: if `algorithm` is not in the registry, training exits with a clear error message.
- Reward plot: transparent background (`fig.patch.set_facecolor("none")`) so it adapts to both light and dark Gradio themes. Cleaner spine and grid styling.

### UI / UX Improvements

**`ui.py` (significant rewrite)**

- **Step 3 — inline questions**: Questions are now rendered directly on the Step 3 tab alongside their answer boxes. Users no longer need to switch between Step 2 and Step 3 to read questions while answering them. The inline panel updates automatically when Step 1 is submitted.
- **Algorithm info banner**: The Step 6 playground shows a live info line below the algorithm dropdown explaining which action-space types the selected algorithm supports (e.g. "⚠️ SAC requires a **continuous** (Box) action space."). Updates instantly on selection change.
- **Configurable algorithm list**: The dropdown in Step 6 is populated from `config.ALGORITHM_NAMES`, so adding an algorithm to `config.py` is all that's needed.
- **Font**: Applied **Inter** (via Google Fonts) as the UI font — formal, warm, and highly readable. Falls back to `ui-sans-serif` / `system-ui` when Google Fonts is unavailable.
- Inline error HTML in all event handlers replaced with `_error_box()` calls.
- Tab labels shortened for better readability at narrow widths.
- Step 5 "next steps" instructions updated to point to Steps 5.5 and 5.7 rather than a standalone `pip install` command.
- Provider dropdown given explicit `scale=1` so it doesn't stretch to fill the full row.

### Theme Compatibility

All `gr.HTML` output is now styled via CSS classes defined in `formatting._THEME_CSS`. The style block:

- Uses `var(--color-accent-soft)`, `var(--background-fill-primary)`, `var(--border-color-primary)` for elements that should follow Gradio's theme.
- Falls back to sensible light-mode hex values when the CSS variable is not set.
- Provides explicit `.dark .ttg-*` overrides for all card backgrounds and mismatch panels.
- Eliminates all hardcoded `color: #000000 !important` on text (which was invisible in dark mode).
