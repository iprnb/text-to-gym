"""
Code pipeline: specification → Gymnasium environment code → validation → auto-fix.

Covers Steps 5, 5.5, and 5.7 of the UI workflow:
  5   – Generate Gymnasium environment Python code from the validated spec.
  5.5 – Validate the generated code against the spec (LLM-based mismatch review).
  5.7 – Runtime test: actually run the code with SB3 and auto-debug if it fails.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import traceback

from llm import call_llm, parse_llm_response
from prompts import CODE_GENERATION_SYSTEM, CODE_VALIDATION_SYSTEM, RUNTIME_DEBUG_SYSTEM
from formatting import (
    format_code_status, format_code_validation_report, format_runtime_test_report,
    format_pipeline_progress, _error_box,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown(code: str) -> str:
    """Remove ```python / ``` fences that some models add around code."""
    code = code.strip()
    for prefix in ("```python", "```"):
        if code.startswith(prefix):
            code = code[len(prefix):].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


def _class_name_from_spec(spec: dict) -> str:
    """Derive a PascalCase class name from the domain summary."""
    import re
    summary = spec.get("domain_summary", "") or spec.get("decision_maker", "")
    words = re.findall(r"[A-Za-z]+", summary)
    significant = [w for w in words if w.lower() not in {"a", "an", "the", "of", "for", "in", "to", "and", "or", "is", "are", "with"}]
    name = "".join(w.capitalize() for w in significant[:3])
    return (name + "Env") if name else "CustomEnv"


# ---------------------------------------------------------------------------
# Step 5 – Generate Gymnasium code
# ---------------------------------------------------------------------------

def generate_environment_code(specification_json, provider: str) -> tuple:
    """
    Generate Gymnasium environment code from a validated specification.
    Returns ``(status_html, python_code_string)``.
    Accepts either a JSON string or an already-parsed dict.
    """
    try:
        if isinstance(specification_json, dict):
            spec = specification_json
        else:
            spec = json.loads(specification_json)
        class_name = _class_name_from_spec(spec)

        code_gen_prompt = (
            f"Generate a complete Gymnasium environment based on this specification:\n\n"
            f"{json.dumps(spec, indent=2)}\n\n"
            f"Requirements:\n"
            f"- Class name: {class_name}\n"
            f"- Full Gymnasium compatibility (gymnasium, not gym)\n"
            f"- Well-documented code with clear docstrings\n"
            f"- Production-ready implementation\n"
            f"- Include proper error handling\n"
            f"- Use numpy for numerical operations\n"
        )

        print(f"\n{'='*60}\nGENERATING ENVIRONMENT CODE...\n{'='*60}")
        raw_code = call_llm(CODE_GENERATION_SYSTEM, code_gen_prompt, provider=provider)
        code = _strip_markdown(raw_code)

        print(f"Code generated! {len(code)} characters, {len(code.splitlines())} lines")

        try:
            ast.parse(code)
            syntax_ok = True
            syntax_msg = "✅ Syntax check passed"
        except SyntaxError as e:
            syntax_ok = False
            syntax_msg = f"❌ Syntax error on line {e.lineno}: {e.msg}"

        status_html = format_code_status(
            class_name, len(code), len(code.splitlines()), syntax_ok, syntax_msg
        )
        return status_html, code

    except Exception as e:
        return _error_box(f"Error generating code: {e}"), f"# Error occurred:\n# {e}\n\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Step 5.5 – Validate generated code against the specification
# ---------------------------------------------------------------------------

def validate_code_against_spec(spec_json, code: str, provider: str) -> tuple:
    """
    Ask the LLM to compare generated code against the spec.
    Returns ``(result_dict | None, raw_json_string)``.
    """
    try:
        if isinstance(spec_json, dict):
            spec_json = json.dumps(spec_json, indent=2)
        prompt = (
            f"## Environment Specification\n{spec_json}\n\n"
            f"## Generated Code\n```python\n{code}\n```\n\n"
            "Find all mismatches between the specification and the code."
        )
        print(f"\n{'='*60}\nVALIDATING CODE AGAINST SPEC...\n{'='*60}")
        response = call_llm(CODE_VALIDATION_SYSTEM, prompt, provider=provider)
        result = parse_llm_response(response)

        if not result:
            return None, response

        print(f"✅ Validation complete! Found {len(result.get('mismatches', []))} mismatches")
        return result, json.dumps(result, indent=2)

    except Exception as e:
        print(f"Error validating code: {e}")
        return None, str(e)


def apply_fixes(spec_json: str, code: str, selected_ids: list, all_mismatches: list, provider: str) -> str:
    """
    Ask the LLM to apply a specific set of mismatch fixes to the code.
    Returns the fixed code, or the original if something went wrong.
    """
    if not selected_ids or not all_mismatches:
        return code

    fixes_to_apply = [m for m in all_mismatches if m.get("id") in selected_ids]
    if not fixes_to_apply:
        return code

    fix_prompt = (
        "You are fixing a Gymnasium environment Python file.\n\n"
        "Apply ONLY these specific fixes to the code. Do not change anything else.\n\n"
        f"## Fixes to Apply\n{json.dumps(fixes_to_apply, indent=2)}\n\n"
        f"## Current Code\n```python\n{code}\n```\n\n"
        "Return ONLY the complete fixed Python code, no markdown, no explanations."
    )

    print(f"\n{'='*60}\nAPPLYING {len(fixes_to_apply)} FIXES...\n{'='*60}")
    fixed = call_llm(
        "You are an expert Python developer. Apply the requested fixes exactly.",
        fix_prompt,
        provider=provider,
    )
    fixed = _strip_markdown(fixed)

    try:
        ast.parse(fixed)
        print("✅ Fixed code passes syntax check")
        return fixed
    except SyntaxError as e:
        print(f"❌ Fixed code has syntax error: {e}")
        return code


# ---------------------------------------------------------------------------
# Step 5.7 – Runtime testing & auto-debug
# ---------------------------------------------------------------------------

# The test script template is defined as a module-level constant so it is
# easy to review and modify without hunting through function bodies.
_RUNTIME_TEST_SCRIPT = """\
import sys
sys.path.insert(0, "{tmpdir}")

import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("test_env", "{env_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

env_class = None
for name in dir(mod):
    obj = getattr(mod, name)
    if isinstance(obj, type) and name.endswith('Env'):
        env_class = obj
        break

if env_class is None:
    print("ERROR: No environment class found")
    sys.exit(1)

print(f"Found class: {{env_class.__name__}}")

try:
    env = env_class()
    print("✓ Environment instantiated")
except Exception as e:
    print(f"ERROR_INIT: {{e}}")
    import traceback; traceback.print_exc(); sys.exit(1)

try:
    obs, info = env.reset()
    print(f"✓ Reset successful, obs type: {{type(obs)}}")
except Exception as e:
    print(f"ERROR_RESET: {{e}}")
    import traceback; traceback.print_exc(); sys.exit(1)

try:
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    print("✓ Manual stepping works (100 steps)")
except Exception as e:
    print(f"ERROR_MANUAL_STEP_{{i}}: {{e}}")
    import traceback; traceback.print_exc(); sys.exit(1)

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

try:
    check_env(env)
    print("✓ SB3 env_checker passed")
except Exception as e:
    print(f"ERROR_ENV_CHECKER: {{e}}")
    import traceback; traceback.print_exc(); sys.exit(1)

try:
    from stable_baselines3 import PPO, SAC

    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Box):
        algorithm, algo_name = PPO, "PPO"
    elif isinstance(action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)):
        algorithm, algo_name = PPO, "PPO"
    else:
        print(f"ERROR_UNSUPPORTED_ACTION_SPACE: {{type(action_space).__name__}} is not supported by SB3")
        sys.exit(1)

    obs_space = env.observation_space
    policy = "MultiInputPolicy" if isinstance(obs_space, gym.spaces.Dict) else "MlpPolicy"
    print(f"✓ Using {{algo_name}} with {{policy}}")

    model = algorithm(policy, env, verbose=0)
    print(f"✓ {{algo_name}} model created successfully")

    model.learn(total_timesteps={num_steps}, progress_bar=False)
    print(f"✓ {{algo_name}} trained for {num_steps} steps")

    print("SUCCESS: Completed all tests including SB3 training")

except Exception as e:
    print(f"ERROR_SB3: {{e}}")
    import traceback; traceback.print_exc(); sys.exit(1)
"""


def test_environment_runtime(code: str, num_steps: int = 1000) -> tuple:
    """
    Run the environment code in a subprocess with a full SB3 compatibility
    test.  Returns ``(success: bool, error_info: dict | None)``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        env_path = os.path.join(tmpdir, "test_env.py")
        with open(env_path, "w") as f:
            f.write(code)

        test_script = _RUNTIME_TEST_SCRIPT.format(
            tmpdir=tmpdir, env_path=env_path, num_steps=num_steps
        )
        test_path = os.path.join(tmpdir, "test_runner.py")
        with open(test_path, "w") as f:
            f.write(test_script)

        try:
            result = subprocess.run(
                [sys.executable, test_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout + result.stderr
            if "SUCCESS:" in output:
                return True, None
            return False, {"output": output, "returncode": result.returncode}

        except subprocess.TimeoutExpired:
            return False, {"output": "Timeout: Environment took >60 s to run", "returncode": -1}
        except Exception as e:
            return False, {"output": str(e), "returncode": -1}


def _debug_and_fix(code: str, error_info: dict, provider: str, attempt: int = 1) -> tuple:
    """
    Ask the LLM to diagnose a runtime error and return fixed code.
    Returns ``(fixed_code, diagnosis_string)``.

    ``attempt`` is passed through to the prompt so the LLM knows this is a
    retry and should try a different approach if the previous fix did not work.
    """
    try:
        retry_note = (
            f"\n\nNOTE: This is attempt {attempt}. A previous fix attempt did not resolve the error. "
            "Try a different approach — look more carefully at the full traceback."
            if attempt > 1
            else ""
        )
        prompt = (
            f"## Environment Code\n```python\n{code}\n```\n\n"
            f"## Runtime Error (full output)\n{error_info['output']}\n\n"
            f"Find the root cause and provide the complete fixed code.{retry_note}"
        )
        response = call_llm(RUNTIME_DEBUG_SYSTEM, prompt, provider=provider)
        result = parse_llm_response(response)

        if not result:
            return code, "Could not parse LLM debug response"

        fixed_code = _strip_markdown(result.get("fixed_code", ""))
        if not fixed_code:
            return code, result.get("diagnosis", "No fix provided")

        try:
            ast.parse(fixed_code)
        except SyntaxError as e:
            return code, f"Fixed code has syntax error: {e}"

        return fixed_code, result.get("diagnosis", "")

    except Exception as e:
        return code, f"Error during debug: {e}"


def apply_all_fixes(spec_json: str, code: str, provider: str) -> str:
    """
    Run Step 5.5 (code-vs-spec validation) and automatically apply ALL
    critical and warning fixes.  Returns the patched code (or the original
    if validation failed or no changes were needed).
    """
    result, _ = validate_code_against_spec(spec_json, code, provider)
    if not result:
        return code

    mismatches = result.get("mismatches", [])
    # Apply critical and warning fixes automatically; skip info-level ones
    auto_ids = [
        m["id"] for m in mismatches
        if m.get("severity") in ("critical", "warning") and m.get("id")
    ]
    if not auto_ids:
        print("✅ No critical/warning mismatches – skipping auto-fix")
        return code

    print(f"🔧 Auto-fixing {len(auto_ids)} mismatch(es): {auto_ids}")
    return apply_fixes(spec_json, code, auto_ids, mismatches, provider)


def run_runtime_testing(
    code: str, provider: str, max_rounds: int = 5, test_steps: int = 1000
) -> tuple:
    """
    Iteratively test and auto-fix the environment code.
    Returns ``(final_code, report_html, log_text)``.

    Always runs up to ``max_rounds`` fix attempts.  The loop only exits early
    on success, or if the LLM returns identical code on **two consecutive**
    attempts (genuine stuck state).
    """
    log_entries: list[str] = []
    current_code = code
    all_rounds: list[dict] = []
    identical_count = 0  # consecutive rounds where LLM made no change

    print(f"\n{'='*60}\nRUNTIME TESTING (max {max_rounds} rounds, {test_steps} steps each)\n{'='*60}")

    for round_num in range(1, max_rounds + 1):
        log_entries.append(f"\n🧪 Round {round_num}/{max_rounds}")
        print(f"\n--- Test Round {round_num}/{max_rounds} ---")

        success, error_info = test_environment_runtime(current_code, num_steps=test_steps)

        if success:
            log_entries.append(f"✅ SUCCESS! Code ran {test_steps} steps with no errors.")
            all_rounds.append({"round": round_num, "success": True, "error": None, "diagnosis": "Code passed runtime test"})
            break

        error_output = error_info.get("output", "Unknown error")
        log_entries.append("❌ Runtime error detected")
        log_entries.append(f"Error: {error_output[:400]}")
        all_rounds.append({"round": round_num, "success": False, "error": error_output[:500], "diagnosis": "Running LLM debug..."})

        if round_num == max_rounds:
            log_entries.append(f"⚠️ Reached {max_rounds} rounds without success.")
            break

        log_entries.append("🔧 Asking LLM to debug and fix...")
        fixed_code, diagnosis = _debug_and_fix(current_code, error_info, provider, attempt=round_num)
        all_rounds[-1]["diagnosis"] = diagnosis

        if fixed_code == current_code:
            identical_count += 1
            log_entries.append(f"⚠️ LLM made no change (attempt {identical_count}/2 with no progress).")
            if identical_count >= 2:
                log_entries.append("⛔ Stopping: LLM unable to produce a different fix.")
                break
            # Continue to the next round — maybe a fresh attempt with more context helps
            log_entries.append("↩️ Retrying with updated context...")
        else:
            identical_count = 0
            current_code = fixed_code
            log_entries.append(f"✅ Fix applied: {diagnosis[:120]}")

    report_html = format_runtime_test_report(all_rounds, test_steps)
    return current_code, report_html, "\n".join(log_entries)


# ---------------------------------------------------------------------------
# Merged pipeline: Generate → Auto-check → Auto-test (Step 5 unified)
# ---------------------------------------------------------------------------

def generate_full_pipeline(specification_json: str, provider: str, test_steps: int = 1000, max_debug_rounds: int = 5):
    """
    Generator that runs the full Generate → Spec-check → Runtime-test pipeline
    automatically, yielding live status updates.

    Each yield: ``(progress_html, code_str, runtime_report_html, log_text)``
    where ``code_str`` is always the latest version of the code.
    """
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"

    def _progress(gen, chk, tst, msg="", fixes=0, rounds=0, passed=False):
        return format_pipeline_progress(gen, chk, tst, msg, fixes, rounds, passed)

    # Normalise: accept either a JSON string or an already-parsed dict
    if isinstance(specification_json, dict):
        specification_json = json.dumps(specification_json, indent=2)
    if not specification_json or not specification_json.strip():
        yield _error_box("Please validate specification first in Step 4!"), "", "", ""
        return

    # ── Stage 1: Generate ──────────────────────────────────────────────────
    yield _progress(RUNNING, PENDING, PENDING, "⚙️ Generating environment code…"), "", "", ""

    try:
        status_html, code = generate_environment_code(specification_json, provider)
    except Exception as e:
        yield _progress(ERROR, PENDING, PENDING, f"Generation failed: {e}"), "", "", ""
        return

    if not code or code.startswith("# Error"):
        yield _progress(ERROR, PENDING, PENDING, "Code generation returned an error."), "", status_html, ""
        return

    yield _progress(DONE, RUNNING, PENDING, "🔬 Checking code against spec…"), code, "", ""

    # ── Stage 2: Auto-check & auto-fix ────────────────────────────────────
    fixes_applied = 0
    try:
        result, _ = validate_code_against_spec(specification_json, code, provider)
        if result:
            mismatches = result.get("mismatches", [])
            auto_ids = [
                m["id"] for m in mismatches
                if m.get("severity") in ("critical", "warning") and m.get("id")
            ]
            if auto_ids:
                print(f"🔧 Auto-fixing {len(auto_ids)} mismatch(es)")
                code = apply_fixes(specification_json, code, auto_ids, mismatches, provider)
                fixes_applied = len(auto_ids)
    except Exception as e:
        print(f"Spec-check stage error (non-fatal): {e}")
        # Non-fatal — continue to runtime test with current code

    yield _progress(DONE, DONE, RUNNING, "🧪 Running runtime test…", fixes=fixes_applied), code, "", ""

    # ── Stage 3: Runtime test ─────────────────────────────────────────────
    final_code, report_html, log_text = run_runtime_testing(
        code, provider, max_rounds=max_debug_rounds, test_steps=test_steps
    )

    # Determine how many debug rounds were used
    rounds_used = log_text.count("🧪 Round")
    passed = "SUCCESS:" in log_text or "✅ SUCCESS" in log_text

    final_progress = _progress(
        DONE, DONE, DONE if passed else ERROR,
        "✅ Pipeline complete — ready to train!" if passed
        else "⚠️ Pipeline complete — some runtime issues remain. You may still try training.",
        fixes=fixes_applied, rounds=rounds_used, passed=passed,
    )
    yield final_progress, final_code, report_html, log_text
