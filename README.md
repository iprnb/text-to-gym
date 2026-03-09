# Text to Gym

A Gradio-based tool that turns a plain-language description of any decision-making problem into a working [Gymnasium](https://gymnasium.farama.org/) environment — no RL expertise required.

![CI](https://github.com/iprnb/text-to-gym/actions/workflows/tests.yml/badge.svg)

<img width="1864" height="771" alt="image" src="https://github.com/user-attachments/assets/91799812-34fc-45c9-8411-c877d1294f3f" />


---

## How it works

The app walks you through five steps, each handled by a separate tab:

| Step |  &nbsp;&nbsp;&nbsp;Tab&nbsp;&nbsp;&nbsp;  | What happens |
|------|-------|-------------|
| **1 – Describe** | Step 1 | Write a free-form description of your problem. Optionally fill in structured Part B questions (decision-maker, actions, observations, rewards, time structure, etc.) to give the LLM more context. |
| **2 – Q&A** | Step 2 | The LLM generates up to 10 targeted clarifying questions. Questions and answer boxes are shown as interleaved pairs (Q1 → A1 → Q2 → A2 …). Submit when done — validation starts automatically. |
| **3 – Validate Spec** | Step 3 | The LLM checks completeness and may ask up to 2 rounds of follow-up questions. When it's satisfied it produces a final structured specification and moves you to Step 4. |
| **4 – Generate & Test** | Step 4 | One click runs the full pipeline: **generate** Gymnasium code from the spec → **spec-check** (LLM compares code against spec and auto-applies critical/warning fixes) → **runtime test** with SB3 (auto-debugs errors up to 5 rounds). The final code is ready when the pipeline badge turns green. |
| **5 – Train** | Step 5 | Pick an algorithm (PPO, SAC, A2C, TD3), set total timesteps, and click **Start Training**. A live reward curve updates as training runs in a subprocess. Stop at any time, or re-test and auto-fix the code if needed. |

---

## Pipeline internals (Step 4)

Step 4 runs three stages automatically in sequence:

1. **Generate** — LLM produces a complete Gymnasium environment class with correct imports, `observation_space`, `action_space`, `reset()`, and `step()`.
2. **Spec-check** — A second LLM call compares the generated code against the structured spec and classifies mismatches as *critical*, *warning*, or *info*. All critical and warning fixes are applied automatically before moving on.
3. **Runtime test** — The code is executed in a subprocess with Stable-Baselines3: instantiation → `reset()` → manual steps → `env_checker` → model creation → short training run. If it fails, the LLM diagnoses the traceback and patches the code. Up to 5 debug rounds are attempted.

---

## Project layout

```
text-to-gym/
├── main.py            # Entry point — run this
├── config.py          # API keys, model names, algorithm registry
├── prompts.py         # All LLM system prompts and user-prompt templates
├── llm.py             # Ollama / OpenAI wrappers + JSON parser
├── formatting.py      # Theme-aware HTML helpers for all Gradio components
├── spec_pipeline.py   # Steps 1–3: description → clarifying questions → spec
├── code_pipeline.py   # Step 4: spec → generate → spec-check → runtime test
├── training.py        # Step 5: SB3 subprocess training + live reward plot
├── ui.py              # Gradio app: layout, state, event handlers
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Set your API keys as environment variables:

```bash
export OLLAMA_API_KEY="your-ollama-key"   # Ollama Cloud
export OPENAI_API_KEY="sk-..."            # OpenAI (optional)
export OLLAMA_MODEL="gpt-oss:120b"        # Override the default Ollama model
export MODEL_PROVIDER="ollama"            # "ollama" or "openai" (default: ollama)
```

## Run

```bash
python main.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## Algorithms

The following SB3 algorithms are available out of the box:

| Algorithm | Action space | Notes |
|-----------|-------------|-------|
| **PPO** | Discrete & continuous | Good general-purpose default |
| **A2C** | Discrete & continuous | Faster per-step than PPO, less sample-efficient |
| **SAC** | Continuous only | Best for continuous control |
| **TD3** | Continuous only | Best for continuous control, more stable than SAC |

To add a new algorithm, add one entry to `ALGORITHM_HYPERPARAMS` and one to `ALGORITHM_ACTION_SPACE_SUPPORT` in `config.py`. It will appear in the dropdown automatically. The algorithm name must be importable directly from `stable_baselines3` (e.g. `from stable_baselines3 import DDPG`); algorithms in `stable_baselines3.contrib` are not supported without modifying `training.py`.

---

## Requirements

- Python 3.10+
- An **Ollama Cloud** account **or** an **OpenAI API key**
- [`gradio`](https://www.gradio.app/) ≥ 6.0 — the web UI framework
- [`stable-baselines3`](https://stable-baselines3.readthedocs.io/) — used for runtime testing (Step 4) and training (Step 5)
- [`gymnasium`](https://gymnasium.farama.org/) — the generated environments use this API

---

## Status

Active development. The full pipeline (describe → train) works end-to-end.
Contributions and feedback welcome — open an issue to get started.

---

## License

MIT
