# text-to-Gym-environment Design Pipeline

A Gradio-based tool that turns natural language domain descriptions into working [Gymnasium](https://gymnasium.farama.org/) environments — no RL expertise required.

## How it works

| Step | What happens |
|------|-------------|
| **1 – Describe** | Write a free-form description of your decision-making problem, optionally filling in structured Part B questions. |
| **2 – Questions** | The LLM generates up to 10 targeted clarifying questions. |
| **3 – Answer** | Provide your answers to those questions. |
| **4 – Validate** | The LLM checks completeness and asks up to 2 rounds of follow-up questions before producing a final spec. |
| **5 – Generate** | Full Gymnasium-compatible Python environment code is generated from the spec. |
| **5.5 – Code check** | The LLM compares the code against the spec and flags mismatches you can selectively fix. |
| **5.7 – Runtime test** | The code is actually executed with SB3; if it fails, the LLM debugs and patches it automatically (up to 5 rounds). |
| **6 – Train** | Train with PPO or SAC and watch a live reward curve. |

## Project layout

```
.
├── main.py            # Entry point – run this
├── config.py          # Environment variables & constants
├── prompts.py         # All LLM system prompts & user-prompt templates
├── llm.py             # Ollama / OpenAI API wrappers + JSON parser
├── formatting.py      # HTML formatting helpers for the Gradio UI
├── spec_pipeline.py   # Steps 1-4: description → questions → spec
├── code_pipeline.py   # Steps 5-5.7: spec → code → validate → runtime test
├── training.py        # Step 6: SB3 training loop + reward plot
├── ui.py              # Gradio interface (tabs + event handlers)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Set your API keys:

```bash
export OLLAMA_API_KEY="your-ollama-key"   # Ollama Cloud
export OPENAI_API_KEY="sk-..."            # OpenAI (optional)
export OLLAMA_MODEL="gpt-oss:120b"        # Override default model
export MODEL_PROVIDER="ollama"            # "ollama" or "openai"
```

## Run

```bash
python main.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

## Requirements

- Python 3.10+
- An Ollama Cloud account **or** an OpenAI API key
- `stable-baselines3` for runtime testing and training (Steps 5.7 & 6)


## Note
This repository contains early-stage research prototypes.
APIs are incomplete and subject to change.
The focus is on exploring system design and evaluation patterns.

---

## License

MIT
