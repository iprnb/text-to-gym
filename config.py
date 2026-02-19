"""
Configuration: model providers and environment variables.
"""

import os

# ---------------------------------------------------------------------------
# Model provider selection
# ---------------------------------------------------------------------------

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4"

# Ollama Cloud
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")

# ---------------------------------------------------------------------------
# Algorithm hyper-parameters used when launching SB3 training
# ---------------------------------------------------------------------------

ALGORITHM_HYPERPARAMS: dict[str, str] = {
    "PPO": "learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10",
    "SAC": "learning_rate=3e-4, buffer_size=100000, batch_size=256, learning_starts=1000",
}
