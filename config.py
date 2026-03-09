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
# Algorithm registry used when launching SB3 training
#
# To add a new algorithm:
#   1. Add an entry to ALGORITHM_HYPERPARAMS with its SB3 keyword args.
#   2. Add an entry to ALGORITHM_ACTION_SPACE_SUPPORT indicating whether the
#      algorithm supports "discrete", "continuous", or "both".
#   The UI will automatically pick up new entries.
# ---------------------------------------------------------------------------

ALGORITHM_HYPERPARAMS: dict[str, str] = {
    "PPO": "learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10",
    "SAC": "learning_rate=3e-4, buffer_size=100_000, batch_size=256, learning_starts=1000",
    "A2C": "learning_rate=7e-4, n_steps=5",
    "TD3": "learning_rate=1e-3, buffer_size=100_000, batch_size=256, learning_starts=1000",
}

# Which action-space types each algorithm supports.
# "continuous" = gym.spaces.Box
# "discrete"   = gym.spaces.Discrete / MultiDiscrete / MultiBinary
# "both"       = works for either
ALGORITHM_ACTION_SPACE_SUPPORT: dict[str, str] = {
    "PPO": "both",
    "SAC": "continuous",   # SAC requires a continuous (Box) action space
    "A2C": "both",
    "TD3": "continuous",   # TD3 requires a continuous (Box) action space
}

ALGORITHM_NAMES: list[str] = list(ALGORITHM_HYPERPARAMS.keys())
