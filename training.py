"""
Step 6 – Training playground.

Launches an SB3 training run in a subprocess, streams live reward data, and
renders a matplotlib reward curve.  Supports PPO and SAC.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback

from config import ALGORITHM_HYPERPARAMS
from formatting import format_training_status


# ---------------------------------------------------------------------------
# Training script template (filled in at runtime)
# ---------------------------------------------------------------------------

_TRAINING_SCRIPT = """\
import sys, json, time, numpy as np

sys.path.insert(0, "{env_dir}")

import importlib.util
spec = importlib.util.spec_from_file_location("environment", "{env_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

env_class = None
for name in dir(mod):
    obj = getattr(mod, name)
    if isinstance(obj, type) and name.endswith('Env'):
        env_class = obj
        break

if env_class is None:
    print("ERROR: Could not find environment class ending in 'Env'")
    sys.exit(1)

print(f"Found environment class: {{env_class.__name__}}")

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from stable_baselines3 import {algorithm}
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


class LiveLogCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.step_count = 0

    def _on_step(self):
        self.step_count += 1
        self.current_episode_reward += self.locals.get("rewards", [0])[0]
        if self.locals.get("dones", [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            with open(self.log_path, 'a') as f:
                f.write(json.dumps({{
                    "episode": len(self.episode_rewards),
                    "reward": float(self.current_episode_reward),
                    "step": self.step_count
                }}) + "\\n")
            self.current_episode_reward = 0
        return True


class DiscreteToBoxWrapper(ObservationWrapper):
    \"\"\"Converts Discrete sub-spaces in Dict obs to Box(shape=(1,)).

    Also reshapes scalar Box spaces with shape () to shape (1,).
    Required because SB3 MultiInputPolicy cannot flatten 0-dim observations.
    \"\"\"
    def __init__(self, env):
        super().__init__(env)
        if isinstance(env.observation_space, gym.spaces.Dict):
            new_spaces = {{}}
            for key, space in env.observation_space.spaces.items():
                if isinstance(space, gym.spaces.Discrete):
                    new_spaces[key] = gym.spaces.Box(low=0, high=space.n - 1, shape=(1,), dtype=np.int64)
                elif isinstance(space, gym.spaces.Box) and space.shape == ():
                    new_spaces[key] = gym.spaces.Box(low=space.low.reshape(1), high=space.high.reshape(1), shape=(1,), dtype=space.dtype)
                else:
                    new_spaces[key] = space
            self.observation_space = gym.spaces.Dict(new_spaces)
        self._discrete_keys = [
            k for k, s in env.observation_space.spaces.items() if isinstance(s, gym.spaces.Discrete)
        ] if isinstance(env.observation_space, gym.spaces.Dict) else []
        self._scalar_box_keys = [
            k for k, s in env.observation_space.spaces.items() if isinstance(s, gym.spaces.Box) and s.shape == ()
        ] if isinstance(env.observation_space, gym.spaces.Dict) else []

    def observation(self, obs):
        for key in self._discrete_keys:
            obs[key] = np.array([obs[key]], dtype=np.int64)
        for key in self._scalar_box_keys:
            obs[key] = np.array([obs[key]], dtype=self.env.observation_space[key].dtype)
        return obs


try:
    env = Monitor(env_class())
    print(f"Environment created | Action space: {{env.action_space}} | Obs space: {{env.observation_space}}")
except Exception as e:
    print(f"ERROR creating environment: {{e}}")
    import traceback; traceback.print_exc(); sys.exit(1)

if isinstance(env.observation_space, gym.spaces.Dict):
    has_discrete = any(isinstance(s, gym.spaces.Discrete) for s in env.observation_space.spaces.values())
    has_scalar_box = any(isinstance(s, gym.spaces.Box) and s.shape == () for s in env.observation_space.spaces.values())
    if has_discrete or has_scalar_box:
        env = DiscreteToBoxWrapper(env)
        print("Applied DiscreteToBoxWrapper to fix 0-dim observation spaces")

policy = "MultiInputPolicy" if isinstance(env.observation_space, gym.spaces.Dict) else "MlpPolicy"
print(f"Using policy: {{policy}}")

try:
    model = {algorithm}(policy, env, verbose=0, {hyperparams})
    print(f"Model created: {algorithm}")
except Exception as e:
    print(f"ERROR creating model: {{e}}")
    import traceback; traceback.print_exc(); sys.exit(1)

log_path = "{log_path}"
callback = LiveLogCallback(log_path)

print(f"Starting training for {total_timesteps} timesteps...")
try:
    model.learn(total_timesteps={total_timesteps}, callback=callback, progress_bar=False)
    print("TRAINING_COMPLETE")
except Exception as e:
    print(f"ERROR during training: {{e}}")
    import traceback; traceback.print_exc(); sys.exit(1)
"""


# ---------------------------------------------------------------------------
# Global training state (module-level so it can be stopped from any thread)
# ---------------------------------------------------------------------------

# Module-level state: this app is single-user (Gradio default), so globals are safe here.
training_process = None
training_should_stop = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_training(env_code: str, algorithm: str, total_timesteps: int, provider: str):
    """
    Generator that trains an SB3 agent and yields live status updates.
    Each ``yield`` is a tuple of ``(status_html, reward_plot_figure | None, log_text)``.
    """
    # NOTE: provider is not used during training (pure SB3 subprocess).
    # Reserved for future: auto-debug training failures via LLM.
    global training_process, training_should_stop
    training_should_stop = False

    if not env_code or not env_code.strip():
        yield (
            format_training_status("error", "No environment code found. Please generate code in Step 5 first."),
            None, "",
        )
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        env_path = os.path.join(tmpdir, "environment.py")
        log_path = os.path.join(tmpdir, "training_log.jsonl")

        with open(env_path, "w") as f:
            f.write(env_code)

        hyperparams = ALGORITHM_HYPERPARAMS.get(algorithm, "")
        script = _TRAINING_SCRIPT.format(
            env_dir=tmpdir,
            env_path=env_path,
            algorithm=algorithm,
            hyperparams=hyperparams,
            total_timesteps=int(total_timesteps),
            log_path=log_path,
        )

        script_path = os.path.join(tmpdir, "train.py")
        with open(script_path, "w") as f:
            f.write(script)

        yield (
            format_training_status("running", f"Starting {algorithm} training for {int(total_timesteps):,} timesteps..."),
            None, "",
        )

        try:
            training_process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            episodes, rewards, log_lines = [], [], []
            last_read_pos = 0
            start_time = time.time()

            while training_process.poll() is None:
                if training_should_stop:
                    training_process.terminate()
                    yield (
                        format_training_status("stopped", "Training stopped by user."),
                        _build_reward_plot(episodes, rewards, algorithm),
                        "\n".join(log_lines[-20:]),
                    )
                    return

                line = training_process.stdout.readline()
                if line:
                    log_lines.append(line.strip())

                if os.path.exists(log_path):
                    with open(log_path, "r") as f:
                        f.seek(last_read_pos)
                        new_lines = f.readlines()
                        last_read_pos = f.tell()
                    for nl in new_lines:
                        try:
                            entry = json.loads(nl.strip())
                            episodes.append(entry["episode"])
                            rewards.append(entry["reward"])
                        except Exception:
                            pass

                elapsed = time.time() - start_time
                yield (
                    format_training_status("running", f"Training {algorithm} | Episodes: {len(episodes)} | Elapsed: {elapsed:.0f}s"),
                    _build_reward_plot(episodes, rewards, algorithm),
                    "\n".join(log_lines[-20:]),
                )
                time.sleep(1.0)

            # Drain remaining log entries and stdout
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    f.seek(last_read_pos)
                    for nl in f.readlines():
                        try:
                            entry = json.loads(nl.strip())
                            episodes.append(entry["episode"])
                            rewards.append(entry["reward"])
                        except Exception:
                            pass

            remaining = training_process.stdout.read()
            if remaining:
                log_lines.extend(remaining.strip().split("\n"))

            elapsed = time.time() - start_time
            if training_process.returncode == 0:
                final_status = format_training_status("complete", f"✅ Training complete! {len(episodes)} episodes in {elapsed:.0f}s")
            else:
                final_status = format_training_status("error", f"❌ Training failed (exit code {training_process.returncode}). Check logs below.")

            yield (
                final_status,
                _build_reward_plot(episodes, rewards, algorithm),
                "\n".join(log_lines[-50:]),
            )

        except Exception as e:
            yield (
                format_training_status("error", f"Error: {e}\n{traceback.format_exc()}"),
                None,
                traceback.format_exc(),
            )


def stop_training() -> str:
    """Terminate the running training subprocess and return a status HTML string."""
    global training_should_stop, training_process
    training_should_stop = True
    if training_process and training_process.poll() is None:
        training_process.terminate()
    return format_training_status("stopped", "Stop requested. Training will halt shortly.")


# ---------------------------------------------------------------------------
# Reward plot
# ---------------------------------------------------------------------------

def _build_reward_plot(episodes: list, rewards: list, algorithm: str):
    """Return a matplotlib figure for the live reward curve, or ``None``."""
    if not episodes or not rewards:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(episodes, rewards, alpha=0.3, color="#90caf9", linewidth=1, label="Episode reward")

        if len(rewards) >= 5:
            window = max(5, len(rewards) // 20)
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(episodes[window - 1:], smoothed, color="#1976d2", linewidth=2, label=f"Smoothed (w={window})")

        ax.set_xlabel("Episode", color="#000000")
        ax.set_ylabel("Total Reward", color="#000000")
        ax.set_title(f"{algorithm} Training – Reward Curve", color="#000000", fontsize=13)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors="#000000")
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f9f9f9")
        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Plot error: {e}")
        return None
