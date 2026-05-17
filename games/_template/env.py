"""<GAME_NAME> Gymnasium environment.

Copy this file to ``games/<your_game>/env.py`` and implement the
``reset()`` and ``step()`` methods.

If your game requires an external binary or SDK, guard the import behind
a try/except block so the rest of the framework can still be imported
without those dependencies installed.

Example (lazy import with helpful error)::

    try:
        import your_game_sdk
    except ImportError as _exc:
        raise ImportError(
            "<YourGame> requires the 'your-game-sdk' package.  Install with:\\n"
            "    pip install your-game-sdk"
        ) from _exc
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from framework.base_env import BaseGameEnv

# Import your obs spec dimension count:
# from games.<your_game>.obs_spec import BASE_OBS_DIM

# Import your reward calculator:
# from games.<your_game>.reward import YourRewardConfig, YourRewardCalculator

logger = logging.getLogger(__name__)

# Placeholder — replace with your game's observation dimension.
BASE_OBS_DIM = 1


class TemplateEnv(BaseGameEnv):
    """Gymnasium environment for <GAME_NAME>.

    Rename this class to ``<YourGame>Env``.

    Parameters
    ----------
    max_episode_steps :
        Maximum steps per episode before truncation.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        max_episode_steps: int = 1000,
    ) -> None:
        super().__init__()

        self._max_episode_steps = max_episode_steps

        # Define observation and action spaces for your game.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(BASE_OBS_DIM,),
            dtype=np.float32,
        )

        # Replace with your game's action space.
        # Continuous example: Box([-1, 0], [1, 1], shape=(2,))
        # Discrete example: Discrete(9)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._step_count: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to an initial state.

        Must return (observation, info).

        Implement:
        1. Reset your game to the starting state.
        2. Build and return the initial observation vector.
        3. Reset any internal reward calculator state.
        """
        super().reset(seed=seed)
        self._step_count = 0

        raise NotImplementedError(
            "Reset your game state, build the initial observation, and "
            "return (obs, info).  Example:\n"
            "    obs = np.zeros(BASE_OBS_DIM, dtype=np.float32)\n"
            "    return obs, {}"
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Run one step of the environment.

        Must return (observation, reward, terminated, truncated, info).

        Implement:
        1. Apply the action to your game.
        2. Read the new game state.
        3. Build the observation vector from the new state.
        4. Compute the reward using your RewardCalculator.
        5. Determine terminated (episode ended naturally) vs truncated
           (time limit or external cutoff).
        """
        self._step_count += 1

        raise NotImplementedError(
            "Apply action, observe new state, compute reward, and return "
            "(obs, reward, terminated, truncated, info)."
        )

    def close(self) -> None:
        """Clean up resources (game process, connections, etc.)."""
        pass

    def _build_obs(self, step: Any) -> np.ndarray:
        """Build a feature vector from raw game state.

        This method is called by the framework for observation construction.
        Convert your game's native state representation into a flat numpy
        array of shape (BASE_OBS_DIM,).
        """
        raise NotImplementedError(
            "Convert raw game state into a flat np.float32 observation array."
        )

    def get_episode_time_limit(self) -> float:
        """Return the current episode time limit in seconds."""
        raise NotImplementedError("Return episode time limit in seconds.")

    def set_episode_time_limit(self, seconds: float) -> None:
        """Update the episode time limit."""
        raise NotImplementedError("Set the episode time limit from seconds.")


def make_env(
    experiment_dir: str | Path,
    max_episode_time_s: float = 20.0,
) -> TemplateEnv:
    """Factory that wires up a TemplateEnv from an experiment directory.

    Rename to match your game.  Load reward config from the experiment
    directory and pass it to your Env class.
    """
    raise NotImplementedError(
        "Create and return your game environment.  Example:\n"
        "    return TemplateEnv(max_episode_steps=int(max_episode_time_s * 50))"
    )
