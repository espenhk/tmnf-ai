"""<GAME_NAME> reward configuration and calculator.

Copy this to ``games/<your_game>/reward.py``.

The reward calculator computes a scalar reward after each ``env.step()``.
It wraps whatever native reward your game provides and optionally applies
shaping (bonuses, penalties, progress signals).

The framework calls ``compute()`` through the ``RewardCalculatorBase``
interface — your calculator is never imported directly by framework code.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import yaml

from framework.base_reward import RewardCalculatorBase


@dataclasses.dataclass
class TemplateRewardConfig:
    """Reward hyperparameters for <GAME_NAME>.

    Add fields for each tuneable reward parameter.  These map 1:1 to keys
    in ``config/reward_config.yaml``.
    """

    # Example fields — replace with your game's reward parameters:
    native_reward_scale: float = 1.0
    step_penalty: float = -0.1
    completion_bonus: float = 100.0

    @classmethod
    def from_yaml(cls, path: str) -> "TemplateRewardConfig":
        """Load config from a YAML file, ignoring unknown keys."""
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TemplateRewardCalculator(RewardCalculatorBase):
    """Stateful reward calculator for <GAME_NAME> episodes.

    Rename to ``<YourGame>RewardCalculator``.

    If your reward needs episode-level state (e.g. cumulative distance,
    visited checkpoints), track it here and reset in ``reset()``.
    """

    def __init__(self, config: TemplateRewardConfig) -> None:
        self._cfg = config

    def reset(self) -> None:
        """Reset episode-level reward state (called at env.reset())."""
        pass

    def compute(
        self,
        prev_state: Any,
        curr_state: Any,
        finished: bool,
        elapsed_s: float,
        info: dict,
        n_ticks: int = 1,
    ) -> float:
        """Compute the scalar reward for this step.

        Parameters
        ----------
        prev_state :
            Game state from the previous step (game-specific type).
        curr_state :
            Game state from the current step.
        finished :
            True if the episode terminated (not truncated).
        elapsed_s :
            Wall-clock seconds elapsed in this episode.
        info :
            The info dict from env.step().
        n_ticks :
            Number of game ticks this step covers (for variable framerate).

        Returns
        -------
        float
            The scalar reward signal.
        """
        raise NotImplementedError(
            "Compute and return a float reward.  Example:\n"
            "    reward = self._cfg.native_reward_scale * info.get('native_reward', 0.0)\n"
            "    reward += self._cfg.step_penalty\n"
            "    if finished:\n"
            "        reward += self._cfg.completion_bonus\n"
            "    return reward"
        )
