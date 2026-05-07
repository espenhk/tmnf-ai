"""StarCraft 2 reward calculator.

For minigames the canonical signal is the cumulative ``score`` returned by
PySC2 itself — so the calculator's main job is to compute the score *delta*
each step and add small shaping terms.  For the ladder game stub the same
signal is used as a placeholder; richer reward shaping (kill credit, mineral
income, supply lead) is left for the follow-up issue that adds learning.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import yaml

from framework.base_reward import RewardCalculatorBase


@dataclass
class SC2RewardConfig:
    """Reward weights for the SC2 environment.

    Python defaults match a sensible baseline for ``MoveToBeacon``.  Other
    minigames will typically want a different ``score_weight`` because the
    environment-provided score scales differ (e.g. mineral count vs.
    beacon-touch count).

    Parameters
    ----------
    score_weight :
        Coefficient on the ``score`` delta returned by PySC2 each step.
    win_bonus :
        One-shot bonus when the player reward signals a win (>0 from PySC2).
    loss_penalty :
        One-shot penalty when the player reward signals a loss (<0).
    step_penalty :
        Tiny negative reward every tick — discourages indefinite no-op.
    idle_penalty :
        Per-step penalty when ``army_count == 0 and food_used < food_cap``;
        used by ``BuildMarines`` to discourage doing nothing.
    idle_bonus :
        Per-step bonus awarded when the agent issues ``no_op`` *and* friendly
        units are within combat range of an enemy on the screen (issue #127).
        Default ``0.0`` — opt-in.  Makes the "stand still so units can shoot"
        lesson learnable rather than only discoverable through luck.  Requires
        the screen summary features (``screen_self_count`` / ``screen_enemy_count``
        / centroids) populated by the client; with the default obs preset
        these are always present.
    economy_weight :
        Coefficient on (minerals + vespene) delta.  Useful for economy
        minigames.  Set to 0 for pure-combat minigames.
    move_exploration_bonus :
        Per-step bonus for issuing ``Move_screen`` commands to spatial targets
        that differ from the previous move target.  Encourages exploration
        instead of spamming one point repeatedly.
    move_repeat_penalty :
        Per-step penalty when a ``Move_screen`` command targets almost the same
        point as the previous ``Move_screen`` command.
    move_self_penalty :
        Per-step penalty when a ``Move_screen`` command targets the centroid of
        currently-visible friendly units (a common "keep moving to where we
        already are" failure mode).
    """

    score_weight:    float = 1.0
    win_bonus:       float = 100.0
    loss_penalty:    float = -100.0
    step_penalty:    float = -0.001
    idle_penalty:    float = 0.0
    idle_bonus:      float = 0.0
    economy_weight:  float = 0.0
    move_exploration_bonus: float = 0.01
    move_repeat_penalty: float = -0.02
    move_self_penalty: float = -0.01

    @classmethod
    def from_yaml(cls, path: str) -> SC2RewardConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid_keys = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid_keys
        if unknown:
            raise ValueError(
                f"{path}: unknown reward config keys: {sorted(unknown)}\n"
                f"Valid keys: {sorted(valid_keys)}"
            )
        return cls(**data)


class SC2RewardCalculator(RewardCalculatorBase):
    """Reward computation for the SC2 environment.

    Stateless — episode-state derivatives (``prev_score``) are passed in via
    the ``info`` dict produced by :class:`games.sc2.env.SC2Env`.

    Expected keys in ``info``:
        ``score``         — current cumulative environment score
        ``prev_score``    — previous step's environment score
        ``minerals``, ``vespene`` — current totals
        ``prev_minerals``, ``prev_vespene`` — previous totals
        ``army_count``, ``food_used``, ``food_cap``
        ``player_outcome`` — None / +1 / -1 (only set on the final step)
        ``action_fn_idx`` — fn_idx of the action issued this step (for ``idle_bonus``)
        ``action_target_x`` / ``action_target_y`` — action target in [0, 1]
            (for movement exploration shaping)
        ``prev_move_target_x`` / ``prev_move_target_y`` — previous move target
            in [0, 1], or None if no previous move target is known
        ``screen_self_count`` / ``screen_enemy_count`` — friendly / enemy
            pixel counts on screen (for ``idle_bonus`` combat-range check)
        ``screen_self_cx`` / ``screen_self_cy`` / ``screen_enemy_cx`` /
            ``screen_enemy_cy`` — centroids in screen pixels
    """

    # Maximum centroid-distance for friendly units to be considered
    # "in combat range" for the ``idle_bonus`` shaping reward.  Expressed
    # as a fraction of the screen feature-layer side so the threshold
    # scales with non-default screen_size values (~Marine range at the
    # 64-pixel default ≈ 25 px).
    _COMBAT_RANGE_FRAC: float = 25.0 / 64.0
    # Radius (as a screen fraction) considered "same target" for repeated
    # move commands.
    _MOVE_REPEAT_RADIUS_FRAC: float = 2.0 / 64.0
    # Radius (as a screen fraction) considered "targeting where my units are".
    _MOVE_SELF_RADIUS_FRAC: float = 6.0 / 64.0
    # Distance normaliser for movement exploration bonus.
    _MOVE_EXPLORATION_NORM: float = 0.5

    def __init__(self, config: SC2RewardConfig) -> None:
        self.config = config

    def compute(
        self,
        prev_state: Any,
        curr_state: Any,
        finished: bool,
        elapsed_s: float,
        info: dict,
        n_ticks: int = 1,
    ) -> float:
        return self.compute_with_components(
            prev_state, curr_state, finished, elapsed_s, info, n_ticks,
        )[0]

    def compute_with_components(
        self,
        prev_state: Any,
        curr_state: Any,
        finished: bool,
        elapsed_s: float,
        info: dict,
        n_ticks: int = 1,
    ) -> tuple[float, dict[str, float]]:
        """Return ``(reward, components)`` for this step (issue #128/2b).

        ``components`` exposes a per-term breakdown so analytics can
        attribute reward to ``score``, ``economy``, ``idle_penalty``,
        ``idle_bonus``, ``move_exploration``, ``move_repeat_penalty``,
        ``move_self_penalty``, ``step_penalty`` and ``terminal`` separately.
        """
        cfg = self.config
        components: dict[str, float] = {}

        # Score delta — primary signal for minigames.
        prev_score = info.get("prev_score", 0.0)
        curr_score = info.get("score", 0.0)
        components["score"] = float(cfg.score_weight * (curr_score - prev_score))

        # Economy delta (optional — typically 0 for pure-combat minigames).
        if cfg.economy_weight != 0.0:
            prev_min = info.get("prev_minerals", 0.0)
            curr_min = info.get("minerals", 0.0)
            prev_vesp = info.get("prev_vespene", 0.0)
            curr_vesp = info.get("vespene", 0.0)
            components["economy"] = float(cfg.economy_weight * (
                (curr_min - prev_min) + (curr_vesp - prev_vesp)
            ))
        else:
            components["economy"] = 0.0

        # Idle penalty: nothing built and supply slack — encourages building.
        idle_pen = 0.0
        if cfg.idle_penalty != 0.0:
            army = info.get("army_count", 0.0)
            food_used = info.get("food_used", 0.0)
            food_cap = info.get("food_cap", 0.0)
            if army == 0 and food_used < food_cap:
                idle_pen = cfg.idle_penalty * n_ticks
        components["idle_penalty"] = float(idle_pen)

        # Idle bonus (issue #127): reward standing still when units are in
        # combat range of an enemy.  The pixel threshold scales with the
        # screen feature-layer size so non-default screen_size resolutions
        # behave consistently.
        idle_bonus = 0.0
        if cfg.idle_bonus != 0.0 and info.get("action_fn_idx") == 0:
            self_count  = info.get("screen_self_count", 0.0)
            enemy_count = info.get("screen_enemy_count", 0.0)
            if self_count > 0 and enemy_count > 0:
                dx = float(info.get("screen_self_cx", 0.0)) - float(info.get("screen_enemy_cx", 0.0))
                dy = float(info.get("screen_self_cy", 0.0)) - float(info.get("screen_enemy_cy", 0.0))
                dist = (dx * dx + dy * dy) ** 0.5
                screen_size = float(info.get("screen_size", 64))
                if dist <= self._COMBAT_RANGE_FRAC * screen_size:
                    idle_bonus = cfg.idle_bonus * n_ticks
        components["idle_bonus"] = float(idle_bonus)

        # Move shaping: encourage varied move targets and
        # discourage repeatedly targeting where units already are.
        move_exploration = 0.0
        move_repeat_penalty = 0.0
        move_self_penalty = 0.0
        if info.get("action_fn_idx") == 2:
            x = float(info.get("action_target_x", 0.5))
            y = float(info.get("action_target_y", 0.5))

            prev_x = info.get("prev_move_target_x")
            prev_y = info.get("prev_move_target_y")
            if prev_x is not None and prev_y is not None:
                dx_prev = x - float(prev_x)
                dy_prev = y - float(prev_y)
                dist_prev = (dx_prev * dx_prev + dy_prev * dy_prev) ** 0.5
                if cfg.move_exploration_bonus != 0.0:
                    novelty = min(1.0, dist_prev / self._MOVE_EXPLORATION_NORM)
                    move_exploration = cfg.move_exploration_bonus * novelty * n_ticks
                if cfg.move_repeat_penalty != 0.0 and dist_prev <= self._MOVE_REPEAT_RADIUS_FRAC:
                    move_repeat_penalty = cfg.move_repeat_penalty * n_ticks

            if cfg.move_self_penalty != 0.0:
                self_count = float(info.get("screen_self_count", 0.0))
                if self_count > 0:
                    screen_size = max(1.0, float(info.get("screen_size", 64)))
                    self_x = float(info.get("screen_self_cx", 0.0)) / screen_size
                    self_y = float(info.get("screen_self_cy", 0.0)) / screen_size
                    dx_self = x - self_x
                    dy_self = y - self_y
                    dist_self = (dx_self * dx_self + dy_self * dy_self) ** 0.5
                    if dist_self <= self._MOVE_SELF_RADIUS_FRAC:
                        move_self_penalty = cfg.move_self_penalty * n_ticks
        components["move_exploration"] = float(move_exploration)
        components["move_repeat_penalty"] = float(move_repeat_penalty)
        components["move_self_penalty"] = float(move_self_penalty)

        # Time cost.
        components["step_penalty"] = float(cfg.step_penalty * n_ticks)

        # Terminal win/loss bonus (only set when the env signals an outcome).
        terminal = 0.0
        outcome = info.get("player_outcome")
        if finished and outcome is not None:
            if outcome > 0:
                terminal = cfg.win_bonus
            elif outcome < 0:
                terminal = cfg.loss_penalty
        components["terminal"] = float(terminal)

        reward = float(sum(components.values()))
        return reward, components
