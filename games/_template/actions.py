"""<GAME_NAME> discrete action definitions.

Copy this to ``games/<your_game>/actions.py``.

Define the discrete action set your agent can choose from.  Each row in
``DISCRETE_ACTIONS`` maps to one atomic action the policy can select.

For the TMNF convention, actions are a 9-element grid:
  {left, straight, right} × {brake, coast, accelerate}

Your game may use a completely different scheme.  The only requirement
is that ``DISCRETE_ACTIONS`` is a numpy array of shape (N, action_dim).
"""

from __future__ import annotations

import numpy as np


# Replace with your game's action set.
# Each row is one selectable action.  Columns correspond to your action dims.
#
# Example (9-action grid for a racing game):
#   Column 0: steer  (-1 = left, 0 = straight, 1 = right)
#   Column 1: accel  (0 = none, 1 = full throttle)
#   Column 2: brake  (0 = none, 1 = full brake)
#
DISCRETE_ACTIONS = np.array([
    # [action_dim_1, action_dim_2, ...]
    # Example:
    # [-1., 0., 1.],  # left + brake
    # [ 0., 0., 1.],  # straight + brake
    # [ 1., 0., 1.],  # right + brake
    # [-1., 0., 0.],  # left + coast
    # [ 0., 0., 0.],  # straight + coast (no-op)
    # [ 1., 0., 0.],  # right + coast
    # [-1., 1., 0.],  # left + accel
    # [ 0., 1., 0.],  # straight + accel
    # [ 1., 1., 0.],  # right + accel
], dtype=np.float32).reshape(0, 0)  # Empty — fill in with your actions

# Optional: Named probe actions for the probe/cold-start phase.
# Each tuple is (action_array, human_label).
PROBE_ACTIONS: list[tuple[np.ndarray, str]] = [
    # (np.array([0., 1., 0.], dtype=np.float32), "full throttle"),
]

# Optional: Default warmup action (e.g. "do nothing" or "go forward").
# WARMUP_ACTION = np.array([0., 0., 0.], dtype=np.float32)
