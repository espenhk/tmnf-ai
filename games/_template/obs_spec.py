"""<GAME_NAME> observation space definition.

Copy this to ``games/<your_game>/obs_spec.py``.

Define each feature dimension your environment observes.  The framework
uses this for:
- Policy weight sizing (one weight per dimension per action head).
- Observation normalisation (raw_obs / scales).
- Weight file migration (new features auto-initialise to 0).

Example from CarRacing::

    ObsDim("speed", 100.0, "Vehicle speed (normalised)")

The ``scale`` value is the expected magnitude of that feature — used for
normalisation.  If the feature is already in [−1, 1], use scale=1.0.
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec


# Define your observation dimensions here.
# Each ObsDim is (name: str, scale: float, description: str).
_TEMPLATE_DIMS: list[ObsDim] = [
    # Example features — replace with your game's observations:
    # ObsDim("position_x",    100.0, "Agent X position (world units)"),
    # ObsDim("position_y",    100.0, "Agent Y position (world units)"),
    # ObsDim("velocity",       50.0, "Agent scalar velocity"),
    # ObsDim("heading",         6.3, "Agent heading in radians [0, 2π]"),
    ObsDim("placeholder", 1.0, "Remove this — add your real features above"),
]

#: The canonical observation spec for your game.
TEMPLATE_OBS_SPEC: ObsSpec = ObsSpec(_TEMPLATE_DIMS)

#: Number of base observation features.
BASE_OBS_DIM: int = TEMPLATE_OBS_SPEC.dim

#: Ordered list of feature names.
OBS_NAMES: list[str] = TEMPLATE_OBS_SPEC.names

#: Float32 scale array, shape (BASE_OBS_DIM,).
OBS_SCALES: np.ndarray = TEMPLATE_OBS_SPEC.scales

#: Plain OBS_SPEC list for callers that iterate over ObsDim entries.
OBS_SPEC: list[ObsDim] = _TEMPLATE_DIMS
