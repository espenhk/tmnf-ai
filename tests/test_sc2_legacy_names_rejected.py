"""Regression tests for removed SC2 bare-name policy types."""

from __future__ import annotations

import numpy as np
import pytest

from framework.training import _make_policy
from games.sc2.actions import DISCRETE_ACTIONS
from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC


def _import_policy_modules() -> None:
    import games.tmnf.policies  # noqa: F401
    import games.sc2.sc2_policies  # noqa: F401
    import games.sc2.cnn_policy  # noqa: F401


def test_sc2_bare_cmaes_rejected(tmp_path):
    """SC2 bare-name 'cmaes' should fail with Unknown policy_type."""
    _import_policy_modules()
    with pytest.raises(ValueError, match=r"Unknown policy_type: 'cmaes'") as exc:
        _make_policy(
            "cmaes",
            obs_spec=SC2_MINIGAME_OBS_SPEC,
            head_names=["fn_idx", "x", "y", "queue"],
            discrete_actions=np.array(DISCRETE_ACTIONS, dtype=np.float32),
            weights_file=str(tmp_path / "weights.yaml"),
            policy_params={},
            re_initialize=True,
            game_name="sc2",
        )
    assert "sc2_cmaes" in str(exc.value)
