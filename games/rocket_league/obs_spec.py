"""Rocket League observation space definition.

Based on RLGym's ``DefaultObs`` which exposes car state, ball state, opponent
car state, and derived relative features.  The full vector is 70 floats.

Feature groups
--------------
  [0–17]   *Self car state* — position, linear velocity, angular velocity,
            forward/up unit vectors, on-ground flag, has-flip flag, boost amount.
  [18–26]  *Ball state* — position, linear velocity, angular velocity.
  [27–44]  *Opponent car state* — same layout as self car (18 dims).
  [45–47]  *Relative ball position* (ball_pos − car_pos), normalised.
  [48–50]  *Relative ball velocity* (ball_vel − car_vel), normalised.
  [51]     *Distance to ball* (normalised by arena diagonal ≈ 13 272 UU).
  [52]     *Velocity towards ball* — signed scalar, normalised.
  [53]     *Ball distance to opponent goal*, normalised.
  [54]     *Ball distance to own goal*, normalised.
  [55]     *Car distance to opponent goal*, normalised.
  [56]     *Car distance to own goal*, normalised.
  [57–59]  *Relative opponent position* (opp_pos − car_pos), normalised.
  [60–69]  *Nearest boost pads* — binary availability flags (10 pads).
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec

# ---------------------------------------------------------------------------
# Rocket League arena constants (Unreal Units)
# ---------------------------------------------------------------------------
# Arena: ±4096 × ±5120 × [0, 2044] UU; diagonal ≈ sqrt(8192² + 10240² + 2044²)
_ARENA_DIAG_UU: float = 13272.0
_MAX_VEL_UU: float = 2300.0      # max car linear velocity (UU/s)
_MAX_ANG_VEL: float = 5.5        # max angular velocity (rad/s)
_MAX_HEIGHT_UU: float = 2044.0   # ceiling height
_MAX_GOAL_DIST: float = 10240.0  # arena length

# ---------------------------------------------------------------------------
# Observation dimensions
# ---------------------------------------------------------------------------

_RL_DIMS: list[ObsDim] = [
    # --- Self car state (indices 0–17) ---
    ObsDim("car_pos_x",       4096.0,  "Self car X position (UU)"),
    ObsDim("car_pos_y",       5120.0,  "Self car Y position (UU)"),
    ObsDim("car_pos_z",       2044.0,  "Self car Z position (UU)"),
    ObsDim("car_vel_x",       2300.0,  "Self car X linear velocity (UU/s)"),
    ObsDim("car_vel_y",       2300.0,  "Self car Y linear velocity (UU/s)"),
    ObsDim("car_vel_z",       2300.0,  "Self car Z linear velocity (UU/s)"),
    ObsDim("car_ang_vel_x",   5.5,     "Self car X angular velocity (rad/s)"),
    ObsDim("car_ang_vel_y",   5.5,     "Self car Y angular velocity (rad/s)"),
    ObsDim("car_ang_vel_z",   5.5,     "Self car Z angular velocity (rad/s)"),
    ObsDim("car_forward_x",   1.0,     "Self car forward unit vector X"),
    ObsDim("car_forward_y",   1.0,     "Self car forward unit vector Y"),
    ObsDim("car_forward_z",   1.0,     "Self car forward unit vector Z"),
    ObsDim("car_up_x",        1.0,     "Self car up unit vector X"),
    ObsDim("car_up_y",        1.0,     "Self car up unit vector Y"),
    ObsDim("car_up_z",        1.0,     "Self car up unit vector Z"),
    ObsDim("is_on_ground",    1.0,     "1.0 if car is on ground, else 0.0"),
    ObsDim("has_flip",        1.0,     "1.0 if car has a flip available"),
    ObsDim("boost_amount",    1.0,     "Boost fuel remaining [0, 1]"),
    # --- Ball state (indices 18–26) ---
    ObsDim("ball_pos_x",      4096.0,  "Ball X position (UU)"),
    ObsDim("ball_pos_y",      5120.0,  "Ball Y position (UU)"),
    ObsDim("ball_pos_z",      2044.0,  "Ball Z position (UU)"),
    ObsDim("ball_vel_x",      6000.0,  "Ball X linear velocity (UU/s)"),
    ObsDim("ball_vel_y",      6000.0,  "Ball Y linear velocity (UU/s)"),
    ObsDim("ball_vel_z",      6000.0,  "Ball Z linear velocity (UU/s)"),
    ObsDim("ball_ang_vel_x",  6.0,     "Ball X angular velocity (rad/s)"),
    ObsDim("ball_ang_vel_y",  6.0,     "Ball Y angular velocity (rad/s)"),
    ObsDim("ball_ang_vel_z",  6.0,     "Ball Z angular velocity (rad/s)"),
    # --- Opponent car state (indices 27–44) ---
    ObsDim("opp_pos_x",       4096.0,  "Opponent car X position (UU)"),
    ObsDim("opp_pos_y",       5120.0,  "Opponent car Y position (UU)"),
    ObsDim("opp_pos_z",       2044.0,  "Opponent car Z position (UU)"),
    ObsDim("opp_vel_x",       2300.0,  "Opponent car X linear velocity (UU/s)"),
    ObsDim("opp_vel_y",       2300.0,  "Opponent car Y linear velocity (UU/s)"),
    ObsDim("opp_vel_z",       2300.0,  "Opponent car Z linear velocity (UU/s)"),
    ObsDim("opp_ang_vel_x",   5.5,     "Opponent car X angular velocity (rad/s)"),
    ObsDim("opp_ang_vel_y",   5.5,     "Opponent car Y angular velocity (rad/s)"),
    ObsDim("opp_ang_vel_z",   5.5,     "Opponent car Z angular velocity (rad/s)"),
    ObsDim("opp_forward_x",   1.0,     "Opponent car forward unit vector X"),
    ObsDim("opp_forward_y",   1.0,     "Opponent car forward unit vector Y"),
    ObsDim("opp_forward_z",   1.0,     "Opponent car forward unit vector Z"),
    ObsDim("opp_up_x",        1.0,     "Opponent car up unit vector X"),
    ObsDim("opp_up_y",        1.0,     "Opponent car up unit vector Y"),
    ObsDim("opp_up_z",        1.0,     "Opponent car up unit vector Z"),
    ObsDim("opp_on_ground",   1.0,     "1.0 if opponent is on ground"),
    ObsDim("opp_has_flip",    1.0,     "1.0 if opponent has a flip available"),
    ObsDim("opp_boost",       1.0,     "Opponent boost fuel remaining [0, 1]"),
    # --- Relative features (indices 45–59) ---
    ObsDim("rel_ball_pos_x",  4096.0,  "ball_pos_x − car_pos_x"),
    ObsDim("rel_ball_pos_y",  5120.0,  "ball_pos_y − car_pos_y"),
    ObsDim("rel_ball_pos_z",  2044.0,  "ball_pos_z − car_pos_z"),
    ObsDim("rel_ball_vel_x",  6000.0,  "ball_vel_x − car_vel_x"),
    ObsDim("rel_ball_vel_y",  6000.0,  "ball_vel_y − car_vel_y"),
    ObsDim("rel_ball_vel_z",  6000.0,  "ball_vel_z − car_vel_z"),
    ObsDim("dist_to_ball",    13272.0, "Euclidean distance from car to ball (UU)"),
    ObsDim("vel_towards_ball", 2300.0, "Signed velocity component towards ball (UU/s)"),
    ObsDim("ball_to_opp_goal_dist",  10240.0, "Ball distance to opponent goal (UU)"),
    ObsDim("ball_to_own_goal_dist",  10240.0, "Ball distance to own goal (UU)"),
    ObsDim("car_to_opp_goal_dist",   13272.0, "Car distance to opponent goal (UU)"),
    ObsDim("car_to_own_goal_dist",   13272.0, "Car distance to own goal (UU)"),
    ObsDim("rel_opp_pos_x",   4096.0,  "opp_pos_x − car_pos_x"),
    ObsDim("rel_opp_pos_y",   5120.0,  "opp_pos_y − car_pos_y"),
    ObsDim("rel_opp_pos_z",   2044.0,  "opp_pos_z − car_pos_z"),
    # --- Boost pad availability (indices 60–69, nearest 10 pads) ---
    ObsDim("boost_pad_0",     1.0, "Nearest boost pad 0 availability (0 or 1)"),
    ObsDim("boost_pad_1",     1.0, "Nearest boost pad 1 availability (0 or 1)"),
    ObsDim("boost_pad_2",     1.0, "Nearest boost pad 2 availability (0 or 1)"),
    ObsDim("boost_pad_3",     1.0, "Nearest boost pad 3 availability (0 or 1)"),
    ObsDim("boost_pad_4",     1.0, "Nearest boost pad 4 availability (0 or 1)"),
    ObsDim("boost_pad_5",     1.0, "Nearest boost pad 5 availability (0 or 1)"),
    ObsDim("boost_pad_6",     1.0, "Nearest boost pad 6 availability (0 or 1)"),
    ObsDim("boost_pad_7",     1.0, "Nearest boost pad 7 availability (0 or 1)"),
    ObsDim("boost_pad_8",     1.0, "Nearest boost pad 8 availability (0 or 1)"),
    ObsDim("boost_pad_9",     1.0, "Nearest boost pad 9 availability (0 or 1)"),
]

assert len(_RL_DIMS) == 70, f"Expected 70 dims, got {len(_RL_DIMS)}"

#: Canonical Rocket League observation spec (70 dims).
ROCKET_LEAGUE_OBS_SPEC: ObsSpec = ObsSpec(_RL_DIMS)

#: Number of base observation features.
BASE_OBS_DIM: int = ROCKET_LEAGUE_OBS_SPEC.dim

#: Ordered list of feature names.
OBS_NAMES: list[str] = ROCKET_LEAGUE_OBS_SPEC.names

#: Float32 scale array, shape (BASE_OBS_DIM,).
OBS_SCALES: np.ndarray = ROCKET_LEAGUE_OBS_SPEC.scales

#: Plain list of ObsDim entries.
OBS_SPEC: list[ObsDim] = _RL_DIMS
