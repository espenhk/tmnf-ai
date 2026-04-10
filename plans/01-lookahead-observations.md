# Plan: Lookahead Track Observations

## Problem

The agent currently observes only its relationship to the **nearest centerline point**.
It has no awareness of upcoming track geometry — a sharp turn 2 seconds ahead looks
identical to a straight. This forces the policy to react to bends instead of anticipating
them, leading to late braking and wide cornering.

## Proposed Solution

Extend the observation vector with **N\_LOOKAHEAD = 3** upcoming waypoints. For each
waypoint (at centerline indices `+10`, `+25`, `+50` ahead of the current nearest point),
compute two values:

- **lateral\_offset** at that waypoint — how far the track curves left/right
- **yaw\_error** at that waypoint — the heading change relative to current car yaw

This adds **6 floats** (2 × 3) to the observation, bringing the base dimension from 15 to 21.

## Files to Change

### `obs_spec.py`
- Add constant `N_LOOKAHEAD = 3` and `LOOKAHEAD_STEPS = [10, 25, 50]`
- Append 6 entries to `OBS_SPEC` (after the existing 15):
  ```python
  ObsDim("lookahead_10_lat",  5.0,  "lateral offset 10 pts ahead (m)"),
  ObsDim("lookahead_10_yaw",  3.14, "yaw error 10 pts ahead (rad)"),
  ObsDim("lookahead_25_lat",  5.0,  "lateral offset 25 pts ahead (m)"),
  ObsDim("lookahead_25_yaw",  3.14, "yaw error 25 pts ahead (rad)"),
  ObsDim("lookahead_50_lat",  5.0,  "lateral offset 50 pts ahead (m)"),
  ObsDim("lookahead_50_yaw",  3.14, "yaw error 50 pts ahead (rad)"),
  ```
- Update `BASE_OBS_DIM = 15` → `BASE_OBS_DIM = 21`

### `track.py` — `Centerline` class
Add a new method `project_ahead(pos, nearest_idx, steps)`:
```python
def project_ahead(self, pos: Vec3, nearest_idx: int, steps: int) -> tuple[float, float]:
    """Return (lateral_offset, yaw_error) at centerline[nearest_idx + steps]."""
```
- Clamp `target_idx = min(nearest_idx + steps, len(self._points) - 2)`
- Compute forward direction from segment at `target_idx`
- Compute offset vector from `self._points[target_idx]` to `pos`
- Return lateral component (dot with right vector) and yaw error (angle between
  forward and car heading — caller must pass car yaw or compute from obs)

Alternatively, simplify by returning the **heading change** at the target point
(angle between segment at `nearest_idx` and segment at `target_idx`) rather than
relative to car yaw. This avoids needing to pass car yaw into the track module and
provides pure geometric curvature information.

### `utils.py` — `StateData.__init__`
After the existing `project_with_forward()` call that populates `track_progress`,
`lateral_offset`, etc., add:
```python
if centerline is not None:
    self.lookahead: list[tuple[float, float]] = [
        centerline.project_ahead(self.position, self._centerline_idx, s)
        for s in [10, 25, 50]
    ]
else:
    self.lookahead = [(0.0, 0.0)] * 3
```

### `rl/env.py` — `TMNFEnv`
- `__init__`: update `observation_space` shape from `15 + n_lidar` to `21 + n_lidar`
- `_make_obs()`: after building the 15-element base array, append
  `[lat for lat, _ in state_data.lookahead] + [yaw for _, yaw in state_data.lookahead]`
  (or interleaved — match `OBS_SPEC` ordering)

### No changes needed
- `clients/rl_client.py`: `StateData` is already constructed with the centerline; the new
  fields populate automatically
- `policies.py` `WeightedLinearPolicy._load_or_init()`: already detects obs dimension
  from the weight array shape and re-initialises on mismatch — no code change needed
- `rl/reward.py`: reward computation does not use lookahead

## Backward Compatibility

Existing saved weights (shape `3 × 15`) will not match the new obs dimension (`3 × 21`).
`WeightedLinearPolicy._load_or_init()` already handles this by re-initialising randomly
when the loaded shape mismatches. Add a log warning:
```
WARNING: Loaded weights shape (15,) doesn't match obs_dim=21. Re-initialising.
```
Old experiments continue to load; they just restart training from scratch.

## Testing

1. Run `python main.py test_lookahead` — confirm `env.observation_space.shape == (21,)`
2. Add a unit test in `tests/` that constructs a `StateData` with a known centerline
   and checks `lookahead` has 3 tuples of finite floats
3. Run a short training session (20 sims) and confirm `track_progress` improves compared
   to baseline (the agent should corner better after a few hundred sims)
