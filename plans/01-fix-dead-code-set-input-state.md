# Plan: Fix Dead Code — `set_input_state` Never Called on Normal Running Ticks

## Problem

In `clients/rl_client.py`, the block that applies the policy's action to the game
is **unreachable dead code**. The relevant section of `on_run_step()` (lines 269–286):

```python
if self._finish_respawn_pending:
    self._finish_respawn_pending = False
    self._episode_ready.clear()
    iface.give_up()
    self._simulation_finish_delivered = False
    self._running = False
    return           # ← exits here

        with self._action_lock:   # ← dead code: same indent level as return, never reached
            action = self._action
        steer_norm = float(np.clip(action[0], -1.0, 1.0))
        ...
        iface.set_input_state(accelerate=accel, brake=brake, steer=...)
```

The `with self._action_lock:` block is still inside the `if self._finish_respawn_pending:`
branch (16-space indentation), placed after `return`. It never executes.

On every **normal running tick** (when `_finish_respawn_pending` is False),
`iface.set_input_state()` is never called. The game receives no steering or throttle
input from the policy. The car drives using whatever TMInterface's last-set input state
is — either the braking state from the stop phase, or an accidental default.

## Root Cause

This was a misindentation during a refactor. The action-application block was meant to
run on every normal running tick (outside the `if _finish_respawn_pending:` guard), but
it was left at the wrong indentation level after the early-return was added.

## Fix

Move the action-application block to execute unconditionally when `_running` is True
and `_finish_respawn_pending` is False. The corrected structure:

```python
else:  # _running == True
    if self._finish_respawn_pending:
        self._finish_respawn_pending = False
        self._episode_ready.clear()
        iface.give_up()
        self._simulation_finish_delivered = False
        self._running = False
        return

    # Apply the RL thread's chosen action — runs every normal tick.
    with self._action_lock:
        action = self._action
    steer_norm = float(np.clip(action[0], -1.0, 1.0))
    accel = bool(float(action[1]) >= 0.5)
    brake = bool(float(action[2]) >= 0.5)
    iface.set_input_state(
        accelerate=accel,
        brake=brake,
        steer=int(steer_norm * 65536),
    )

    finished = ...
    hard_crash = ...
    ...
    self._drain_and_put(step_state)
```

The only change is dedenting the `with self._action_lock:` block by 4 spaces so it
falls outside the `if _finish_respawn_pending:` guard.

## Files to Change

| File | Change |
|------|--------|
| `clients/rl_client.py` | Dedent lines 277–286 by 4 spaces |

## Testing

1. Add a test that mocks `iface.set_input_state` and verifies it is called with the
   expected steer/accel/brake on each `on_run_step()` call when `_running=True` and
   `_finish_respawn_pending=False`
2. Run a short training session and confirm the car actually steers; before the fix
   the car drives straight regardless of policy output
3. Verify `_finish_respawn_pending=True` still triggers give_up without calling
   set_input_state (respawn path unchanged)
