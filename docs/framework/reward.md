# `RewardCalculatorBase` and the `reward_config.yaml` round-trip

**Source:** `framework/base_reward.py` · **You implement:**
`games/<name>/reward.py`

The framework training loop calls your reward calculator after **every**
`env.step()`, without knowing anything about the game-specific signals. It
only ever holds a reference of type `RewardCalculatorBase`.

## The interface

```python
from framework.base_reward import RewardCalculatorBase


class MyGameRewardCalculator(RewardCalculatorBase):
    def compute(self, prev_state, curr_state, finished,
                elapsed_s, info, n_ticks=1) -> float:
        ...
```

### `compute(...)` — **abstract, you must implement**

Returns the scalar reward for one RL step.

| Param | Type | Meaning |
|---|---|---|
| `prev_state` | `Any` | Game-specific state snapshot from the *previous* step. |
| `curr_state` | `Any` | Game-specific state snapshot from the *current* step. |
| `finished` | `bool` | `True` when the game signalled episode completion (finish line, reached goal, …). |
| `elapsed_s` | `float` | Wall-clock seconds elapsed in the current episode. |
| `info` | `dict` | The dict produced by `env._get_game_info()` for this step (see [`base_env.md`](base_env.md)). **Read your reward signals from here** rather than adding bespoke positional params. |
| `n_ticks` | `int` | Number of game ticks covered by this RL step (`≥ 1`). Tick-rate-independent rewards should be scaled by `n_ticks`. |

### `compute_with_components(...)` — optional, default delegates

```python
def compute_with_components(...) -> tuple[float, dict]:
    return self.compute(...), {}   # default: no breakdown
```

Override to expose a per-term breakdown for analytics. Return
`(scalar, components)` where `components: dict[str, float]` maps each named
reward term to its contribution, and the scalar equals the **sum** of all
component values (identical to what `compute` returns). The default
returns an empty dict, signalling "no breakdown available". Component
breakdowns drive the reward-component charts in analytics, so populate
this if you want them.

### `reset()` — optional, default no-op

```python
def reset(self) -> None:
    pass   # default
```

Called at the start of each episode. Override if your calculator is
stateful (e.g. tracks a previous-position or a rolling baseline).

## The `RewardConfig` convention

`RewardCalculatorBase` only mandates `compute`. The repo-wide **convention**
(not enforced by the framework) is to pair each calculator with a
`RewardConfig` dataclass that:

1. declares every reward knob as a typed field with a default, and
2. provides a `from_yaml(path)` classmethod that loads
   `reward_config.yaml`, **ignoring unknown keys** so old/foreign config
   files still load.

```python
import dataclasses, yaml


@dataclasses.dataclass
class MyGameRewardConfig:
    native_reward_scale: float = 1.0
    step_penalty: float = -0.1
    finish_bonus: float = 100.0

    @classmethod
    def from_yaml(cls, path: str) -> "MyGameRewardConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        # Drop unknown keys so foreign / older configs still load.
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})
```

## The config round-trip

How a `reward_config.yaml` becomes a live calculator:

1. **Master copy.** `main.py` copies
   `games/<name>/config/reward_config.yaml` into the experiment directory
   on first run (so editing the experiment copy never touches the master).
2. **Decoration.** `main.py` loads that copy and calls
   `adapter.decorate_reward_cfg(reward_cfg, training_params, track_override)`,
   which **mutates the dict in place** to inject game-specific keys the
   YAML can't carry by itself — e.g. TMNF resolves the track's
   `centerline_path` and writes it in. The decorated dict is written back
   to disk. (See [`game_adapter.md`](game_adapter.md).)
3. **Load.** Your `GameSpec.reward_config_file` points at that decorated
   file; your calculator's `RewardConfig.from_yaml(...)` reads it and
   constructs the calculator.
4. **Per-step.** The loop calls `calc.compute(...)` after every step,
   feeding it `info` from `env._get_game_info()`.

`decorate_reward_cfg` is the hook for anything that depends on the chosen
track/map and therefore can't live as a static default in the master YAML.
If your game has no such keys, make it a no-op (`pass`) — as CarRacing
does.

## Worked example: CarRacing

CarRacing-v2 already emits a dense native reward (tiles visited per step),
so its calculator just shapes on top of it:

```python
@dataclasses.dataclass
class CarRacingRewardConfig:
    native_reward_scale: float = 1.0
    step_penalty: float = -0.1
    finish_bonus: float = 100.0
    crash_threshold_m: float = 25.0

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CarRacingRewardCalculator(RewardCalculatorBase):
    def __init__(self, config: CarRacingRewardConfig):
        self._cfg = config

    def reset(self):
        pass

    def compute(self, prev_state, curr_state, finished, elapsed_s, info, n_ticks=1):
        reward  = self._cfg.native_reward_scale * info.get("native_reward", 0.0)
        reward += self._cfg.step_penalty
        if finished:
            reward += self._cfg.finish_bonus
        return float(reward)
```

Note how the native reward arrives through `info["native_reward"]` (set by
the env's `_get_game_info`), keeping the calculator decoupled from the
env internals.
