# <GAME_NAME>

> ⚠️ **This is a template, not a runnable game.**
> Copy this entire directory to `games/<your_game>/` and fill in the blanks.

## Quick Start

```bash
cp -r games/_template games/<your_game>

# Then:
# 1. Rename classes (TemplateAdapter → YourGameAdapter, etc.)
# 2. Fill in NotImplementedError bodies
# 3. Define your observation space in obs_spec.py
# 4. Define your action space in actions.py
# 5. Implement env.py (reset + step)
# 6. Implement reward.py (compute)
# 7. Register in framework/game_adapter.py GAME_ADAPTERS dict
# 8. Add your game to main.py --game choices
# 9. Write a README.md for your game
# 10. Add tests
```

## Directory Structure

```
games/<your_game>/
├── __init__.py           # Package marker
├── adapter.py            # GameAdapter — wires everything together
├── env.py                # Gymnasium environment (BaseGameEnv subclass)
├── obs_spec.py           # Observation features (ObsSpec)
├── actions.py            # Discrete action set (numpy array)
├── reward.py             # RewardCalculator + config dataclass
├── analytics.py          # Results reporting + plots
├── config/
│   ├── training_params.yaml   # Training hyperparameters
│   └── reward_config.yaml     # Reward shaping parameters
└── README.md             # Game-specific docs (installation, running, etc.)
```

## Checklist

- [ ] Renamed all `Template*` classes to `<YourGame>*`
- [ ] Defined observation features in `obs_spec.py`
- [ ] Defined discrete actions in `actions.py`
- [ ] Implemented `env.py` (`reset()` and `step()`)
- [ ] Implemented `reward.py` (`compute()`)
- [ ] Implemented `adapter.py` (all methods)
- [ ] Updated `analytics.py` game name
- [ ] Configured `config/training_params.yaml`
- [ ] Configured `config/reward_config.yaml`
- [ ] Registered in `framework/game_adapter.py`:`GAME_ADAPTERS`
- [ ] Added to `main.py` `--game` choices
- [ ] Wrote game-specific `README.md`
- [ ] Added tests

## Registration

After filling in the template, register your game in two places:

### 1. `framework/game_adapter.py`

```python
GAME_ADAPTERS: dict[str, Callable[[], GameAdapter]] = {
    # ... existing games ...
    "your_game": lambda: __import__("games.your_game.adapter", fromlist=["make_adapter"]).make_adapter(),
}
```

### 2. `main.py`

Add your game slug to the `--game` `choices` list:

```python
choices=["tmnf", "beamng", "assetto", "car_racing", "torcs", "sc2", "your_game"],
```

## Reference Implementations

- **Smallest:** `games/car_racing/` — no external binary, pure Gymnasium
- **Medium:** `games/torcs/` — external game process, socket API
- **Full-featured:** `games/tmnf/` — Windows-only, TMInterface, centerlines, probes

## Protocols to Implement

| Protocol | File | Key methods |
|---|---|---|
| `GameAdapter` | `adapter.py` | `build_game_spec()`, `experiment_dir()`, etc. |
| `BaseGameEnv` | `env.py` | `reset()`, `step()`, `close()` |
| `RewardCalculatorBase` | `reward.py` | `compute()`, `reset()` |
| `ObsSpec` | `obs_spec.py` | Just construct with your `ObsDim` list |

See `docs/` or `CONTRIBUTING.md` for full protocol documentation.
