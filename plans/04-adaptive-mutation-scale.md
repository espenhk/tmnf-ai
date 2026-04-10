# Plan: Adaptive Mutation Scale (1/5th Success Rule)

## Problem

`MUTATION_SCALE = 0.1` in `main.py` is a fixed constant applied throughout all greedy
iterations. A large scale is beneficial early in training (broad exploration of weight
space) but detrimental late (disrupts near-optimal weights with excessive noise).

Currently, users tune this by hand per experiment, or accept suboptimal exploration
throughout the run. Neither ES gradient updates (`_greedy_loop()`, hill-climbing path)
nor the mutation-based greedy search (neural\_net path) adapt the scale automatically.

## Proposed Solution

Implement the **1/5th success rule** (Rechenberg, 1973) — a simple, parameter-free
adaptation heuristic used in Evolution Strategies:

- Track whether each simulation improved the best reward (rolling window of last
  `ADAPT_WINDOW = 20` sims)
- Every `ADAPT_WINDOW` sims, check the improvement rate `p = successes / ADAPT_WINDOW`:
  - `p > 1/5` → scale is too small (easy to improve) → **increase** by `ADAPT_UP = 1.2`
  - `p < 1/5` → scale is too large (rarely improving) → **decrease** by `ADAPT_DOWN = 0.85`
  - `p == 1/5` → no change
- Clamp to `[SCALE_MIN = 0.001, SCALE_MAX = 1.0]`

## Changes to `main.py`

### `_greedy_loop()` (lines 429–549)

Add at top of function:
```python
from collections import deque

improvement_history: deque[bool] = deque(maxlen=ADAPT_WINDOW)
ADAPT_WINDOW = 20
ADAPT_UP     = 1.2
ADAPT_DOWN   = 0.85
SCALE_MIN    = 0.001
SCALE_MAX    = 1.0
current_scale = mutation_scale  # local mutable copy
```

Inside the loop, after each simulation, record whether it improved:
```python
improved = reward > best_reward  # existing logic
improvement_history.append(improved)

# Adapt every ADAPT_WINDOW steps
if len(improvement_history) == ADAPT_WINDOW and sim_idx % ADAPT_WINDOW == 0:
    p = sum(improvement_history) / ADAPT_WINDOW
    if p > 1/5:
        current_scale = min(current_scale * ADAPT_UP, SCALE_MAX)
    elif p < 1/5:
        current_scale = max(current_scale * ADAPT_DOWN, SCALE_MIN)
```

Pass `current_scale` instead of `mutation_scale` to `policy.mutated()` and to the ES
perturbation generator:
```python
# ES path (line ~497):
epsilon = rng.normal(0, current_scale, flat_dim)
# Greedy mutate path:
candidate = current_policy.mutated(scale=current_scale)
```

### `train_rl()` — new parameter

```python
def train_rl(..., adaptive_mutation: bool = True, ...):
```

Pass `adaptive_mutation` into `_greedy_loop()`:
```python
best_policy, best_reward, sims = _greedy_loop(
    ..., mutation_scale=mutation_scale, adaptive_mutation=adaptive_mutation
)
```

If `adaptive_mutation=False`, skip the adaptation block entirely (for reproducibility
when comparing experiments with a fixed scale).

### Analytics

Log `current_scale` per sim alongside the existing per-sim data in `ExperimentData`.
This allows plotting `mutation_scale vs sim index` to visualise convergence behaviour.

## Why These Constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| `ADAPT_WINDOW` | 20 | Large enough to get a stable estimate; small enough to react quickly |
| `ADAPT_UP` | 1.2 | Modest increase to avoid overshooting |
| `ADAPT_DOWN` | 0.85 | Faster decrease than increase (asymmetry is standard in 1/5th rule) |
| `SCALE_MIN` | 0.001 | Prevents scale from collapsing to zero |
| `SCALE_MAX` | 1.0 | Prevents runaway exploration that degrades all weights |

## Files to Change

| File | Change |
|------|--------|
| `main.py` | Add adaptation logic to `_greedy_loop()`; add `adaptive_mutation` param to `train_rl()` |

No changes needed to `policies.py` — `WeightedLinearPolicy.mutated(scale=...)` already
accepts scale as an argument (line ~244).

## Testing

1. Run `python main.py adaptive_test` with `hill_climbing`, 200 sims, `adaptive_mutation=True`
2. Confirm `mutation_scale` logged per sim is not constant
3. Plot `mutation_scale vs sim` — expect it to trend downward over time as the policy converges
4. Compare final best reward vs a run with `adaptive_mutation=False` and the same seed
5. Verify `adaptive_mutation=False` produces identical results to the current codebase
   (regression test)
