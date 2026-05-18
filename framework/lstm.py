"""LSTM cell and evolution-strategy wrapper for recurrent policies.

LSTMCore           — single-layer LSTM cell (pure numpy).  Manages (h, c)
                     state and exposes a flat parameter interface for ES.
LSTMEvolutionPolicy — wraps LSTMCore + game-injected head_decoder + isotropic-σ
                      ES outer optimiser.  Uses the same _greedy_loop_cmaes
                      interface as CMAESPolicy (sample_population /
                      update_distribution).

The isotropic-σ step-size adaptation follows the 1/5 success rule and is
identical to games/tmnf/policies.py:LSTMEvolutionPolicy.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


# ---------------------------------------------------------------------------
# LSTMCore — single-layer LSTM cell, no action emission
# ---------------------------------------------------------------------------

class LSTMCore:
    """Single-layer LSTM cell (pure numpy).

    Manages the hidden state ``(h, c)`` across steps.  Action emission is
    **not** part of this class; game-specific heads are injected via
    :class:`LSTMEvolutionPolicy`.

    Parameters
    ----------
    obs_dim :
        Dimensionality of the input observation (after normalisation).
    hidden_size :
        LSTM hidden / cell state dimensionality.
    seed :
        Optional RNG seed for He-initialised weights.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_size: int,
        seed: int | None = None,
    ) -> None:
        self._obs_dim     = int(obs_dim)
        self._hidden_size = int(hidden_size)

        h    = hidden_size
        c_in = h + obs_dim
        rng  = np.random.default_rng(seed)
        gain = np.sqrt(2.0 / c_in)

        self._W_f = rng.standard_normal((h, c_in)).astype(np.float32) * gain
        self._b_f = np.zeros(h, dtype=np.float32)
        self._W_i = rng.standard_normal((h, c_in)).astype(np.float32) * gain
        self._b_i = np.zeros(h, dtype=np.float32)
        self._W_g = rng.standard_normal((h, c_in)).astype(np.float32) * gain
        self._b_g = np.zeros(h, dtype=np.float32)
        self._W_o = rng.standard_normal((h, c_in)).astype(np.float32) * gain
        self._b_o = np.zeros(h, dtype=np.float32)

        self._h = np.zeros(h, dtype=np.float32)
        self._c = np.zeros(h, dtype=np.float32)

    # ------------------------------------------------------------------
    # Flat parameter interface (used by LSTMEvolutionPolicy)
    # ------------------------------------------------------------------

    @property
    def flat_dim(self) -> int:
        """Total number of LSTM cell parameters (weights + biases)."""
        h    = self._hidden_size
        c_in = h + self._obs_dim
        return 4 * (h * c_in + h)

    def to_flat(self) -> np.ndarray:
        """Serialise all LSTM cell parameters to a 1-D float32 vector."""
        return np.concatenate([
            self._W_f.ravel(), self._b_f,
            self._W_i.ravel(), self._b_i,
            self._W_g.ravel(), self._b_g,
            self._W_o.ravel(), self._b_o,
        ]).astype(np.float32)

    def from_flat(self, theta: np.ndarray) -> None:
        """Load LSTM cell parameters in-place from a flat vector."""
        theta = np.asarray(theta, dtype=np.float32)
        if theta.shape[0] != self.flat_dim:
            raise ValueError(
                f"LSTMCore.from_flat: expected {self.flat_dim} elements, "
                f"got {theta.shape[0]}"
            )
        h    = self._hidden_size
        c_in = h + self._obs_dim
        off  = 0

        def _take(shape: tuple) -> np.ndarray:
            nonlocal off
            n   = int(np.prod(shape))
            out = theta[off: off + n].reshape(shape).copy()
            off += n
            return out

        self._W_f = _take((h, c_in))
        self._b_f = _take((h,))
        self._W_i = _take((h, c_in))
        self._b_i = _take((h,))
        self._W_g = _take((h, c_in))
        self._b_g = _take((h,))
        self._W_o = _take((h, c_in))
        self._b_o = _take((h,))

    @classmethod
    def _empty(cls, obs_dim: int, hidden_size: int) -> "LSTMCore":
        """Create an LSTMCore without initialising weights.

        Use immediately before :meth:`from_flat` to avoid wasting random-number
        generation that would be discarded anyway.  The hidden and cell states
        are still zeroed.
        """
        obj = object.__new__(cls)
        obj._obs_dim     = int(obs_dim)
        obj._hidden_size = int(hidden_size)
        h    = hidden_size
        c_in = h + obs_dim
        obj._W_f = np.empty((h, c_in), dtype=np.float32)
        obj._b_f = np.empty(h, dtype=np.float32)
        obj._W_i = np.empty((h, c_in), dtype=np.float32)
        obj._b_i = np.empty(h, dtype=np.float32)
        obj._W_g = np.empty((h, c_in), dtype=np.float32)
        obj._b_g = np.empty(h, dtype=np.float32)
        obj._W_o = np.empty((h, c_in), dtype=np.float32)
        obj._b_o = np.empty(h, dtype=np.float32)
        obj._h   = np.zeros(h, dtype=np.float32)
        obj._c   = np.zeros(h, dtype=np.float32)
        return obj

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run one LSTM step on normalised input *x*, return updated hidden state.

        Parameters
        ----------
        x :
            Normalised observation vector of shape ``(obs_dim,)``.

        Returns
        -------
        np.ndarray
            Updated hidden state ``h``, shape ``(hidden_size,)``.
        """
        hx = np.concatenate([self._h, x.astype(np.float32)])

        f = _sigmoid(self._W_f @ hx + self._b_f)
        i = _sigmoid(self._W_i @ hx + self._b_i)
        g = np.tanh(self._W_g  @ hx + self._b_g)
        o = _sigmoid(self._W_o @ hx + self._b_o)

        self._c = f * self._c + i * g
        self._h = o * np.tanh(self._c)
        return self._h.copy()

    def reset(self) -> None:
        """Reset hidden and cell states to zero."""
        self._h = np.zeros(self._hidden_size, dtype=np.float32)
        self._c = np.zeros(self._hidden_size, dtype=np.float32)


# ---------------------------------------------------------------------------
# _LSTMIndividual — callable BasePolicy wrapping LSTMCore + head weights
# ---------------------------------------------------------------------------

class _LSTMIndividual(BasePolicy):
    """A single LSTMEvolutionPolicy population member.

    Wraps an :class:`LSTMCore` together with a flat head-weight vector and a
    game-injected *head_decoder* that converts ``(h, head_params)`` → action.

    Exposes :meth:`to_flat` so that the convergence test can measure distance
    from a target parameter vector in the combined parameter space.
    """

    def __init__(
        self,
        core: LSTMCore,
        head_params: np.ndarray,
        head_decoder: Callable[[np.ndarray, np.ndarray], np.ndarray],
        scales: np.ndarray,
    ) -> None:
        self._core         = core
        self._head_params  = np.asarray(head_params, dtype=np.float32)
        self._head_decoder = head_decoder
        self._scales       = scales

    # Expose h/c from the core for test assertions
    @property
    def _h(self) -> np.ndarray:
        return self._core._h

    @property
    def _c(self) -> np.ndarray:
        return self._core._c

    def to_flat(self) -> np.ndarray:
        """Concatenate [lstm_core_params | head_params] into one float32 vector."""
        return np.concatenate([self._core.to_flat(), self._head_params]).astype(np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs / self._scales).astype(np.float32)
        h = self._core.forward(x)
        return self._head_decoder(h, self._head_params)

    def update(self, obs, action, reward, next_obs, done, **kwargs) -> None:
        pass  # no online update; training via outer evolutionary optimiser

    def on_episode_start(self, **kwargs) -> None:
        self._core.reset()

    def on_episode_end(self) -> None:
        self._core.reset()

    def to_cfg(self) -> dict:
        return {"policy_type": "lstm_individual"}


# ---------------------------------------------------------------------------
# LSTMEvolutionPolicy — isotropic-σ ES over LSTMCore + head
# ---------------------------------------------------------------------------

class LSTMEvolutionPolicy(BasePolicy):
    """Isotropic-σ ES outer optimiser wrapping :class:`LSTMCore` + head_decoder.

    Uses the ``_greedy_loop_cmaes`` interface (``sample_population`` /
    ``update_distribution``).  The search distribution is an isotropic Gaussian
    with step-size adaptation via the 1/5 success rule.

    The full flat parameter vector is ``[lstm_core_params | head_params]``.

    Parameters
    ----------
    obs_spec :
        Observation spec (provides ``dim`` and ``scales``).
    head_decoder :
        Callable ``(h: np.ndarray, head_params: np.ndarray) -> np.ndarray``
        that maps ``(hidden_state, head_weight_vector)`` to an action array.
        For TMNF this would compute steer/accel/brake from the hidden state
        using the head weights.
    head_param_count :
        Number of head parameters (length of the head_params slice).
    hidden_size :
        LSTM hidden / cell dimensionality (default 32).
    population_size :
        λ — offspring per generation (default 20).
    initial_sigma :
        Starting step size (default 0.05; smaller than CMAESPolicy because
        LSTM weight spaces are much larger).
    available_actions_fn :
        Optional ``(info: dict) -> np.ndarray[bool]`` mask applied by the
        head_decoder to restrict actions.  Passed through to individuals for
        games that use masking.
    seed :
        Optional RNG seed.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        head_decoder: Callable[[np.ndarray, np.ndarray], np.ndarray],
        head_param_count: int,
        *,
        hidden_size: int = 32,
        population_size: int = 20,
        initial_sigma: float = 0.05,
        available_actions_fn: Callable[[dict], np.ndarray] | None = None,
        seed: int | None = None,
    ) -> None:
        self._obs_spec        = obs_spec
        self._obs_dim         = obs_spec.dim
        self._scales          = obs_spec.scales
        self._head_decoder    = head_decoder
        self._head_param_count = int(head_param_count)
        self._avail_fn        = available_actions_fn

        self._lam   = int(population_size)
        self._sigma = float(initial_sigma)
        self._rng   = np.random.default_rng(seed)

        # Template core used to derive flat_dim and generate individuals
        self._lstm_core_template = LSTMCore(
            obs_dim     = self._obs_dim,
            hidden_size = int(hidden_size),
            seed        = seed,
        )
        self._lstm_core_flat_dim = self._lstm_core_template.flat_dim
        self._flat_dim = self._lstm_core_flat_dim + self._head_param_count

        # Search distribution mean (core + head concatenated)
        self._mean = np.concatenate([
            self._lstm_core_template.to_flat(),
            np.zeros(self._head_param_count, dtype=np.float32),
        ]).astype(np.float64)

        # Weighted recombination (log-based, top-mu elites)
        mu               = self._lam // 2
        self._mu         = mu
        raw_w            = np.array(
            [np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)],
            dtype=np.float64,
        )
        self._recomb_w   = raw_w / raw_w.sum()

        self._pop: list[np.ndarray] = []
        self._champion: BasePolicy | None = None
        self._champion_reward: float      = float("-inf")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flat_to_individual(self, flat: np.ndarray) -> _LSTMIndividual:
        """Build a new :class:`_LSTMIndividual` from a flat parameter vector."""
        flat = np.asarray(flat, dtype=np.float32)
        # Use _empty to skip weight initialisation that would be discarded.
        core = LSTMCore._empty(
            obs_dim     = self._obs_dim,
            hidden_size = self._lstm_core_template._hidden_size,
        )
        core.from_flat(flat[:self._lstm_core_flat_dim])
        head_params = flat[self._lstm_core_flat_dim:].copy()
        return _LSTMIndividual(core, head_params, self._head_decoder, self._scales)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def population_size(self) -> int:
        return self._lam

    @property
    def champion_reward(self) -> float:
        return self._champion_reward

    @property
    def sigma(self) -> float:
        return self._sigma

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_from_champion(self, champion) -> None:
        """Seed the search mean from an existing champion's to_flat() vector."""
        if champion.to_flat().shape[0] != self._flat_dim:
            raise ValueError(
                f"LSTMEvolutionPolicy: champion flat_dim mismatch — "
                f"expected {self._flat_dim}, got {champion.to_flat().shape[0]}."
            )
        self._champion = champion
        self._mean     = champion.to_flat().astype(np.float64)
        logger.info("[LSTMEvolutionPolicy] seeded mean from champion")

    # ------------------------------------------------------------------
    # Population-loop interface
    # ------------------------------------------------------------------

    def sample_population(self) -> list[_LSTMIndividual]:
        """Draw λ offspring from the isotropic Gaussian and decode to individuals."""
        self._pop = []
        for _ in range(self._lam):
            z = self._rng.standard_normal(self._flat_dim)
            self._pop.append(self._mean + self._sigma * z)
        return [self._flat_to_individual(np.array(x, dtype=np.float32))
                for x in self._pop]

    def update_distribution(self, rewards: list[float]) -> bool:
        """Weighted mean recombination + 1/5 success rule step-size adaptation."""
        if len(rewards) != self._lam:
            raise ValueError(f"Expected {self._lam} rewards, got {len(rewards)}")
        if len(self._pop) != self._lam:
            raise RuntimeError(
                "update_distribution() called before a matching sample_population(). "
                f"Expected {self._lam} samples in _pop, got {len(self._pop)}. "
                "Call sample_population() first."
            )

        order    = np.argsort(rewards)[::-1]
        prev_best = self._champion_reward
        improved  = False

        best_r = float(rewards[order[0]])
        if best_r > self._champion_reward:
            self._champion_reward = best_r
            self._champion        = self._flat_to_individual(
                np.array(self._pop[order[0]], dtype=np.float32)
            )
            improved = True

        # Weighted mean recombination (top-mu elites)
        elite_xs   = np.stack([self._pop[order[i]] for i in range(self._mu)])
        self._mean = np.einsum("i,ij->j", self._recomb_w, elite_xs)

        # 1/5 success rule for step-size adaptation
        n_success    = sum(1 for r in rewards if float(r) > prev_best)
        success_rate = n_success / self._lam
        self._sigma  = float(np.clip(
            self._sigma * (1.2 if success_rate > 0.2 else 0.85),
            1e-6, 1e2,
        ))

        return improved

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self._champion is None:
            raise RuntimeError(
                "LSTMEvolutionPolicy: no champion yet — call sample_population() "
                "and update_distribution() for at least one generation first."
            )
        return self._champion(obs)

    def update(self, obs, action, reward, next_obs, done, **kwargs) -> None:
        pass

    def on_episode_start(self, **kwargs) -> None:
        if self._champion is not None:
            self._champion.on_episode_start(**kwargs)

    def on_episode_end(self) -> None:
        if self._champion is not None:
            self._champion.on_episode_end()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type":      "lstm",
            "hidden_size":      self._lstm_core_template._hidden_size,
            "head_param_count": self._head_param_count,
            "population_size":  self._lam,
            "sigma":            float(self._sigma),
            "obs_dim":          self._obs_dim,
            "champion_reward":  float(self._champion_reward),
        }

    def save(self, path: str) -> None:
        """Save the champion's flat parameter vector to a .npz file.

        Logs a warning and writes nothing if no champion exists yet (i.e.
        :meth:`sample_population` / :meth:`update_distribution` have not been
        called).
        """
        if self._champion is None:
            logger.warning(
                "[LSTMEvolutionPolicy] save() called but no champion yet — "
                "nothing written to %s", path,
            )
            return
        flat = self._champion.to_flat()
        np.savez(
            path,
            flat        = flat,
            flat_dim    = np.int64(self._flat_dim),
            hidden_size = np.int64(self._lstm_core_template._hidden_size),
            obs_dim     = np.int64(self._obs_dim),
        )

    def save_trainer_state(self, path: str) -> None:
        """Persist isotropic ES distribution state and champion to an .npz file.

        The champion flat vector and reward are included so that a loaded policy
        can call :meth:`__call__` immediately without running another generation.
        """
        arrays: dict = dict(
            mean            = self._mean,
            sigma           = np.float64(self._sigma),
            flat_dim        = np.int64(self._flat_dim),
            champion_reward = np.float64(self._champion_reward),
            has_champion    = np.bool_(self._champion is not None),
        )
        if self._champion is not None:
            arrays["champion_flat"] = self._champion.to_flat().astype(np.float32)
        np.savez(path, **arrays)
        logger.debug("[LSTMEvolutionPolicy] trainer state saved → %s", path)

    def load_trainer_state(self, path: str) -> None:
        """Restore ES distribution state (and champion, if saved) from an .npz file.

        Raises ValueError if the saved flat_dim does not match.
        """
        with np.load(path) as data:
            saved_flat_dim = int(data["flat_dim"])
            if saved_flat_dim != self._flat_dim:
                raise ValueError(
                    f"LSTMEvolutionPolicy: trainer state flat_dim mismatch — "
                    f"saved={saved_flat_dim}, current={self._flat_dim}. "
                    f"The network architecture (hidden_size or obs_dim) may have changed. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._mean  = data["mean"].astype(np.float64)
            self._sigma = float(data["sigma"])
            if "champion_reward" in data:
                self._champion_reward = float(data["champion_reward"])
            if (
                "has_champion" in data and bool(data["has_champion"])
                and "champion_flat" in data
            ):
                self._champion = self._flat_to_individual(
                    data["champion_flat"].astype(np.float32)
                )
        logger.info(
            "[LSTMEvolutionPolicy] trainer state loaded from %s (sigma=%.4f, champion=%s)",
            path, self._sigma,
            f"reward={self._champion_reward:.4f}" if self._champion is not None else "none",
        )
