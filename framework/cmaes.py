"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

CMAESDistribution — pure (μ/μ_w, λ)-CMA-ES math (Hansen 2016).  Operates
                    only on θ ∈ R^n; no policy coupling.
CMAESPolicy       — wraps CMAESDistribution with a game-injected
                    parameter_decoder that converts flat θ → BasePolicy.

The algorithm constants are byte-identical to those in:
  games/tmnf/policies.py:712-722
  games/sc2/policies.py:947-960
  games/sc2/sc2_policies.py:1054-1062
"""
from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np

from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CMAESDistribution — pure math
# ---------------------------------------------------------------------------

class CMAESDistribution:
    """Pure (μ/μ_w, λ)-CMA-ES distribution (Hansen 2016).

    Operates entirely on θ ∈ R^n.  Has no concept of policies or games;
    produces and consumes raw numpy vectors.

    Parameters
    ----------
    n :
        Parameter dimension.
    lam :
        Population size λ (offspring per generation).
    sigma :
        Initial step size σ.
    mean :
        Optional initial mean (zeros by default).
    seed :
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        n: int,
        lam: int,
        sigma: float,
        mean: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        self._n   = int(n)
        self._lam = int(lam)

        mu            = self._lam // 2
        self._mu      = mu
        raw_w         = np.array(
            [np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)],
            dtype=np.float64,
        )
        self._weights = raw_w / raw_w.sum()
        self._mu_eff  = 1.0 / float(np.sum(self._weights ** 2))

        # Step-size adaptation constants (Hansen 2016, §3)
        self._cs   = (self._mu_eff + 2) / (n + self._mu_eff + 5)
        self._ds   = (
            1 + 2 * max(0.0, float(np.sqrt((self._mu_eff - 1) / (n + 1))) - 1)
            + self._cs
        )
        self._chin = float(np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2)))

        # Covariance adaptation constants (Hansen 2016, §3)
        self._cc  = (4 + self._mu_eff / n) / (n + 4 + 2 * self._mu_eff / n)
        self._c1  = 2.0 / ((n + 1.3) ** 2 + self._mu_eff)
        self._cmu = min(
            1.0 - self._c1,
            2.0 * (self._mu_eff - 2 + 1.0 / self._mu_eff) / ((n + 2) ** 2 + self._mu_eff),
        )

        self._rng   = np.random.default_rng(seed)
        self._mean  = (
            mean.copy().astype(np.float64)
            if mean is not None
            else np.zeros(n, dtype=np.float64)
        )
        self._sigma = float(sigma)

        self._ps       = np.zeros(n, dtype=np.float64)
        self._pc       = np.zeros(n, dtype=np.float64)
        self._C        = np.eye(n, dtype=np.float64)
        self._B        = np.eye(n, dtype=np.float64)
        self._D        = np.ones(n, dtype=np.float64)
        self._invsqrtC = np.eye(n, dtype=np.float64)
        self._eigengen = 0
        self._gen      = 0

        self._pop_xs: list[np.ndarray] = []
        self._pop_ys: list[np.ndarray] = []

        # Running best-fitness tracker used by update() to compute the improved flag.
        # Initialised here (not lazily) so save/load and standalone use are consistent.
        self._best_fitness: float = float("-inf")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return self._n

    @property
    def lam(self) -> int:
        return self._lam

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def gen(self) -> int:
        return self._gen

    # ------------------------------------------------------------------
    # Eigen decomposition (lazy, per-generation)
    # ------------------------------------------------------------------

    def _update_eigen(self) -> None:
        self._C         = np.triu(self._C) + np.triu(self._C, 1).T
        eigvals, self._B = np.linalg.eigh(self._C)
        eigvals          = np.maximum(eigvals, 1e-20)
        self._D          = np.sqrt(eigvals)
        self._invsqrtC   = self._B @ np.diag(1.0 / self._D) @ self._B.T
        self._eigengen   = self._gen

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def sample(self) -> list[np.ndarray]:
        """Draw λ offspring from N(mean, σ² · C).

        Returns a list of λ float64 vectors of length *n*.
        """
        n = self._n
        if self._gen - self._eigengen >= max(1, self._lam // max(1, 10 * n)):
            self._update_eigen()

        self._pop_xs = []
        self._pop_ys = []
        for _ in range(self._lam):
            z = self._rng.standard_normal(n)
            y = self._B @ (self._D * z)
            x = self._mean + self._sigma * y
            self._pop_xs.append(x)
            self._pop_ys.append(y)

        return [x.copy() for x in self._pop_xs]

    def update(
        self,
        fitnesses: np.ndarray,
    ) -> bool:
        """Apply (μ/μ_w, λ)-CMA-ES update.

        Parameters
        ----------
        fitnesses :
            Float array of length *λ* corresponding to the candidates returned
            by the last :meth:`sample` call (in the same order).
            Higher = better.

        Returns
        -------
        bool
            ``True`` if the best fitness this generation is a new all-time best
            (tracked internally via :attr:`_best_fitness`).
        """
        if len(fitnesses) != self._lam:
            raise ValueError(f"Expected {self._lam} fitnesses, got {len(fitnesses)}")
        if len(self._pop_xs) != self._lam or len(self._pop_ys) != self._lam:
            raise RuntimeError(
                "update() called before a matching sample(). "
                f"Expected {self._lam} samples, got {len(self._pop_xs)}. "
                "Call sample() first."
            )
        n = self._n

        order = np.argsort(fitnesses)[::-1]

        # Champion tracking
        improved = False
        best_f = float(fitnesses[order[0]])
        if best_f > self._best_fitness:
            self._best_fitness = best_f
            improved = True

        elite_ys = np.stack([self._pop_ys[order[i]] for i in range(self._mu)])
        step     = np.einsum("i,ij->j", self._weights, elite_ys)

        self._mean = self._mean + self._sigma * step

        ps_scale  = float(np.sqrt(self._cs * (2 - self._cs) * self._mu_eff))
        self._ps  = (1 - self._cs) * self._ps + ps_scale * (self._invsqrtC @ step)

        ps_norm     = float(np.linalg.norm(self._ps))
        self._sigma = float(np.clip(
            self._sigma * np.exp((self._cs / self._ds) * (ps_norm / self._chin - 1)),
            1e-10, 1e6,
        ))

        ps_norm_normed = ps_norm / float(
            np.sqrt(1 - (1 - self._cs) ** (2 * (self._gen + 1)))
        )
        h_sigma = 1.0 if ps_norm_normed < (1.4 + 2.0 / (n + 1)) * self._chin else 0.0

        pc_scale = float(np.sqrt(self._cc * (2 - self._cc) * self._mu_eff))
        self._pc = (1 - self._cc) * self._pc + h_sigma * pc_scale * step

        delta_h = (1 - h_sigma) * self._cc * (2 - self._cc)
        rank1   = np.outer(self._pc, self._pc)
        rank_mu = np.einsum("i,ij,ik->jk", self._weights, elite_ys, elite_ys)
        self._C = (
            (1 - self._c1 - self._cmu) * self._C
            + self._c1 * (rank1 + delta_h * self._C)
            + self._cmu * rank_mu
        )

        self._gen += 1
        return improved

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Persist full CMA-ES distribution state to an .npz file."""
        np.savez(
            path,
            mean         = self._mean,
            sigma        = np.float64(self._sigma),
            C            = self._C,
            B            = self._B,
            D            = self._D,
            invsqrtC     = self._invsqrtC,
            ps           = self._ps,
            pc           = self._pc,
            gen          = np.int64(self._gen),
            n            = np.int64(self._n),
            best_fitness = np.float64(self._best_fitness),
        )
        logger.debug("[CMAESDistribution] state saved → %s", path)

    def load_state(self, path: str) -> None:
        """Restore CMA-ES distribution state from an .npz file.

        Raises ValueError if the saved dimension does not match *n*.
        """
        with np.load(path) as data:
            n_saved = int(data["n"])
            if n_saved != self._n:
                raise ValueError(
                    f"CMAESDistribution: state dimension mismatch — "
                    f"saved n={n_saved}, current n={self._n}."
                )
            self._mean     = data["mean"].astype(np.float64)
            self._sigma    = float(data["sigma"])
            self._C        = data["C"].astype(np.float64)
            self._B        = data["B"].astype(np.float64)
            self._D        = data["D"].astype(np.float64)
            self._invsqrtC = data["invsqrtC"].astype(np.float64)
            self._ps       = data["ps"].astype(np.float64)
            self._pc       = data["pc"].astype(np.float64)
            self._gen      = int(data["gen"])
            if "best_fitness" in data:
                self._best_fitness = float(data["best_fitness"])
        logger.info(
            "[CMAESDistribution] state loaded from %s (gen=%d, sigma=%.4f)",
            path, self._gen, self._sigma,
        )


# ---------------------------------------------------------------------------
# CMAESPolicy — wraps CMAESDistribution + parameter_decoder
# ---------------------------------------------------------------------------

class CMAESPolicy(BasePolicy):
    """CMA-ES policy wrapper.

    Internally holds a :class:`CMAESDistribution` over θ ∈ R^{flat_dim} and
    calls *parameter_decoder* to convert sampled θ vectors into callable
    :class:`~framework.policies.BasePolicy` instances.

    Parameters
    ----------
    obs_spec :
        Observation spec (kept for metadata / serialisation context; the
        algorithm itself operates on flat vectors).
    parameter_decoder :
        Callable ``(θ: np.ndarray) -> BasePolicy`` that converts a flat
        parameter vector into a policy instance suitable for evaluation.
        For TMNF this would build a ``WeightedLinearPolicy`` from the steer /
        accel / brake weight blocks; for SC2 it builds an
        ``SC2MultiHeadLinearPolicy``.
    flat_dim :
        Dimension of the flat parameter vector θ.
    population_size :
        λ — offspring sampled per generation.
    initial_sigma :
        Starting step size σ.
    eval_episodes :
        Episodes per individual per generation (averaged for fitness).
    seed :
        Optional RNG seed.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        parameter_decoder: Callable[[np.ndarray], BasePolicy],
        flat_dim: int,
        *,
        population_size: int = 20,
        initial_sigma: float = 0.3,
        eval_episodes: int = 1,
        seed: int | None = None,
    ) -> None:
        self._obs_spec         = obs_spec
        self._parameter_decoder = parameter_decoder
        self._flat_dim         = int(flat_dim)
        self._eval_episodes    = max(1, int(eval_episodes))

        self._dist = CMAESDistribution(
            n     = self._flat_dim,
            lam   = int(population_size),
            sigma = float(initial_sigma),
            seed  = seed,
        )

        self._champion: BasePolicy | None = None
        self._champion_reward: float      = float("-inf")

    # ------------------------------------------------------------------
    # Delegate distribution attributes for backward compat
    # ------------------------------------------------------------------

    @property
    def _lam(self) -> int:
        return self._dist._lam

    @property
    def _mu(self) -> int:
        return self._dist._mu

    @property
    def _weights(self) -> np.ndarray:
        return self._dist._weights

    @property
    def _mu_eff(self) -> float:
        return self._dist._mu_eff

    @property
    def _n(self) -> int:
        return self._dist._n

    @property
    def _mean(self) -> np.ndarray:
        return self._dist._mean

    @_mean.setter
    def _mean(self, v: np.ndarray) -> None:
        self._dist._mean = v

    @property
    def _sigma(self) -> float:
        return self._dist._sigma

    @property
    def _C(self) -> np.ndarray:
        return self._dist._C

    @property
    def _B(self) -> np.ndarray:
        return self._dist._B

    @property
    def _D(self) -> np.ndarray:
        return self._dist._D

    @property
    def _invsqrtC(self) -> np.ndarray:
        return self._dist._invsqrtC

    @property
    def _ps(self) -> np.ndarray:
        return self._dist._ps

    @property
    def _pc(self) -> np.ndarray:
        return self._dist._pc

    @property
    def _gen(self) -> int:
        return self._dist._gen

    @property
    def _pop_xs(self) -> list:
        return self._dist._pop_xs

    @property
    def _pop_ys(self) -> list:
        return self._dist._pop_ys

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def champion_reward(self) -> float:
        return self._champion_reward

    @property
    def population_size(self) -> int:
        return self._dist.lam

    @property
    def sigma(self) -> float:
        return self._dist.sigma

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_random(self) -> None:
        """Initialise the search mean at zero."""
        self._dist._mean = np.zeros(self._flat_dim, dtype=np.float64)
        logger.info("[CMAESPolicy] initialised with zero mean, sigma=%.3f", self._dist._sigma)

    def initialize_from_champion(self, champion) -> None:
        """Seed the search mean from an existing champion's flat weight vector."""
        self._champion = champion

        seeded_reward = None
        for attr_name in ("champion_reward", "reward"):
            reward_value = getattr(champion, attr_name, None)
            if reward_value is not None:
                try:
                    seeded_reward = float(reward_value)
                except (TypeError, ValueError):
                    seeded_reward = None
                else:
                    if math.isfinite(seeded_reward):
                        break
                    seeded_reward = None

        if seeded_reward is None and math.isfinite(self._champion_reward):
            seeded_reward = float(self._champion_reward)

        self._champion_reward = seeded_reward if seeded_reward is not None else float("-inf")
        self._dist._mean = champion.to_flat().astype(np.float64)
        logger.info(
            "[CMAESPolicy] seeded mean from champion%s",
            "" if seeded_reward is None else f" (baseline reward={self._champion_reward:.6f})",
        )

    # ------------------------------------------------------------------
    # Population-loop interface (used by _greedy_loop_cmaes)
    # ------------------------------------------------------------------

    def sample_population(self) -> list[BasePolicy]:
        """Sample λ offspring from N(mean, σ² · C) and decode to policies."""
        thetas = self._dist.sample()
        return [self._parameter_decoder(theta) for theta in thetas]

    def update_distribution(self, rewards: list[float]) -> bool:
        """Apply (μ/μ_w, λ)-CMA-ES update.  Returns True if champion improved."""
        if len(rewards) != self._dist._lam:
            raise ValueError(f"Expected {self._dist._lam} rewards, got {len(rewards)}")
        if len(self._dist._pop_xs) != self._dist._lam:
            raise RuntimeError(
                "update_distribution() called before a matching sample_population(). "
                f"Expected {self._dist._lam} samples in _pop_xs/_pop_ys, "
                f"got {len(self._dist._pop_xs)}/{len(self._dist._pop_ys)}. "
                "Call sample_population() first."
            )

        order = np.argsort(rewards)[::-1]

        improved = False
        best_r   = float(rewards[order[0]])
        if best_r > self._champion_reward:
            self._champion_reward = best_r
            self._champion        = self._parameter_decoder(
                self._dist._pop_xs[order[0]]
            )
            improved = True

        fitnesses = np.asarray(rewards, dtype=np.float64)
        # Delegate the pure-math update to the distribution.  The distribution
        # tracks its own _best_fitness for the `improved` return value; we ignore
        # that flag here because CMAESPolicy maintains its own champion separately.
        self._dist.update(fitnesses)

        return improved

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self._champion is None:
            raise RuntimeError(
                "CMAESPolicy: no champion yet — call sample_population() "
                "and update_distribution() for at least one generation first."
            )
        return self._champion(obs)

    def update(self, obs, action, reward, next_obs, done, **kwargs) -> None:
        pass  # training is generation-based, not step-based

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type":     "cmaes",
            "population_size": self._dist._lam,
            "sigma":           float(self._dist._sigma),
            "flat_dim":        self._flat_dim,
            "eval_episodes":   self._eval_episodes,
            "champion_reward": float(self._champion_reward),
        }

    def save(self, path: str) -> None:
        """Save champion if it has a save() method (e.g. WeightedLinearPolicy)."""
        if self._champion is not None and hasattr(self._champion, "save"):
            self._champion.save(path)

    def save_trainer_state(self, path: str) -> None:
        """Persist full CMA-ES distribution state to an .npz file."""
        self._dist.save_state(path)
        logger.debug("[CMAESPolicy] trainer state saved → %s", path)

    def load_trainer_state(self, path: str) -> None:
        """Restore CMA-ES distribution state from an .npz file.

        Raises ValueError if the saved dimension does not match the current
        flat_dim.
        """
        try:
            self._dist.load_state(path)
        except ValueError as exc:
            raise ValueError(
                str(exc).replace(
                    "CMAESDistribution: state dimension mismatch",
                    "CMAESPolicy: trainer state dimension mismatch",
                )
                + " Use --re-initialize to restart from scratch."
            ) from exc
        logger.info(
            "[CMAESPolicy] trainer state loaded from %s (gen=%d, sigma=%.4f)",
            path, self._dist._gen, self._dist._sigma,
        )
