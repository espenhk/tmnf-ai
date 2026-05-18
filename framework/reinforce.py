"""Generic REINFORCE (Monte Carlo Policy Gradient) policies.

REINFORCEPolicy        — single softmax head over a discrete action set;
                         parameterised on ObsSpec + action_decoder callable.
TwoHeadREINFORCEPolicy — shared trunk + fn-idx softmax head + spatial sigmoid
                         head; mirrors SC2REINFORCEPolicy from
                         games/sc2/sc2_policies.py.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from framework.obs_spec import ObsSpec
from framework.policies import BasePolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-head REINFORCE
# ---------------------------------------------------------------------------

class REINFORCEPolicy(BasePolicy):
    """REINFORCE (Monte Carlo Policy Gradient) with optional entropy regularisation.

    Action head: softmax over *output_dim* discrete actions.  Each episode
    accumulates (log_prob, reward) pairs; a gradient update fires on
    :meth:`on_episode_end` using discounted, normalised returns.

    Parameters
    ----------
    obs_spec :
        Observation spec.  Provides ``dim`` (input width) and ``scales``
        (per-feature normalisation).
    action_decoder :
        Callable ``(action_idx: int) -> np.ndarray`` that converts a sampled
        discrete-action index to the action array returned by :meth:`__call__`.
        Typically ``lambda i: DISCRETE_ACTIONS[i].copy()``.
    output_dim :
        Number of discrete actions (softmax output size).
    available_actions_fn :
        Optional callable ``(info: dict) -> np.ndarray[bool]`` returning a mask
        of shape ``(output_dim,)`` — ``True`` where the action is legal.
        Masked logits are set to ``-inf`` before softmax.
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        action_decoder: Callable[[int], np.ndarray],
        *,
        output_dim: int,
        hidden_sizes: list[int] = (64, 64),
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        baseline: str = "running_mean",
        available_actions_fn: Callable[[dict], np.ndarray] | None = None,
        seed: int | None = None,
    ) -> None:
        self._obs_spec      = obs_spec
        self._obs_dim       = obs_spec.dim
        self._scales        = obs_spec.scales
        self._action_decoder = action_decoder
        self._output_dim    = int(output_dim)
        self._avail_fn      = available_actions_fn

        self._hidden        = list(hidden_sizes)
        self._lr            = float(learning_rate)
        self._gamma         = float(gamma)
        self._entropy_coeff = float(entropy_coeff)
        self._baseline_type = baseline

        self._weights, self._biases = self._build_net(seed)

        # Per-episode trajectory storage
        self._ep_grads: list[tuple]   = []
        self._ep_rewards: list[float] = []

        # Running-mean baseline (EMA of total episode returns)
        self._baseline_val   = 0.0
        self._baseline_alpha = 0.05

        # Cache of latest info dict for available_actions_fn
        self._cached_info: dict = {}

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_net(
        self, seed: int | None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        rng  = np.random.default_rng(seed)
        dims = [self._obs_dim] + self._hidden + [self._output_dim]
        weights, biases = [], []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            w = rng.standard_normal((dims[i + 1], fan_in)).astype(np.float32)
            w *= np.sqrt(2.0 / fan_in)
            b = np.zeros(dims[i + 1], dtype=np.float32)
            weights.append(w)
            biases.append(b)
        return weights, biases

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_s = z - z.max()
        e   = np.exp(z_s)
        return e / e.sum()

    def _forward(self, obs_norm: np.ndarray):
        """Forward pass; caches (layer_inputs, pre_relu) for backprop."""
        x: np.ndarray      = obs_norm.astype(np.float32)
        layer_inputs: list = []
        pre_relu: list     = []
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            layer_inputs.append(x.copy())
            z = w @ x + b
            if i < len(self._weights) - 1:
                pre_relu.append(z.copy())
                x = np.maximum(0.0, z)
            else:
                logits = z
        probs = self._softmax(logits)
        return probs, layer_inputs, pre_relu

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def _get_available_mask(self) -> np.ndarray | None:
        if self._avail_fn is None:
            return None
        return self._avail_fn(self._cached_info)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_norm              = obs / self._scales
        probs, l_in, pre_r   = self._forward(obs_norm)

        mask = self._get_available_mask()
        if mask is not None:
            logits_masked = np.log(probs + 1e-30)
            logits_masked[~mask] = -np.inf
            probs = self._softmax(logits_masked)

        action_idx = int(np.random.choice(self._output_dim, p=probs))
        self._ep_grads.append((l_in, pre_r, probs.copy(), action_idx))
        return self._action_decoder(action_idx)

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray | int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs,
    ) -> None:
        self._ep_rewards.append(float(reward))
        self._cached_info = kwargs.get("info") or {}

    def on_episode_end(self) -> None:
        T = min(len(self._ep_grads), len(self._ep_rewards))
        if T == 0:
            self._ep_grads.clear()
            self._ep_rewards.clear()
            return

        # Discounted returns
        G = np.zeros(T, dtype=np.float64)
        running = 0.0
        for t in reversed(range(T)):
            running = self._ep_rewards[t] + self._gamma * running
            G[t]    = running

        # Capture baseline before updating
        baseline_for_advantages = self._baseline_val

        # Baseline update (EMA of total episode return)
        if self._baseline_type == "running_mean":
            self._baseline_val = (
                (1 - self._baseline_alpha) * self._baseline_val
                + self._baseline_alpha * float(G[0])
            )

        # Normalise returns
        G_std = float(G.std())
        if G_std > 1e-6:
            G_norm = (G - G.mean()) / (G_std + 1e-8)
        else:
            G_norm = G - baseline_for_advantages

        # Accumulate gradients
        dW = [np.zeros_like(w, dtype=np.float64) for w in self._weights]
        dB = [np.zeros_like(b, dtype=np.float64) for b in self._biases]

        for t in range(T):
            l_in, pre_r, probs, a_idx = self._ep_grads[t]
            advantage = float(G_norm[t])

            # Policy gradient: ∂J/∂z = advantage × (one_hot(a) − probs)
            delta           = -probs.copy().astype(np.float64)
            delta[a_idx]   += 1.0
            delta          *= advantage

            if self._entropy_coeff > 0.0:
                log_p        = np.log(probs.astype(np.float64) + 1e-8)
                H            = -float(np.dot(probs, log_p))
                entropy_grad = -probs.astype(np.float64) * (log_p + H)
                delta       += self._entropy_coeff * entropy_grad

            # Backprop through MLP (gradient ascent)
            g = delta
            for i in range(len(self._weights) - 1, -1, -1):
                dW[i] += np.outer(g, l_in[i])
                dB[i] += g
                if i > 0:
                    g = self._weights[i].T @ g * (pre_r[i - 1] > 0)

        # Parameter update (gradient ascent: θ += lr * grad / T)
        lr_t = self._lr / T
        for i in range(len(self._weights)):
            self._weights[i] += (lr_t * dW[i]).astype(np.float32)
            self._biases[i]  += (lr_t * dB[i]).astype(np.float32)

        self._ep_grads.clear()
        self._ep_rewards.clear()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type":    "reinforce",
            "hidden_sizes":   self._hidden,
            "output_dim":     self._output_dim,
            "learning_rate":  float(self._lr),
            "gamma":          float(self._gamma),
            "entropy_coeff":  float(self._entropy_coeff),
            "baseline":       self._baseline_type,
            "obs_dim":        self._obs_dim,
            "baseline_value": float(self._baseline_val),
            "weights":        [w.tolist() for w in self._weights],
            "biases":         [b.tolist() for b in self._biases],
        }

    @classmethod
    def from_cfg(
        cls,
        cfg: dict,
        obs_spec: ObsSpec,
        action_decoder: Callable[[int], np.ndarray],
    ) -> "REINFORCEPolicy":
        obj = cls(
            obs_spec      = obs_spec,
            action_decoder = action_decoder,
            output_dim    = cfg.get("output_dim", len(cfg.get("weights", [[]])[0]) if cfg.get("weights") else 9),
            hidden_sizes  = cfg.get("hidden_sizes",  [64, 64]),
            learning_rate = cfg.get("learning_rate", 0.001),
            gamma         = cfg.get("gamma",         0.99),
            entropy_coeff = cfg.get("entropy_coeff", 0.01),
            baseline      = cfg.get("baseline",      "running_mean"),
        )
        if "weights" in cfg:
            obj._weights = [np.array(w, dtype=np.float32) for w in cfg["weights"]]
            obj._biases  = [np.array(b, dtype=np.float32) for b in cfg["biases"]]
        if "baseline_value" in cfg:
            obj._baseline_val = float(cfg["baseline_value"])
        return obj

    def save_trainer_state(self, path: str) -> None:
        """Persist REINFORCE running-mean baseline and obs_dim to an .npz file."""
        np.savez(
            path,
            baseline_val = np.float64(self._baseline_val),
            obs_dim      = np.int64(self._obs_dim),
        )
        logger.debug("[REINFORCEPolicy] trainer state saved → %s", path)

    def load_trainer_state(self, path: str) -> None:
        """Restore REINFORCE baseline from an .npz file.

        Raises ValueError if the saved obs_dim does not match.
        """
        with np.load(path) as data:
            saved_obs_dim = int(data["obs_dim"])
            if saved_obs_dim != self._obs_dim:
                raise ValueError(
                    f"REINFORCEPolicy: trainer state obs_dim mismatch — "
                    f"saved={saved_obs_dim}, current={self._obs_dim}. "
                    f"The observation space may have changed. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._baseline_val = float(data["baseline_val"])
        logger.info(
            "[REINFORCEPolicy] trainer state loaded from %s (baseline=%.4f)",
            path, self._baseline_val,
        )


# ---------------------------------------------------------------------------
# Two-head REINFORCE (fn-idx softmax + spatial sigmoid) — mirrors SC2
# ---------------------------------------------------------------------------

def _sigmoid_scalar(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-np.clip(x, -20.0, 20.0))))


class TwoHeadREINFORCEPolicy(BasePolicy):
    """REINFORCE with a shared trunk + discrete fn-head + continuous spatial head.

    Architecture mirrors :class:`games.sc2.sc2_policies.SC2REINFORCEPolicy`:

    * **Shared trunk** — hidden FC layers with ReLU; produces latent ``h_last``.
    * **fn_head** — ``fn_dim`` logits, softmax → sampled ``fn_idx``; unavailable
      function IDs are masked to ``-inf`` before sampling when
      *available_actions_fn* is supplied.
    * **spatial_head** — ``spatial_dim`` logits, sigmoid → continuous coordinates
      in ``[0, 1]^spatial_dim``.

    Parameters
    ----------
    obs_spec :
        Observation spec.
    action_decoder :
        Callable ``(fn_idx: int, sp: np.ndarray) -> np.ndarray`` that converts
        a sampled (fn_idx, spatial vector) pair to the final action array.
    fn_dim :
        Number of discrete function choices (fn_head softmax size).
    spatial_dim :
        Number of sigmoid spatial outputs (typically 2 for (x, y)).
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        action_decoder: Callable[[int, np.ndarray], np.ndarray],
        *,
        fn_dim: int,
        spatial_dim: int,
        hidden_sizes: list[int] = (128, 64),
        learning_rate: float = 0.0003,
        gamma: float = 0.995,
        entropy_coeff: float = 0.05,
        baseline: str = "running_mean",
        available_actions_fn: Callable[[dict], np.ndarray] | None = None,
        seed: int | None = None,
    ) -> None:
        self._obs_spec       = obs_spec
        self._obs_dim        = obs_spec.dim
        self._scales         = obs_spec.scales
        self._action_decoder = action_decoder
        self._fn_dim         = int(fn_dim)
        self._spatial_dim    = int(spatial_dim)
        self._avail_fn       = available_actions_fn

        self._hidden        = list(hidden_sizes)
        self._lr            = float(learning_rate)
        self._gamma         = float(gamma)
        self._entropy_coeff = float(entropy_coeff)
        self._baseline_type = baseline
        self._rng           = np.random.default_rng(seed)

        (
            self._trunk_w,
            self._trunk_b,
            self._fn_w,
            self._fn_b,
            self._sp_w,
            self._sp_b,
        ) = self._build_net(seed)

        self._ep_grads: list[tuple]   = []
        self._ep_rewards: list[float] = []
        self._baseline_val   = 0.0
        self._baseline_alpha = 0.05
        self._available_fn_ids: set[int] | None = None
        self._cached_info: dict = {}

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_net(self, seed):
        rng = np.random.default_rng(seed)

        trunk_w: list[np.ndarray] = []
        trunk_b: list[np.ndarray] = []
        dims = [self._obs_dim] + self._hidden
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            w = rng.standard_normal((dims[i + 1], fan_in)).astype(np.float32)
            w *= np.sqrt(2.0 / fan_in)
            b = np.zeros(dims[i + 1], dtype=np.float32)
            trunk_w.append(w)
            trunk_b.append(b)

        h_dim = self._hidden[-1] if self._hidden else self._obs_dim

        fn_w = rng.standard_normal((self._fn_dim, h_dim)).astype(np.float32)
        fn_w *= np.sqrt(2.0 / h_dim)
        fn_b = np.zeros(self._fn_dim, dtype=np.float32)

        sp_w = rng.standard_normal((self._spatial_dim, h_dim)).astype(np.float32)
        sp_w *= np.sqrt(2.0 / h_dim)
        sp_b = np.zeros(self._spatial_dim, dtype=np.float32)

        return trunk_w, trunk_b, fn_w, fn_b, sp_w, sp_b

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_s = z - z.max()
        e   = np.exp(z_s)
        return e / e.sum()

    def _trunk_forward(self, obs_norm: np.ndarray):
        x: np.ndarray      = obs_norm.astype(np.float32)
        layer_inputs: list = []
        pre_relu: list     = []
        for w, b in zip(self._trunk_w, self._trunk_b):
            layer_inputs.append(x.copy())
            z = w @ x + b
            pre_relu.append(z.copy())
            x = np.maximum(0.0, z)
        return x, layer_inputs, pre_relu

    def _build_fn_mask(self, available_fn_ids: set[int] | None) -> np.ndarray:
        mask = np.ones(self._fn_dim, dtype=bool)
        if available_fn_ids is not None:
            for i in range(self._fn_dim):
                mask[i] = i in available_fn_ids
        if not mask.any():
            mask[0] = True
        return mask

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def on_episode_start(self, **kwargs) -> None:
        self._ep_grads.clear()
        self._ep_rewards.clear()
        info = kwargs.get("info") or {}
        available = info.get("available_fn_ids")
        self._available_fn_ids = set(available) if available is not None else None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_norm = obs / self._scales
        h_last, l_in, pre_r = self._trunk_forward(obs_norm)

        # fn head
        fn_logits = self._fn_w @ h_last + self._fn_b
        fn_mask   = self._build_fn_mask(self._available_fn_ids)
        fn_logits_m = fn_logits.copy()
        fn_logits_m[~fn_mask] = -np.inf
        fn_probs  = self._softmax(fn_logits_m)
        fn_idx    = int(self._rng.choice(self._fn_dim, p=fn_probs))

        # spatial head (sigmoid continuous)
        sp_logits = self._sp_w @ h_last + self._sp_b
        sp_sig    = np.array(
            [_sigmoid_scalar(float(sp_logits[i])) for i in range(self._spatial_dim)],
            dtype=np.float32,
        )

        self._ep_grads.append((l_in, pre_r, h_last.copy(),
                               fn_probs.copy(), fn_idx, sp_sig.copy(), fn_mask.copy()))

        return self._action_decoder(fn_idx, sp_sig)

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray | int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        **kwargs,
    ) -> None:
        self._ep_rewards.append(float(reward))
        info = kwargs.get("info") or {}
        available = info.get("available_fn_ids")
        if available is not None:
            self._available_fn_ids = set(available)

    def on_episode_end(self) -> None:
        T = min(len(self._ep_grads), len(self._ep_rewards))
        if T == 0:
            self._ep_grads.clear()
            self._ep_rewards.clear()
            return

        G = np.zeros(T, dtype=np.float64)
        running = 0.0
        for t in reversed(range(T)):
            running = self._ep_rewards[t] + self._gamma * running
            G[t]    = running

        baseline_for_advantages = self._baseline_val
        if self._baseline_type == "running_mean":
            self._baseline_val = (
                (1 - self._baseline_alpha) * self._baseline_val
                + self._baseline_alpha * float(G[0])
            )

        G_std = float(G.std())
        if G_std > 1e-6:
            G_norm = (G - G.mean()) / (G_std + 1e-8)
        else:
            G_norm = G - baseline_for_advantages

        dW_trunk = [np.zeros_like(w, dtype=np.float64) for w in self._trunk_w]
        dB_trunk = [np.zeros_like(b, dtype=np.float64) for b in self._trunk_b]
        dW_fn    = np.zeros_like(self._fn_w, dtype=np.float64)
        dB_fn    = np.zeros_like(self._fn_b, dtype=np.float64)
        dW_sp    = np.zeros_like(self._sp_w, dtype=np.float64)
        dB_sp    = np.zeros_like(self._sp_b, dtype=np.float64)

        for t in range(T):
            l_in, pre_r, h_last, fn_probs, fn_idx, sp_sig, fn_mask = self._ep_grads[t]
            advantage = float(G_norm[t])
            sp_sig_d  = sp_sig.astype(np.float64)
            h_last_d  = h_last.astype(np.float64)

            # fn head gradient
            delta_fn            = -fn_probs.copy().astype(np.float64)
            delta_fn[fn_idx]   += 1.0
            delta_fn[~fn_mask]  = 0.0
            delta_fn           *= advantage

            if self._entropy_coeff > 0.0:
                log_p_fn  = np.log(fn_probs.astype(np.float64) + 1e-8)
                H_fn      = -float(np.dot(fn_probs[fn_mask], log_p_fn[fn_mask]))
                ent_grad  = np.zeros(self._fn_dim, dtype=np.float64)
                ent_grad[fn_mask] = -(
                    fn_probs[fn_mask].astype(np.float64) * (log_p_fn[fn_mask] + H_fn)
                )
                delta_fn += self._entropy_coeff * ent_grad

            # spatial head gradient (deterministic policy, sigmoid derivative)
            delta_sp = advantage * (sp_sig_d * (1.0 - sp_sig_d))

            dW_fn += np.outer(delta_fn, h_last_d)
            dB_fn += delta_fn
            dW_sp += np.outer(delta_sp, h_last_d)
            dB_sp += delta_sp

            if self._trunk_w:
                g_trunk = (
                    self._fn_w.T.astype(np.float64) @ delta_fn
                    + self._sp_w.T.astype(np.float64) @ delta_sp
                )
                n_trunk = len(self._trunk_w)
                for i in range(n_trunk - 1, -1, -1):
                    g_trunk = g_trunk * (pre_r[i] > 0).astype(np.float64)
                    dW_trunk[i] += np.outer(g_trunk, l_in[i].astype(np.float64))
                    dB_trunk[i] += g_trunk
                    if i > 0:
                        g_trunk = self._trunk_w[i].T.astype(np.float64) @ g_trunk

        lr_t = self._lr / T
        for i in range(len(self._trunk_w)):
            self._trunk_w[i] += (lr_t * dW_trunk[i]).astype(np.float32)
            self._trunk_b[i] += (lr_t * dB_trunk[i]).astype(np.float32)
        self._fn_w += (lr_t * dW_fn).astype(np.float32)
        self._fn_b += (lr_t * dB_fn).astype(np.float32)
        self._sp_w += (lr_t * dW_sp).astype(np.float32)
        self._sp_b += (lr_t * dB_sp).astype(np.float32)

        self._ep_grads.clear()
        self._ep_rewards.clear()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_cfg(self) -> dict:
        return {
            "policy_type":   "two_head_reinforce",
            "hidden_sizes":  self._hidden,
            "fn_dim":        self._fn_dim,
            "spatial_dim":   self._spatial_dim,
            "learning_rate": float(self._lr),
            "gamma":         float(self._gamma),
            "entropy_coeff": float(self._entropy_coeff),
            "baseline":      self._baseline_type,
            "obs_dim":       self._obs_dim,
            "baseline_value": float(self._baseline_val),
            "trunk_weights": [w.tolist() for w in self._trunk_w],
            "trunk_biases":  [b.tolist() for b in self._trunk_b],
            "fn_weights":    self._fn_w.tolist(),
            "fn_biases":     self._fn_b.tolist(),
            "sp_weights":    self._sp_w.tolist(),
            "sp_biases":     self._sp_b.tolist(),
        }

    @classmethod
    def from_cfg(
        cls,
        cfg: dict,
        obs_spec: ObsSpec,
        action_decoder: Callable[[int, np.ndarray], np.ndarray],
    ) -> "TwoHeadREINFORCEPolicy":
        obj = cls(
            obs_spec       = obs_spec,
            action_decoder = action_decoder,
            fn_dim         = cfg.get("fn_dim",        6),
            spatial_dim    = cfg.get("spatial_dim",   2),
            hidden_sizes   = cfg.get("hidden_sizes",  [128, 64]),
            learning_rate  = cfg.get("learning_rate", 0.0003),
            gamma          = cfg.get("gamma",         0.995),
            entropy_coeff  = cfg.get("entropy_coeff", 0.05),
            baseline       = cfg.get("baseline",      "running_mean"),
        )
        if "trunk_weights" in cfg:
            obj._trunk_w = [np.array(w, dtype=np.float32) for w in cfg["trunk_weights"]]
            obj._trunk_b = [np.array(b, dtype=np.float32) for b in cfg["trunk_biases"]]
        if "fn_weights" in cfg:
            obj._fn_w = np.array(cfg["fn_weights"], dtype=np.float32)
            obj._fn_b = np.array(cfg["fn_biases"],  dtype=np.float32)
        if "sp_weights" in cfg:
            obj._sp_w = np.array(cfg["sp_weights"], dtype=np.float32)
            obj._sp_b = np.array(cfg["sp_biases"],  dtype=np.float32)
        if "baseline_value" in cfg:
            obj._baseline_val = float(cfg["baseline_value"])
        return obj

    def save_trainer_state(self, path: str) -> None:
        np.savez(
            path,
            baseline_val = np.float64(self._baseline_val),
            obs_dim      = np.int64(self._obs_dim),
        )
        logger.debug("[TwoHeadREINFORCEPolicy] trainer state saved → %s", path)

    def load_trainer_state(self, path: str) -> None:
        with np.load(path) as data:
            saved_obs_dim = int(data["obs_dim"])
            if saved_obs_dim != self._obs_dim:
                raise ValueError(
                    f"TwoHeadREINFORCEPolicy: trainer state obs_dim mismatch — "
                    f"saved={saved_obs_dim}, current={self._obs_dim}. "
                    f"Use --re-initialize to restart from scratch."
                )
            self._baseline_val = float(data["baseline_val"])
        logger.info(
            "[TwoHeadREINFORCEPolicy] trainer state loaded from %s (baseline=%.4f)",
            path, self._baseline_val,
        )
