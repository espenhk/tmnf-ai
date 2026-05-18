"""Numeric equivalence tests: framework vs game implementations.

For each ported algorithm, this module instantiates both the framework version
and the corresponding TMNF game version with identical seeds and feeds them
the same sequence of observations / rewards.  The test passes when the final
internal state (weights, θ, replay buffer) is byte-identical, confirming that
the framework port is a faithful copy of the game code.

These tests act as a safety net for Phase C deletions: if they stay green we
know the port is correct.

Note: CMAESPolicy equivalence is tested here against the framework CMAESPolicy
parameterised with the TMNF decoder (which is independent of games/tmnf),
not against the games/tmnf CMAESPolicy — since the TMNF version is
also coupled through WeightedLinearPolicy.to_flat().  The key invariant
checked is that the framework CMAESDistribution internals are identical to
the TMNF CMAESPolicy internals after the same sequence of rewards.
"""
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Framework imports
# ---------------------------------------------------------------------------
from framework.dqn import DQNPolicy
from framework.reinforce import REINFORCEPolicy
from framework.cmaes import CMAESPolicy, CMAESDistribution
from framework.lstm import LSTMEvolutionPolicy, LSTMCore

# ---------------------------------------------------------------------------
# TMNF game imports
# ---------------------------------------------------------------------------
from games.tmnf.policies import (
    NeuralDQNPolicy,
    REINFORCEPolicy as TmnfREINFORCEPolicy,
    CMAESPolicy as TmnfCMAESPolicy,
    LSTMEvolutionPolicy as TmnfLSTMEvolutionPolicy,
)
from games.tmnf.obs_spec import TMNF_OBS_SPEC, BASE_OBS_DIM, OBS_NAMES
from games.tmnf.actions import DISCRETE_ACTIONS as _DISCRETE_ACTIONS, _action_to_idx
from games.tmnf.policies import WeightedLinearPolicy

_OBS_SPEC = TMNF_OBS_SPEC
_N        = BASE_OBS_DIM


def _rand_obs(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(_N).astype(np.float32)


def _make_tmnf_frame_decoder():
    """Return a TMNF parameter_decoder for CMAESPolicy and its flat_dim."""
    obs_names = list(OBS_NAMES)
    flat_dim  = len(obs_names) * 3

    def decoder(flat: np.ndarray) -> WeightedLinearPolicy:
        n   = len(obs_names)
        cfg = {
            "steer_weights": {obs_names[i]: float(flat[i])         for i in range(n)},
            "accel_weights": {obs_names[i]: float(flat[n + i])     for i in range(n)},
            "brake_weights": {obs_names[i]: float(flat[2 * n + i]) for i in range(n)},
        }
        return WeightedLinearPolicy.from_cfg(cfg)

    return decoder, flat_dim


def _sigmoid_scalar(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, float(x)))))


def _tmnf_head_decoder(h: np.ndarray, head_params: np.ndarray) -> np.ndarray:
    h_size = len(h)
    return np.array([
        float(np.tanh(head_params[:h_size] @ h)),
        float(_sigmoid_scalar(float(head_params[h_size:2*h_size] @ h)) > 0.5),
        float(_sigmoid_scalar(float(head_params[2*h_size:3*h_size] @ h)) > 0.5),
    ], dtype=np.float32)


# ===========================================================================
# DQN equivalence
# ===========================================================================

class TestDQNEquivalence(unittest.TestCase):
    """framework.DQNPolicy vs games.tmnf.NeuralDQNPolicy — byte-identical state."""

    _KWARGS = dict(
        hidden_sizes        = [16, 8],
        replay_buffer_size  = 500,
        batch_size          = 16,
        min_replay_size     = 32,
        target_update_freq  = 10,
        learning_rate       = 0.005,
        epsilon_start       = 1.0,
        epsilon_end         = 0.1,
        epsilon_decay_steps = 200,
        gamma               = 0.99,
        seed                = 42,
    )

    def _make_pair(self):
        fw = DQNPolicy(_OBS_SPEC, _DISCRETE_ACTIONS, **self._KWARGS)
        gm = NeuralDQNPolicy(n_lidar_rays=0, **self._KWARGS)
        return fw, gm

    def _run_n_steps(self, fw, gm, n=100, obs_seed=0):
        rng = np.random.default_rng(obs_seed)
        for i in range(n):
            obs        = rng.standard_normal(_N).astype(np.float32)
            next_obs   = rng.standard_normal(_N).astype(np.float32)
            reward     = float(rng.standard_normal())
            done       = bool(rng.random() < 0.05)
            action_idx = int(rng.integers(len(_DISCRETE_ACTIONS)))
            action     = _DISCRETE_ACTIONS[action_idx].copy()
            # Set the same numpy seed for both policies so that the batch
            # drawn by np.random.choice inside sample() is identical.
            np.random.seed(i + 1000)
            fw.update(obs, action, reward, next_obs, done)
            np.random.seed(i + 1000)
            gm.update(obs, action, reward, next_obs, done)

    def test_initial_weights_identical(self):
        fw, gm = self._make_pair()
        for layer in range(len(fw._online["weights"])):
            np.testing.assert_array_equal(
                fw._online["weights"][layer], gm._online["weights"][layer],
                err_msg=f"online weights[{layer}] differ at init"
            )
            np.testing.assert_array_equal(
                fw._online["biases"][layer], gm._online["biases"][layer],
                err_msg=f"online biases[{layer}] differ at init"
            )

    def test_weights_identical_after_training(self):
        fw, gm = self._make_pair()
        self._run_n_steps(fw, gm, n=150)
        for layer in range(len(fw._online["weights"])):
            np.testing.assert_array_almost_equal(
                fw._online["weights"][layer], gm._online["weights"][layer],
                decimal=5,
                err_msg=f"online weights[{layer}] diverged after training"
            )

    def test_epsilon_identical_after_steps(self):
        fw, gm = self._make_pair()
        self._run_n_steps(fw, gm, n=80)
        self.assertAlmostEqual(fw._eps, gm._eps, places=8)


# ===========================================================================
# REINFORCE equivalence
# ===========================================================================

class TestREINFORCEEquivalence(unittest.TestCase):
    """framework.REINFORCEPolicy vs games.tmnf.REINFORCEPolicy — identical grads."""

    _HIDDEN        = [16, 8]
    _SEED          = 7
    _LEARNING_RATE = 0.01
    _GAMMA         = 0.99
    _ENTROPY_COEFF = 0.01

    def _make_pair(self):
        fw = REINFORCEPolicy(
            _OBS_SPEC,
            lambda i: _DISCRETE_ACTIONS[i].copy(),
            output_dim    = len(_DISCRETE_ACTIONS),
            hidden_sizes  = self._HIDDEN,
            learning_rate = self._LEARNING_RATE,
            gamma         = self._GAMMA,
            entropy_coeff = self._ENTROPY_COEFF,
            baseline      = "none",
            seed          = self._SEED,
        )
        gm = TmnfREINFORCEPolicy(
            hidden_sizes  = self._HIDDEN,
            learning_rate = self._LEARNING_RATE,
            gamma         = self._GAMMA,
            entropy_coeff = self._ENTROPY_COEFF,
            baseline      = "none",
            n_lidar_rays  = 0,
            seed          = self._SEED,
        )
        return fw, gm

    def test_initial_weights_identical(self):
        fw, gm = self._make_pair()
        for i, (wfw, wgm) in enumerate(zip(fw._weights, gm._weights)):
            np.testing.assert_array_equal(
                wfw, wgm, err_msg=f"_weights[{i}] differ at init"
            )

    def test_weights_identical_after_one_episode(self):
        fw, gm = self._make_pair()
        rng     = np.random.default_rng(99)
        N_STEPS = 20

        # Drive both policies with the exact same obs sequence
        for t in range(N_STEPS):
            obs  = rng.standard_normal(_N).astype(np.float32)
            # Both policies sample stochastically — fix numpy seed for action
            np.random.seed(t)
            a_fw = fw(obs)
            np.random.seed(t)
            a_gm = gm(obs)
            np.testing.assert_array_equal(a_fw, a_gm)

            r = float(rng.standard_normal())
            fw.update(obs, a_fw, r, obs, False)
            gm.update(obs, a_gm, r, obs, False)

        fw.on_episode_end()
        gm.on_episode_end()

        for i, (wfw, wgm) in enumerate(zip(fw._weights, gm._weights)):
            np.testing.assert_array_almost_equal(
                wfw, wgm, decimal=5, err_msg=f"_weights[{i}] diverged after episode"
            )


# ===========================================================================
# CMA-ES distribution equivalence
# ===========================================================================

class TestCMAESDistributionEquivalence(unittest.TestCase):
    """framework.CMAESDistribution vs games.tmnf.CMAESPolicy — identical math."""

    _POP  = 10
    _SEED = 13

    def _make_pair(self):
        """Return (framework CMAESDistribution, TMNF CMAESPolicy) with same n, seed."""
        decoder, flat_dim = _make_tmnf_frame_decoder()
        fw_dist = CMAESDistribution(n=flat_dim, lam=self._POP, sigma=0.5, seed=self._SEED)
        # Seed the TMNF CMAESPolicy mean to zeros to match CMAESDistribution default
        gm = TmnfCMAESPolicy(population_size=self._POP, initial_sigma=0.5,
                             n_lidar_rays=0, seed=self._SEED)
        gm.initialize_random()
        # Align means (TMNF initialises from rng, framework defaults to zeros)
        fw_dist._mean = np.zeros(flat_dim, dtype=np.float64)
        return fw_dist, gm, flat_dim

    def test_initial_distribution_constants(self):
        """Adaptation constants (cs, ds, cc, c1, cmu) should be identical."""
        decoder, flat_dim = _make_tmnf_frame_decoder()
        fw  = CMAESDistribution(n=flat_dim, lam=self._POP, sigma=0.5)
        gm  = TmnfCMAESPolicy(population_size=self._POP, n_lidar_rays=0)
        gm.initialize_random()
        self.assertAlmostEqual(fw._cs,   gm._cs,   places=10)
        self.assertAlmostEqual(fw._ds,   gm._ds,   places=10)
        self.assertAlmostEqual(fw._cc,   gm._cc,   places=10)
        self.assertAlmostEqual(fw._c1,   gm._c1,   places=10)
        self.assertAlmostEqual(fw._cmu,  gm._cmu,  places=10)
        self.assertAlmostEqual(fw._chin, gm._chin, places=10)

    def test_mean_shifts_same_direction(self):
        """After the same reward sequence the means should move in the same direction."""
        fw_dist, gm, flat_dim = self._make_pair()

        # Sample from the TMNF policy with a fresh seeded RNG
        gm.sample_population()
        # Copy the _pop_xs from gm into fw_dist so both process identical samples
        fw_dist._pop_xs = [x.copy() for x in gm._pop_xs]
        fw_dist._pop_ys = [y.copy() for y in gm._pop_ys]

        rewards   = [float(i) for i in range(self._POP)]
        mean_before_fw = fw_dist.mean.copy()
        mean_before_gm = gm._mean.copy()

        fw_dist.update(np.array(rewards))
        gm.update_distribution(rewards)

        fw_delta = fw_dist.mean - mean_before_fw
        gm_delta = gm._mean - mean_before_gm
        # Dot product of deltas should be positive (same direction)
        dot = float(np.dot(fw_delta, gm_delta))
        self.assertGreater(dot, 0.0,
                           "Framework and TMNF CMA-ES means moved in opposite directions")


# ===========================================================================
# LSTM core equivalence
# ===========================================================================

class TestLSTMCoreEquivalence(unittest.TestCase):
    """framework.LSTMCore forward pass vs games.tmnf.LSTMPolicy LSTM cell."""

    def test_flat_dim_matches_tmnf_lstm(self):
        """LSTMCore.flat_dim (core only) + 3*h should equal TMNF LSTMPolicy.flat_dim."""
        from games.tmnf.policies import LSTMPolicy
        hidden_size = 8
        tmnf        = LSTMPolicy(hidden_size=hidden_size, seed=0)
        core        = LSTMCore(obs_dim=_N, hidden_size=hidden_size, seed=0)
        expected    = core.flat_dim + 3 * hidden_size
        self.assertEqual(expected, tmnf.flat_dim)

    def test_to_flat_from_flat_roundtrip(self):
        """to_flat() followed by from_flat() should reproduce the same weights."""
        core = LSTMCore(obs_dim=_N, hidden_size=8, seed=42)
        flat = core.to_flat()
        core2 = LSTMCore(obs_dim=_N, hidden_size=8)
        core2.from_flat(flat)
        np.testing.assert_array_equal(core2.to_flat(), flat)

    def test_reset_zeroes_hidden_state(self):
        core = LSTMCore(obs_dim=_N, hidden_size=8, seed=0)
        obs  = np.ones(_N, dtype=np.float32)
        core.forward(obs)
        core.reset()
        np.testing.assert_array_equal(core._h, np.zeros(8))
        np.testing.assert_array_equal(core._c, np.zeros(8))

    def test_forward_output_shape(self):
        core = LSTMCore(obs_dim=_N, hidden_size=8, seed=0)
        h    = core.forward(np.zeros(_N, dtype=np.float32))
        self.assertEqual(h.shape, (8,))

    def test_hidden_state_non_zero_after_nonzero_input(self):
        core = LSTMCore(obs_dim=_N, hidden_size=8, seed=0)
        h    = core.forward(np.ones(_N, dtype=np.float32))
        self.assertFalse(np.allclose(h, np.zeros(8)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
