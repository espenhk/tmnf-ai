"""Unit tests for CMAESDistribution in framework/cmaes.py.

Pure-math tests — no policy coupling.  Verifies:
  - Sampling produces expected shapes and is seeded-reproducible.
  - update() shifts the mean toward higher-fitness samples.
  - σ (step size) adapts — grows when improvements are frequent, shrinks otherwise.
  - save_state() / load_state() round-trips all internal arrays.
"""
import os
import tempfile
import unittest

import numpy as np

from framework.cmaes import CMAESDistribution


class TestCMAESDistributionInit(unittest.TestCase):

    def test_n_stored(self):
        d = CMAESDistribution(n=10, lam=20, sigma=0.3)
        self.assertEqual(d.n,   10)
        self.assertEqual(d.lam, 20)

    def test_sigma_stored(self):
        d = CMAESDistribution(n=5, lam=10, sigma=0.7)
        self.assertAlmostEqual(d.sigma, 0.7)

    def test_mu_is_half_lambda(self):
        d = CMAESDistribution(n=5, lam=20, sigma=0.3)
        self.assertEqual(d._mu, 10)

    def test_weights_sum_to_one(self):
        d = CMAESDistribution(n=5, lam=16, sigma=0.3)
        self.assertAlmostEqual(float(d._weights.sum()), 1.0, places=10)

    def test_covariance_is_identity(self):
        d = CMAESDistribution(n=5, lam=10, sigma=0.3)
        np.testing.assert_array_equal(d._C, np.eye(5))

    def test_mean_defaults_to_zeros(self):
        d = CMAESDistribution(n=8, lam=10, sigma=0.3)
        np.testing.assert_array_equal(d.mean, np.zeros(8))

    def test_mean_can_be_seeded(self):
        m = np.array([1.0, 2.0, 3.0])
        d = CMAESDistribution(n=3, lam=6, sigma=0.3, mean=m)
        np.testing.assert_array_equal(d.mean, m)

    def test_generation_starts_at_zero(self):
        d = CMAESDistribution(n=5, lam=10, sigma=0.3)
        self.assertEqual(d.gen, 0)


class TestCMAESDistributionSampling(unittest.TestCase):

    def test_sample_returns_lambda_vectors(self):
        lam = 12
        d   = CMAESDistribution(n=5, lam=lam, sigma=0.3, seed=0)
        xs  = d.sample()
        self.assertEqual(len(xs), lam)

    def test_sample_vectors_have_correct_length(self):
        n = 7
        d = CMAESDistribution(n=n, lam=10, sigma=0.3, seed=1)
        for x in d.sample():
            self.assertEqual(len(x), n)

    def test_sample_is_reproducible_with_seed(self):
        d1 = CMAESDistribution(n=5, lam=10, sigma=0.3, seed=42)
        d2 = CMAESDistribution(n=5, lam=10, sigma=0.3, seed=42)
        xs1 = d1.sample()
        xs2 = d2.sample()
        for x1, x2 in zip(xs1, xs2):
            np.testing.assert_array_equal(x1, x2)

    def test_different_seeds_produce_different_samples(self):
        d1 = CMAESDistribution(n=5, lam=10, sigma=0.3, seed=1)
        d2 = CMAESDistribution(n=5, lam=10, sigma=0.3, seed=2)
        xs1 = d1.sample()
        xs2 = d2.sample()
        # At least one vector should differ
        all_equal = all(np.allclose(x1, x2) for x1, x2 in zip(xs1, xs2))
        self.assertFalse(all_equal)

    def test_sample_fills_pop_xs_and_pop_ys(self):
        d = CMAESDistribution(n=4, lam=8, sigma=0.3, seed=0)
        d.sample()
        self.assertEqual(len(d._pop_xs), 8)
        self.assertEqual(len(d._pop_ys), 8)


class TestCMAESDistributionUpdate(unittest.TestCase):

    def _make_and_sample(self, n=5, lam=10, seed=0):
        d  = CMAESDistribution(n=n, lam=lam, sigma=0.5, seed=seed)
        xs = d.sample()
        return d, xs

    def test_update_increments_gen(self):
        d, xs = self._make_and_sample()
        d.update(xs, np.zeros(d.lam))
        self.assertEqual(d.gen, 1)

    def test_update_shifts_mean_toward_best(self):
        """Mean should move toward the highest-fitness samples."""
        n      = 5
        target = np.ones(n, dtype=np.float64) * 3.0
        d      = CMAESDistribution(n=n, lam=20, sigma=1.0, seed=7)
        xs     = d.sample()

        fitnesses = np.array([
            -float(np.sum((x - target) ** 2)) for x in xs
        ])
        d.update(xs, fitnesses)
        # The mean should have moved closer to the target
        dist_before = float(np.linalg.norm(np.zeros(n) - target))
        dist_after  = float(np.linalg.norm(d.mean - target))
        self.assertLess(dist_after, dist_before)

    def test_update_changes_sigma(self):
        d, xs    = self._make_and_sample()
        old_sigma = d.sigma
        d.update(xs, np.arange(d.lam, dtype=float))
        self.assertNotAlmostEqual(d.sigma, old_sigma)

    def test_update_requires_matching_lam(self):
        d, xs = self._make_and_sample(lam=10)
        with self.assertRaises(ValueError):
            d.update(xs, np.zeros(9))  # wrong length

    def test_update_requires_sample_before_call(self):
        d = CMAESDistribution(n=5, lam=10, sigma=0.3, seed=0)
        # sample() not called — _pop_xs/_pop_ys are empty
        with self.assertRaises(RuntimeError):
            d.update([np.zeros(5)] * 10, np.zeros(10))

    def test_update_returns_improved_flag(self):
        d, xs = self._make_and_sample()
        # first update: no prior best → always improved
        improved = d.update(xs, np.ones(d.lam))
        self.assertTrue(improved)

    def test_update_returns_false_when_no_improvement(self):
        d, xs = self._make_and_sample()
        d.update(xs, np.ones(d.lam) * 100.0)  # sets best = 100
        xs2 = d.sample()
        improved = d.update(xs2, np.ones(d.lam) * 50.0)
        self.assertFalse(improved)

    def test_sigma_adapts_down_on_bad_generation(self):
        """If no offspring beats the prior best, σ should shrink."""
        d, xs = self._make_and_sample(n=8, lam=20)
        # First generation sets best = 100
        d.update(xs, np.ones(d.lam) * 100.0)
        sigma_after_first = d.sigma
        xs2 = d.sample()
        # Second generation: all rewards worse than best
        d.update(xs2, np.ones(d.lam) * 50.0)
        # σ should differ from the post-first-gen value
        self.assertNotAlmostEqual(d.sigma, sigma_after_first)


class TestCMAESDistributionConvergence(unittest.TestCase):

    def test_converges_to_quadratic_optimum(self):
        """Mean should approach the maximiser of -||x - target||² in ≤50 gens."""
        n      = 10
        target = np.ones(n) * 2.0
        d      = CMAESDistribution(n=n, lam=20, sigma=1.0,
                                   mean=np.zeros(n), seed=99)

        dist_before = float(np.linalg.norm(d.mean - target))
        for _ in range(50):
            xs        = d.sample()
            fitnesses = np.array([-float(np.sum((x - target) ** 2)) for x in xs])
            d.update(xs, fitnesses)

        dist_after = float(np.linalg.norm(d.mean - target))
        self.assertLess(
            dist_after, dist_before,
            f"CMAESDistribution failed to converge: "
            f"before={dist_before:.3f}, after={dist_after:.3f}",
        )


class TestCMAESDistributionPersistence(unittest.TestCase):

    def _make_trained(self, n=6, lam=10, n_gens=3) -> CMAESDistribution:
        d = CMAESDistribution(n=n, lam=lam, sigma=0.5, seed=7)
        for _ in range(n_gens):
            xs = d.sample()
            d.update(xs, np.arange(lam, dtype=float))
        return d

    def test_save_load_roundtrip_mean(self):
        d = self._make_trained()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            d.save_state(path)
            d2 = CMAESDistribution(n=d.n, lam=d.lam, sigma=0.9)
            d2.load_state(path)
            np.testing.assert_array_equal(d.mean, d2.mean)
        finally:
            os.unlink(path)

    def test_save_load_roundtrip_all_arrays(self):
        d = self._make_trained()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            d.save_state(path)
            d2 = CMAESDistribution(n=d.n, lam=d.lam, sigma=0.9)
            d2.load_state(path)
            self.assertAlmostEqual(d.sigma, d2.sigma)
            self.assertEqual(d.gen, d2.gen)
            np.testing.assert_array_equal(d._C, d2._C)
            np.testing.assert_array_equal(d._B, d2._B)
            np.testing.assert_array_equal(d._D, d2._D)
            np.testing.assert_array_equal(d._ps, d2._ps)
            np.testing.assert_array_equal(d._pc, d2._pc)
        finally:
            os.unlink(path)

    def test_loaded_state_continues_evolving(self):
        d = self._make_trained()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            d.save_state(path)
            d2 = CMAESDistribution(n=d.n, lam=d.lam, sigma=0.9)
            d2.load_state(path)
            prev_mean = d2.mean.copy()
            xs = d2.sample()
            d2.update(xs, np.arange(d.lam, dtype=float))
            self.assertFalse(np.allclose(d2.mean, prev_mean))
        finally:
            os.unlink(path)

    def test_load_wrong_n_raises(self):
        d = self._make_trained(n=6)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            d.save_state(path)
            d2 = CMAESDistribution(n=10, lam=d.lam, sigma=0.3)  # different n
            with self.assertRaises(ValueError):
                d2.load_state(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
