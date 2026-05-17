"""Tests for games/sc2/eval.py.

SC2Env and SC2Client are mocked throughout — no PySC2 installation needed.
"""

import argparse
import collections
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from games.sc2.obs_spec import BASE_OBS_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(dim: int = BASE_OBS_DIM) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def _make_info(outcome: float | None = None, game_loop: float = 500.0) -> dict:
    return {
        "score": 10.0,
        "player_outcome": outcome,
        "game_loop": game_loop,
        "available_fn_ids": [0, 1, 2],
    }


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "track": None,
        "num_episodes": 1,
        "bot_difficulty": None,
        "eval_speed": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# CLI argument validation
# ---------------------------------------------------------------------------

class TestArgValidation(unittest.TestCase):
    """--num-episodes and --eval-speed must be ≥ 1."""

    def _make_parser(self):
        import argparse as ap

        def _positive_int(name):
            def _check(v):
                iv = int(v)
                if iv < 1:
                    raise ap.ArgumentTypeError(f"{name} must be ≥ 1, got {v}")
                return iv
            return _check

        parser = ap.ArgumentParser()
        parser.add_argument("--num-episodes", type=_positive_int("--num-episodes"), default=1)
        parser.add_argument("--eval-speed", type=_positive_int("--eval-speed"), default=None)
        return parser

    def test_num_episodes_zero_rejected(self):
        parser = self._make_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--num-episodes", "0"])

    def test_num_episodes_negative_rejected(self):
        parser = self._make_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--num-episodes", "-3"])

    def test_num_episodes_one_accepted(self):
        parser = self._make_parser()
        ns = parser.parse_args(["--num-episodes", "1"])
        self.assertEqual(ns.num_episodes, 1)

    def test_eval_speed_zero_rejected(self):
        parser = self._make_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--eval-speed", "0"])

    def test_eval_speed_positive_accepted(self):
        parser = self._make_parser()
        ns = parser.parse_args(["--eval-speed", "4"])
        self.assertEqual(ns.eval_speed, 4)

    def test_bot_difficulty_invalid_name_rejected(self):
        """cheater_easy / cheater_hard are not real PySC2 Difficulty names."""
        import argparse as ap
        parser = ap.ArgumentParser()
        parser.add_argument(
            "--bot-difficulty",
            choices=[
                "very_easy", "easy", "medium", "medium_hard",
                "hard", "harder", "very_hard",
                "cheat_vision", "cheat_money", "cheat_insane",
            ],
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["--bot-difficulty", "cheater_easy"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--bot-difficulty", "elite"])

    def test_bot_difficulty_valid_names_accepted(self):
        import argparse as ap
        parser = ap.ArgumentParser()
        parser.add_argument(
            "--bot-difficulty",
            choices=[
                "very_easy", "easy", "medium", "medium_hard",
                "hard", "harder", "very_hard",
                "cheat_vision", "cheat_money", "cheat_insane",
            ],
        )
        for name in ("very_easy", "hard", "cheat_insane"):
            ns = parser.parse_args(["--bot-difficulty", name])
            self.assertEqual(ns.bot_difficulty, name)


# ---------------------------------------------------------------------------
# _print_action_breakdown
# ---------------------------------------------------------------------------

class TestPrintActionBreakdown(unittest.TestCase):

    def _capture(self, counts, total, subs=0, label="test"):
        from games.sc2.eval import _print_action_breakdown
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _print_action_breakdown(counts, total, subs, label=label)
        return buf.getvalue()

    def test_output_contains_fn_names(self):
        counts = collections.Counter({"no_op": 10, "Move_screen": 5})
        out = self._capture(counts, 15)
        self.assertIn("no_op", out)
        self.assertIn("Move_screen", out)

    def test_percentages_sum_to_100_when_complete(self):
        counts = collections.Counter({"no_op": 7, "select_army": 3})
        out = self._capture(counts, 10)
        # Both percentages appear in the output.
        self.assertIn("70.0", out)
        self.assertIn("30.0", out)

    def test_substitution_line_shown_when_nonzero(self):
        counts = collections.Counter({"no_op": 5})
        out = self._capture(counts, 10, subs=3)
        self.assertIn("substituted", out)

    def test_substitution_line_hidden_when_zero(self):
        counts = collections.Counter({"no_op": 10})
        out = self._capture(counts, 10, subs=0)
        self.assertNotIn("substituted", out)

    def test_zero_total_steps_does_not_divide_by_zero(self):
        counts: collections.Counter = collections.Counter()
        # Should not raise.
        self._capture(counts, 0)


# ---------------------------------------------------------------------------
# _print_aggregate_summary
# ---------------------------------------------------------------------------

class TestPrintAggregateSummary(unittest.TestCase):

    def _make_result(self, outcome, score=10.0, game_loop=500, steps=20,
                     reward=5.0, subs=0):
        return {
            "outcome": outcome,
            "score": score,
            "game_loop": game_loop,
            "steps": steps,
            "cumulative_reward": reward,
            "action_counts": collections.Counter({"no_op": steps}),
            "substitution_count": subs,
        }

    def _capture(self, results, counts):
        from games.sc2.eval import _print_aggregate_summary
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _print_aggregate_summary(results, counts)
        return buf.getvalue()

    def test_win_rate_correct(self):
        results = [
            self._make_result(1),   # win
            self._make_result(-1),  # loss
            self._make_result(1),   # win
        ]
        all_counts = sum((r["action_counts"] for r in results), collections.Counter())
        out = self._capture(results, all_counts)
        self.assertIn("66.7%", out)

    def test_all_wins(self):
        results = [self._make_result(1) for _ in range(3)]
        all_counts = sum((r["action_counts"] for r in results), collections.Counter())
        out = self._capture(results, all_counts)
        self.assertIn("100.0%", out)

    def test_single_episode_no_crash(self):
        results = [self._make_result(0)]
        all_counts = results[0]["action_counts"].copy()
        # Should not raise.
        self._capture(results, all_counts)


# ---------------------------------------------------------------------------
# _run_episode
# ---------------------------------------------------------------------------

class TestRunEpisode(unittest.TestCase):
    """Episode loop: policy called, actions logged, lifecycle hooks invoked."""

    def _make_env(self, n_steps=3, obs_dim=BASE_OBS_DIM, outcome=None):
        """Return a mock SC2Env that terminates after n_steps."""
        env = MagicMock()
        env.reset.return_value = (_make_obs(obs_dim), _make_info())

        fake_client = MagicMock()
        fake_client.last_fn_idx = 1  # select_army
        env._client = fake_client

        steps = []
        for i in range(n_steps):
            done = i == n_steps - 1
            steps.append((_make_obs(obs_dim), 1.0, done, False,
                          _make_info(outcome=outcome if done else None)))
        env.step.side_effect = steps
        return env

    def _make_policy(self, obs_dim=BASE_OBS_DIM):
        policy = MagicMock()
        policy.return_value = np.zeros(4, dtype=np.float32)
        return policy

    def test_policy_called_each_step(self):
        from games.sc2.eval import _run_episode
        env = self._make_env(n_steps=4)
        policy = self._make_policy()
        _run_episode(env, policy, ep_idx=1, total_episodes=1)
        self.assertEqual(policy.call_count, 4)

    def test_result_step_count_matches_env_steps(self):
        from games.sc2.eval import _run_episode
        env = self._make_env(n_steps=5)
        policy = self._make_policy()
        result = _run_episode(env, policy, ep_idx=1, total_episodes=1)
        self.assertEqual(result["steps"], 5)

    def test_on_episode_start_receives_info(self):
        """on_episode_start(**info) should receive the reset info dict."""
        from games.sc2.eval import _run_episode
        env = self._make_env(n_steps=1)
        policy = self._make_policy()
        policy.on_episode_start = MagicMock()
        _run_episode(env, policy, ep_idx=1, total_episodes=1)
        policy.on_episode_start.assert_called_once()
        kwargs = policy.on_episode_start.call_args[1]
        self.assertIn("available_fn_ids", kwargs)

    def test_policy_update_called_each_step(self):
        """update() is called once per step so available_fn_ids stay fresh."""
        from games.sc2.eval import _run_episode
        env = self._make_env(n_steps=3)
        policy = self._make_policy()
        policy.update = MagicMock()
        _run_episode(env, policy, ep_idx=1, total_episodes=1)
        self.assertEqual(policy.update.call_count, 3)

    def test_result_outcome_from_terminal_info(self):
        from games.sc2.eval import _run_episode
        env = self._make_env(n_steps=2, outcome=1)
        policy = self._make_policy()
        result = _run_episode(env, policy, ep_idx=1, total_episodes=1)
        self.assertEqual(result["outcome"], 1)

    def test_substitution_counted_when_executed_differs_from_requested(self):
        """When the client substitutes a different action, substitution_count
        should increase."""
        from games.sc2.eval import _run_episode

        env = self._make_env(n_steps=2)
        # Policy requests fn_idx=2 (Move_screen) but client executes fn_idx=1
        # (select_army).
        env._client.last_fn_idx = 1
        policy = self._make_policy()
        policy.return_value = np.array([2, 0.5, 0.5, 0], dtype=np.float32)

        result = _run_episode(env, policy, ep_idx=1, total_episodes=1)
        self.assertEqual(result["substitution_count"], 2)

    def test_no_substitution_when_executed_matches_requested(self):
        from games.sc2.eval import _run_episode

        env = self._make_env(n_steps=3)
        # Policy requests fn_idx=1 and client also executes fn_idx=1.
        env._client.last_fn_idx = 1
        policy = self._make_policy()
        policy.return_value = np.array([1, 0.0, 0.0, 0], dtype=np.float32)

        result = _run_episode(env, policy, ep_idx=1, total_episodes=1)
        self.assertEqual(result["substitution_count"], 0)

    def test_cumulative_reward_summed(self):
        from games.sc2.eval import _run_episode
        env = self._make_env(n_steps=4)
        policy = self._make_policy()
        result = _run_episode(env, policy, ep_idx=1, total_episodes=1)
        self.assertAlmostEqual(result["cumulative_reward"], 4.0)


if __name__ == "__main__":
    unittest.main()
