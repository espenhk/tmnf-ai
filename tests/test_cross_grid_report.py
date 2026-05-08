"""Tests for cross_grid_report.py — recursive cross-grid summary collation."""
from __future__ import annotations

import os
import tempfile
import unittest

from framework.analytics import (
    ExperimentData,
    GreedySimResult,
    RunTrace,
    save_experiment_data_json,
)


def _make_data(
    name: str,
    rewards: list[float],
    training_params: dict,
    weights_file: str,
    reward_config_file: str,
) -> ExperimentData:
    greedy_sims = [
        GreedySimResult(
            sim=idx + 1,
            reward=reward,
            improved=idx == 0 or reward > max(rewards[:idx]),
            throttle_counts=[0, 10, 90],
            total_steps=100,
            trace=RunTrace(
                pos_x=[0.0, 1.0],
                pos_z=[0.0, 1.0],
                throttle_state=[(1.0, 0.0)],
                total_reward=reward,
            ),
        )
        for idx, reward in enumerate(rewards)
    ]
    return ExperimentData(
        experiment_name=name,
        probe_results=[],
        cold_start_restarts=[],
        greedy_sims=greedy_sims,
        probe_floor=None,
        weights_file=weights_file,
        reward_config_file=reward_config_file,
        training_params=training_params,
        timings={"greedy_s": 60.0},
        track="test_track",
    )


def _write_run(
    version_dir: str,
    name: str,
    rewards: list[float],
    training_params: dict,
    reward_yaml: str = "progress_weight: 10000.0\n",
    stored_weights_file: str | None = None,
    stored_reward_config_file: str | None = None,
) -> str:
    experiment_dir = os.path.join(version_dir, name)
    results_dir = os.path.join(experiment_dir, "results")
    os.makedirs(results_dir)
    weights_file = os.path.join(experiment_dir, "policy_weights.yaml")
    reward_config_file = os.path.join(experiment_dir, "reward_config.yaml")
    with open(weights_file, "w", encoding="utf-8") as f:
        f.write("{}\n")
    with open(reward_config_file, "w", encoding="utf-8") as f:
        f.write(reward_yaml)
    save_experiment_data_json(
        _make_data(
            name,
            rewards,
            training_params,
            stored_weights_file or weights_file,
            stored_reward_config_file or reward_config_file,
        ),
        results_dir,
    )
    return experiment_dir


def _write_summary(version_dir: str, summary_name: str, run_name: str) -> str:
    summary_dir = os.path.join(version_dir, summary_name)
    os.makedirs(summary_dir)
    with open(os.path.join(summary_dir, "comparison_rewards.png"), "wb") as f:
        f.write(b"png")
    with open(os.path.join(summary_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Summary\n\n"
            "![Comparison](comparison_rewards.png)\n\n"
            f"![Run](../{run_name}/results/greedy_best_run.png)\n"
        )
    return summary_dir


class TestCrossGridReport(unittest.TestCase):
    def test_discovers_only_policy_version_summaries(self):
        from cross_grid_report import discover_grid_search_families

        with tempfile.TemporaryDirectory() as tmpdir:
            valid_version_dir = os.path.join(tmpdir, "track", "genetic", "v1")
            os.makedirs(valid_version_dir)
            _write_summary(valid_version_dir, "gs_genetic_v1__summary", "run1")

            invalid_version_dir = os.path.join(tmpdir, "track", "genetic", "latest")
            os.makedirs(invalid_version_dir)
            _write_summary(invalid_version_dir, "gs_genetic_latest__summary", "run1")

            families = discover_grid_search_families(tmpdir)

            self.assertEqual(len(families), 1)
            self.assertEqual(families[0].policy_name, "genetic")
            self.assertEqual(families[0].version_name, "v1")

    def test_build_cross_grid_report_copies_summaries_and_compares_families(self):
        from cross_grid_report import build_cross_grid_report

        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = os.path.join(tmpdir, "experiments")

            genetic_version = os.path.join(root_dir, "arena", "genetic", "v1")
            os.makedirs(genetic_version)
            _write_run(
                genetic_version,
                "gs_genetic_v1__pop16__ms0.1",
                [5.0, 12.0, 30.0],
                {"policy_type": "genetic", "population_size": 16, "mutation_scale": 0.1},
            )
            _write_run(
                genetic_version,
                "gs_genetic_v1__pop32__ms0.2",
                [8.0, 9.0, 10.0],
                {"policy_type": "genetic", "population_size": 32, "mutation_scale": 0.2},
            )
            _write_summary(genetic_version, "gs_genetic_v1__summary", "gs_genetic_v1__pop16__ms0.1")

            neural_version = os.path.join(root_dir, "arena", "neural_net", "v2")
            os.makedirs(neural_version)
            _write_run(
                neural_version,
                "gs_nn_v2__hs32x16__lr0.01",
                [1.0, 20.0, 18.0],
                {
                    "policy_type": "neural_net",
                    "hidden_sizes": [32, 16],
                    "learning_rate": 0.01,
                },
            )
            _write_summary(neural_version, "gs_nn_v2__summary", "gs_nn_v2__hs32x16__lr0.01")

            invalid_summary_dir = os.path.join(root_dir, "arena", "neural_net", "draft")
            os.makedirs(invalid_summary_dir)
            _write_summary(invalid_summary_dir, "gs_nn_draft__summary", "ignore_me")

            output_dir = os.path.join(tmpdir, "cross")
            summary_path = build_cross_grid_report(
                root_dir=root_dir,
                output_dir=output_dir,
                summary_name="arena_compare",
            )

            self.assertIsNotNone(summary_path)
            self.assertTrue(os.path.exists(summary_path))

            copied_summary = os.path.join(
                output_dir,
                "grid_summaries",
                "genetic",
                "v1",
                "gs_genetic_v1__summary",
                "summary.md",
            )
            self.assertTrue(os.path.exists(copied_summary))
            with open(copied_summary, encoding="utf-8") as f:
                copied_content = f.read()
            self.assertIn("comparison_rewards.png", copied_content)
            self.assertNotIn("../gs_genetic_v1__pop16__ms0.1/results/greedy_best_run.png", copied_content)
            self.assertIn("genetic/v1/gs_genetic_v1__pop16__ms0.1/results/greedy_best_run.png", copied_content)

            with open(summary_path, encoding="utf-8") as f:
                content = f.read()

            self.assertIn("| 1 | genetic | v1 | 2 | +30.0 | +20.0 |", content)
            self.assertIn("| 2 | neural_net | v2 | 1 | +20.0 | +20.0 |", content)
            self.assertIn("Population sizes: 16, 32", content)
            self.assertIn("Hidden layer sizes: [32, 16]", content)
            self.assertIn("Average generation to best reward: 3", content)
            self.assertIn("Average generation to best reward: 2", content)
            self.assertIn("| `mutation_scale` | 0.1 |", content)
            self.assertIn("| `learning_rate` | 0.01 |", content)
            self.assertNotIn("draft", content)

    def test_missing_rewards_rank_after_real_results(self):
        from cross_grid_report import build_cross_grid_report

        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = os.path.join(tmpdir, "experiments")

            full_version = os.path.join(root_dir, "arena", "genetic", "v1")
            os.makedirs(full_version)
            _write_run(
                full_version,
                "good_run",
                [2.0, 4.0],
                {"policy_type": "genetic", "population_size": 8},
            )
            _write_summary(full_version, "gs_genetic_v1__summary", "good_run")

            empty_version = os.path.join(root_dir, "arena", "neural_net", "v2")
            os.makedirs(empty_version)
            _write_run(
                empty_version,
                "empty_run",
                [],
                {"policy_type": "neural_net", "hidden_sizes": [16, 16]},
            )
            _write_summary(empty_version, "gs_neural_net_v2__summary", "empty_run")

            summary_path = build_cross_grid_report(root_dir, summary_name="compare")

            with open(summary_path, encoding="utf-8") as f:
                content = f.read()

            self.assertIn("| 1 | genetic | v1 | 1 | +4.0 | +4.0 |", content)
            self.assertIn("| 2 | neural_net | v2 | 1 | — | — |", content)

    def test_rerun_ignores_existing_output_tree(self):
        from cross_grid_report import build_cross_grid_report

        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = os.path.join(tmpdir, "experiments")
            version_dir = os.path.join(root_dir, "arena", "genetic", "v1")
            os.makedirs(version_dir)
            _write_run(
                version_dir,
                "run1",
                [1.0, 3.0],
                {"policy_type": "genetic", "population_size": 16},
            )
            _write_summary(version_dir, "gs_genetic_v1__summary", "run1")

            first_summary = build_cross_grid_report(root_dir, summary_name="compare")
            second_summary = build_cross_grid_report(root_dir, summary_name="compare")

            self.assertEqual(first_summary, second_summary)
            with open(second_summary, encoding="utf-8") as f:
                content = f.read()

            self.assertIn("found 1 grid-search summaries", content)
            self.assertEqual(content.count("| 1 | genetic | v1 | 1 | +3.0 | +3.0 |"), 1)

    def test_moved_experiment_paths_remap_reward_config_and_weights(self):
        from cross_grid_report import build_cross_grid_report

        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = os.path.join(tmpdir, "experiments")
            version_dir = os.path.join(root_dir, "arena", "genetic", "v1")
            os.makedirs(version_dir)
            _write_run(
                version_dir,
                "run1",
                [1.0, 3.0],
                {"policy_type": "genetic", "population_size": 16},
                reward_yaml="progress_weight: 10000.0\nstep_penalty: -0.1\n",
                stored_weights_file=os.path.join(tmpdir, "old", "policy_weights.yaml"),
                stored_reward_config_file=os.path.join(tmpdir, "old", "reward_config.yaml"),
            )
            _write_summary(version_dir, "gs_genetic_v1__summary", "run1")

            summary_path = build_cross_grid_report(root_dir, summary_name="compare")

            with open(summary_path, encoding="utf-8") as f:
                content = f.read()

            self.assertIn("| `step_penalty` | -0.1 |", content)
