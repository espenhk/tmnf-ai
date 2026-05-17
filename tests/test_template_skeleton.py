"""Tests that the games/_template/ skeleton is valid, parseable Python.

This catches drift between the template and reality — if a framework
import path changes, this test fails before a new contributor gets
confused by a broken template.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest


TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "games" / "_template"

# All Python files in the template that must be parseable.
TEMPLATE_PYTHON_FILES = [
    "__init__.py",
    "adapter.py",
    "env.py",
    "obs_spec.py",
    "actions.py",
    "reward.py",
    "analytics.py",
]


class TestTemplateSkeleton:
    """Verify the _template directory is structurally valid."""

    def test_template_directory_exists(self):
        """The template directory must exist."""
        assert TEMPLATE_DIR.is_dir(), f"Template directory not found: {TEMPLATE_DIR}"

    def test_config_directory_exists(self):
        """The template config directory must exist."""
        config_dir = TEMPLATE_DIR / "config"
        assert config_dir.is_dir(), f"Config directory not found: {config_dir}"

    @pytest.mark.parametrize("filename", TEMPLATE_PYTHON_FILES)
    def test_python_file_exists(self, filename: str):
        """Each expected Python file must exist in the template."""
        filepath = TEMPLATE_DIR / filename
        assert filepath.is_file(), f"Template file not found: {filepath}"

    @pytest.mark.parametrize("filename", TEMPLATE_PYTHON_FILES)
    def test_python_file_parseable(self, filename: str):
        """Each Python file must be valid syntax (ast.parse succeeds)."""
        filepath = TEMPLATE_DIR / filename
        source = filepath.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(filepath))
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {filepath}: {e}")

    def test_config_training_params_exists(self):
        """training_params.yaml must exist in the config dir."""
        path = TEMPLATE_DIR / "config" / "training_params.yaml"
        assert path.is_file(), f"Training params not found: {path}"

    def test_config_reward_config_exists(self):
        """reward_config.yaml must exist in the config dir."""
        path = TEMPLATE_DIR / "config" / "reward_config.yaml"
        assert path.is_file(), f"Reward config not found: {path}"

    def test_readme_exists(self):
        """README.md must exist in the template."""
        path = TEMPLATE_DIR / "README.md"
        assert path.is_file(), f"README not found: {path}"

    def test_obs_spec_importable(self):
        """obs_spec.py must define BASE_OBS_DIM and OBS_NAMES."""
        filepath = TEMPLATE_DIR / "obs_spec.py"
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)

        top_level_names = {
            node.targets[0].id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }

        assert "BASE_OBS_DIM" in top_level_names, (
            "obs_spec.py must define BASE_OBS_DIM"
        )
        assert "OBS_NAMES" in top_level_names, (
            "obs_spec.py must define OBS_NAMES"
        )

    def test_actions_defines_discrete_actions(self):
        """actions.py must define DISCRETE_ACTIONS."""
        filepath = TEMPLATE_DIR / "actions.py"
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)

        top_level_names = {
            node.targets[0].id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }

        assert "DISCRETE_ACTIONS" in top_level_names, (
            "actions.py must define DISCRETE_ACTIONS"
        )

    def test_adapter_has_make_adapter(self):
        """adapter.py must define a make_adapter() function."""
        filepath = TEMPLATE_DIR / "adapter.py"
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)

        function_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }

        assert "make_adapter" in function_names, (
            "adapter.py must define a make_adapter() function"
        )

    def test_template_not_in_game_choices(self):
        """_template must NOT appear in main.py's --game choices."""
        main_path = Path(__file__).resolve().parent.parent / "main.py"
        if not main_path.exists():
            pytest.skip("main.py not found (running tests outside repo root)")

        source = main_path.read_text(encoding="utf-8")
        # Check that _template is not in the choices list
        assert "_template" not in source, (
            "main.py must not include '_template' in --game choices"
        )
