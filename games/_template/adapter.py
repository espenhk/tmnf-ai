"""<GAME_NAME> game adapter — builds config bundles for train_rl.

Copy this file to ``games/<your_game>/adapter.py`` and replace every
``NotImplementedError`` body with your game-specific logic.  Use
``games/car_racing/adapter.py`` as the reference implementation.
"""

from __future__ import annotations

from framework.run_config import GameSpec, ProbeSpec, WarmupSpec, PolicyExtras


class TemplateAdapter:
    """Adapter stub — rename to ``<YourGame>Adapter``."""

    name = "_template"  # Replace with your game slug, e.g. "my_game"
    config_dir = "games/_template/config"  # Replace with "games/<your_game>/config"

    # ------------------------------------------------------------------
    # Directory & label helpers
    # ------------------------------------------------------------------

    def experiment_dir(
        self, experiment_name: str, training_params: dict,
        track_override: str | None,
    ) -> str:
        """Return the experiment directory path for this run.

        Convention: ``experiments/<game_name>/<experiment_name>``
        For games with multiple tracks/maps, include the track in the path.
        """
        raise NotImplementedError(
            "Return the experiment directory path, e.g. "
            "f'experiments/{self.name}/{experiment_name}'"
        )

    def experiment_dir_root(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        """Return the root directory under which all experiments live.

        Convention: ``experiments/<game_name>``
        """
        raise NotImplementedError(
            "Return the root experiments directory, e.g. "
            "f'experiments/{self.name}'"
        )

    def track_label(
        self, training_params: dict, track_override: str | None,
    ) -> str:
        """Return the human-readable track/map label for directory naming.

        If your game only has one environment (e.g. CarRacing), return the
        game name.  If it has multiple tracks/maps, use ``track_override``
        or fall back to a training_params key.
        """
        raise NotImplementedError(
            "Return a track/map label string, e.g. "
            "track_override or self.name"
        )

    def decorate_reward_cfg(
        self, reward_cfg: dict, training_params: dict,
        track_override: str | None,
    ) -> None:
        """Mutate *reward_cfg* with game-specific keys before training.

        For example, TMNF injects ``centerline_path`` here.  If your game
        has no extra reward config beyond what's in the YAML, leave this
        as a no-op (just ``pass``).
        """
        raise NotImplementedError(
            "Mutate reward_cfg in place, or replace this body with ``pass`` "
            "if no game-specific decoration is needed."
        )

    # ------------------------------------------------------------------
    # Build methods — these assemble the typed config objects
    # ------------------------------------------------------------------

    def build_game_spec(
        self, experiment_name: str, experiment_dir: str,
        weights_file: str, reward_cfg_file: str,
        training_params: dict, track_override: str | None,
    ) -> GameSpec:
        """Build a GameSpec for this experiment.

        You must provide:
        - ``make_env_fn``: a zero-arg callable that returns a BaseGameEnv
        - ``obs_spec``: your game's ObsSpec instance
        - ``discrete_actions``: your DISCRETE_ACTIONS array
        - ``save_results_fn``: your analytics save function

        See ``games/car_racing/adapter.py`` for a working example.
        """
        # -- Lazy imports so the template doesn't crash at module level --
        # from games.<your_game>.obs_spec import YOUR_OBS_SPEC
        # from games.<your_game>.actions import DISCRETE_ACTIONS
        # from games.<your_game>.analytics import save_experiment_results

        # def _make_env():
        #     from games.<your_game>.env import make_env
        #     return make_env(
        #         experiment_dir=experiment_dir,
        #         max_episode_time_s=training_params["in_game_episode_s"],
        #     )

        # return GameSpec(
        #     experiment_name=experiment_name,
        #     track=self.track_label(training_params, track_override),
        #     make_env_fn=_make_env,
        #     obs_spec=YOUR_OBS_SPEC,
        #     head_names=["action_dim_1", "action_dim_2"],  # your action dims
        #     discrete_actions=DISCRETE_ACTIONS,
        #     weights_file=weights_file,
        #     reward_config_file=reward_cfg_file,
        #     save_results_fn=save_experiment_results,
        # )

        raise NotImplementedError(
            "Build and return a GameSpec.  See the commented-out code above."
        )

    def build_probe(self, training_params: dict) -> ProbeSpec | None:
        """Build a ProbeSpec, or return None to skip probe/cold-start.

        Probes run a small set of deterministic actions at the start of
        training to establish baseline reward signals.  Return None if
        your game doesn't need this (e.g. CarRacing skips it).
        """
        raise NotImplementedError(
            "Return a ProbeSpec or None.  "
            "Returning None skips the probe/cold-start phase."
        )

    def build_warmup(self, training_params: dict) -> WarmupSpec | None:
        """Build a WarmupSpec, or return None to skip warmup.

        Warmup runs a fixed action (e.g. full throttle) for N episodes
        before training begins.  Return None if not needed.
        """
        raise NotImplementedError(
            "Return a WarmupSpec or None.  "
            "Returning None skips the warmup phase."
        )

    def build_extras(
        self, weights_file: str, training_params: dict, re_initialize: bool,
    ) -> PolicyExtras | None:
        """Build PolicyExtras, or return None if no game-specific policy types.

        PolicyExtras lets a game register custom policies beyond the
        framework defaults.  Most games return None here.
        """
        raise NotImplementedError(
            "Return PolicyExtras or None.  "
            "Most games return None."
        )


def make_adapter() -> TemplateAdapter:
    """Factory function — must exist in every adapter module."""
    return TemplateAdapter()
