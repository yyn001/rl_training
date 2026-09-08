from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def push_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    push_start_iteration: int = 5000,
    velocity_range: dict | None = None,
):
    """Enable push_robot once learning iteration exceeds threshold.

    Uses the runner's ``current_learning_iteration`` (correctly restored on
    resume) instead of env step counters that reset to zero.
    """
    if velocity_range is None:
        velocity_range = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}

    runner = getattr(env, "_runner_ref", None)
    current_iter = runner.current_learning_iteration if runner else 0

    try:
        term_cfg = env.event_manager.get_term_cfg("push_robots")
        if not getattr(env, "_push_enabled", False) and current_iter >= push_start_iteration:
            term_cfg.params["velocity_range"] = velocity_range
            env._push_enabled = True
    except ValueError:
        pass

    return float(getattr(env, "_push_enabled", False))
