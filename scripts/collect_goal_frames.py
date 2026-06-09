import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import random
from pathlib import Path

import cv2
import numpy as np
import metaworld

from collect_metaworld_avid import (
    AVAILABLE_CAMERA_NAMES,
    DEFAULT_CAMERA_NAMES,
    MAX_STEPS,
    _render_rgb_frame,
    _resolve_camera_id,
    _set_randomized_resets,
    get_policy_for_task,
)


def collect_goal(
    env,
    policy,
    camera_ids: dict[str, int],
    max_steps: int = MAX_STEPS,
    reset_seed: int | None = None,
):
    """Run the expert until success and return the goal observation.

    The episode is reset with ``reset_seed`` so the object/goal placement is
    deterministic and the goal can be regenerated (or the env reset back to the
    same configuration to use the goal as a target).

    Returns a dict ``{"proprio": (7,), "frames": {cam: (H, W, 3)}, "step": int,
    "reset_seed": int|None}`` captured at the first successful timestep, or
    ``None`` if the episode never succeeds within ``max_steps``.
    """
    obs, _ = env.reset(seed=reset_seed)

    for step_idx in range(max_steps):
        action = np.asarray(policy.get_action(obs), dtype=np.float32)
        obs, _reward, terminated, truncated, info = env.step(action)

        if info.get("success", 0.0) > 0.0:
            frames = {
                cam: _render_rgb_frame(env, cam_id)
                for cam, cam_id in camera_ids.items()
            }
            proprio = np.asarray(env.data.qpos[:7], dtype=np.float32).copy()
            return {
                "proprio": proprio,
                "frames": frames,
                "step": step_idx,
                "reset_seed": reset_seed,
            }

        if terminated or truncated:
            break

    return None


def _save_png(frame: np.ndarray, path: Path) -> None:
    """Write an RGB frame to disk as a (lossless) PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(
        description="Collect goal frames at task success using the expert policy"
    )
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=None,
        help="Environment names to collect goals for. Default: all MT50 envs "
        "that have a scripted expert policy.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=list(DEFAULT_CAMERA_NAMES),
        choices=AVAILABLE_CAMERA_NAMES,
        metavar="CAMERA",
        help=f"Cameras to capture the goal frame from (default {list(DEFAULT_CAMERA_NAMES)}).",
    )
    parser.add_argument(
        "--num_goals",
        type=int,
        default=1,
        help="Number of successful goal frames to collect per environment.",
    )
    parser.add_argument(
        "--max_attempts_factor",
        type=int,
        default=5,
        help="Give up on an env after num_goals * this many failed episodes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/metaworld/goals",
        help="Directory to write goal PNGs and proprio .txt files into.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible task/reset sampling.",
    )
    args = parser.parse_args()

    if args.num_goals < 1:
        parser.error("num_goals must be a positive integer")
    if len(set(args.cameras)) != len(args.cameras):
        parser.error("--cameras must not contain duplicate camera names")

    if args.seed is not None:
        random.seed(args.seed)

    print("Initializing Meta-World MT50...", flush=True)
    mt50 = metaworld.MT50()

    if args.envs:
        env_names = args.envs
    else:
        env_names = list(mt50.train_classes.keys())

    tasks_by_env: dict[str, list] = {}
    for task in mt50.train_tasks:
        tasks_by_env.setdefault(task.env_name, []).append(task)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for env_name in env_names:
        if env_name not in mt50.train_classes:
            print(f"[Skip] {env_name}: not a valid MT50 environment.", flush=True)
            continue

        policy = get_policy_for_task(env_name)
        if policy is None:
            print(f"[Skip] {env_name}: no scripted expert policy.", flush=True)
            continue

        tasks = tasks_by_env.get(env_name, [])
        if not tasks:
            print(f"[Skip] {env_name}: no tasks available.", flush=True)
            continue

        env = mt50.train_classes[env_name](
            render_mode="rgb_array", camera_name=args.cameras[0]
        )
        camera_ids = {cam: _resolve_camera_id(env, cam) for cam in args.cameras}

        env_dir = output_dir / env_name
        env_dir.mkdir(parents=True, exist_ok=True)

        collected = 0
        attempts = 0
        max_attempts = args.num_goals * args.max_attempts_factor
        while collected < args.num_goals and attempts < max_attempts:
            attempts += 1
            env.set_task(random.choice(tasks))
            _set_randomized_resets(env, True)
            # Deterministic per-goal reset seed (reproducible from --seed) so the
            # exact object/goal configuration behind this goal can be recreated.
            reset_seed = random.randint(0, 2**31 - 1)
            goal = collect_goal(env, policy, camera_ids, reset_seed=reset_seed)
            if goal is None:
                continue

            # One PNG per camera, plus one proprio .txt (shared across cameras).
            for cam, frame in goal["frames"].items():
                _save_png(frame, env_dir / f"{cam}_goal_{collected}.png")
            np.savetxt(
                env_dir / f"goal_{collected}_proprio.txt",
                goal["proprio"].reshape(1, -1),
                fmt="%.8f",
                header=(
                    f"env={env_name} reset_seed={goal['reset_seed']} "
                    f"success_step={goal['step']} policy=expert "
                    f"proprio=qpos[:7]"
                ),
            )

            collected += 1

        env.close()

        if collected < args.num_goals:
            print(
                f"[Warn] {env_name}: only {collected}/{args.num_goals} goals "
                f"after {attempts} attempts.",
                flush=True,
            )
        else:
            print(f"[OK] {env_name}: {collected} goal(s).", flush=True)

    print(f"\nGoal frames saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
