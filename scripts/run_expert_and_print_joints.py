import argparse
import importlib
import random

import metaworld
import numpy as np


def kebab_to_camel(kebab_case_string: str) -> str:
    parts = kebab_case_string.split("-")
    return "".join(part.capitalize() for part in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Roll out Meta-World and print future robot joint rotations for each action."
        )
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=None,
        help="Environment name (for example: reach-v3). Defaults to a random MT50 env.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of future rollout steps.",
    )
    parser.add_argument(
        "--action",
        type=float,
        nargs=4,
        metavar=("DX", "DY", "DZ", "GRIP"),
        default=(0.0, 0.0, 0.0, 0.0),
        help=(
            "Fixed action to roll out. Used unless --expert-policy is set. "
            "Each value should be in [-1, 1]."
        ),
    )
    parser.add_argument(
        "--expert-policy",
        action="store_true",
        help="Use scripted expert actions instead of --action.",
    )
    parser.add_argument(
        "--disable-collisions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable all MuJoCo contacts so rollout is kinematics-focused.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for env/task selection.",
    )
    parser.add_argument(
        "--print-degrees",
        action="store_true",
        help="Also print joint values in degrees.",
    )
    return parser.parse_args()


def load_policy(env_name: str):
    policy_name = f"sawyer_{env_name.replace('-v3', '')}_v3_policy"
    policy_cls_name = f"Sawyer{kebab_to_camel(env_name.replace('-v3', ''))}V3Policy"

    try:
        module = importlib.import_module(f"metaworld.policies.{policy_name}")
        return getattr(module, policy_cls_name)()
    except (ImportError, AttributeError):
        return None


def set_task_for_env(mt50: metaworld.MT50, env, env_name: str, rng: random.Random) -> None:
    tasks = [task for task in mt50.train_tasks if task.env_name == env_name]
    if not tasks:
        raise RuntimeError(f"No tasks found for {env_name}")
    env.set_task(rng.choice(tasks))


def disable_collisions(env) -> None:
    model = env.model
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0


def get_arm_joint_qpos_indices(env) -> tuple[list[str], np.ndarray]:
    model = env.model
    joint_names: list[str] = []
    joint_qpos_indices: list[int] = []
    for joint_id in range(model.njnt):
        joint_name = model.joint(joint_id).name
        if isinstance(joint_name, str) and joint_name.startswith("right_j"):
            joint_names.append(joint_name)
            joint_qpos_indices.append(int(model.jnt_qposadr[joint_id]))

    if not joint_names:
        raise RuntimeError("Could not find Sawyer arm joints (right_j0..right_j6).")

    sorted_pairs = sorted(
        zip(joint_names, joint_qpos_indices),
        key=lambda pair: int(pair[0].replace("right_j", "")),
    )
    names_sorted = [name for name, _ in sorted_pairs]
    idx_sorted = np.array([idx for _, idx in sorted_pairs], dtype=np.int64)
    return names_sorted, idx_sorted


def format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):+0.5f}" for v in values) + "]"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    mt50 = metaworld.MT50(seed=args.seed)
    env_names = list(mt50.train_classes.keys())
    env_name = args.env_name if args.env_name is not None else rng.choice(env_names)
    if env_name not in mt50.train_classes:
        raise ValueError(f"Unknown env name: {env_name}")

    env = mt50.train_classes[env_name](render_mode="rgb_array")
    set_task_for_env(mt50, env, env_name, rng)

    if args.disable_collisions:
        disable_collisions(env)

    policy = load_policy(env_name) if args.expert_policy else None
    if args.expert_policy and policy is None:
        print(f"Expert policy not found for {env_name}. Falling back to random actions.")

    joint_names, joint_qpos_indices = get_arm_joint_qpos_indices(env)
    print(f"Environment: {env_name}")
    print(f"Joint order: {joint_names}")
    print(f"Collisions disabled: {args.disable_collisions}")
    print(f"Rollout steps: {args.steps}")
    print("-" * 100)

    obs, _ = env.reset(seed=args.seed)
    fixed_action = np.asarray(args.action, dtype=np.float32)
    for step_idx in range(args.steps):
        if policy is not None:
            action = np.asarray(policy.get_action(obs), dtype=np.float32)
        elif args.expert_policy:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = fixed_action

        obs, _, _, truncated, _ = env.step(action)
        joint_rad = env.data.qpos[joint_qpos_indices].copy()
        line = (
            f"step={step_idx:03d} "
            f"action={format_vector(action)} "
            f"joints_rad={format_vector(joint_rad)}"
        )
        if args.print_degrees:
            joint_deg = np.rad2deg(joint_rad)
            line += f" joints_deg={format_vector(joint_deg)}"
        print(line)

        if truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
