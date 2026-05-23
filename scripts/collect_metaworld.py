import metaworld
import metaworld.policies as mw_policies
import numpy as np
import cv2
import random
import argparse
import multiprocessing
import os
import mujoco
from metaworld_policies import (
    MetaWorldPolicy,
    NoisyExpertPolicy,
    RandomWalk,
)
from metaworld.sensors.force_torque_sensor import ForceTorqueSensor
from metaworld.sensors.tactile_digit_sensor import TactileDigitSensor
from metaworld.sensors.visual import DepthCameraSensor

from pathlib import Path

# Defaults
DEFAULT_DATASET_PATH = "data/metaworld/metaworld_corner2.hdf5"
DEFAULT_TEMP_DIR = "data/metaworld/temp/"
DEFAULT_NUM_EPISODES = 1
DEFAULT_EXPERT_NOISE_MIN = 0.1
DEFAULT_EXPERT_NOISE_MAX = 0.25
DEFAULT_RANDOMIZE_EVERY_RESET = True
DEFAULT_RANDOM_WALK_DIRECTION_STDDEV = 0.45
DEFAULT_RANDOM_WALK_GRAVITY_STRENGTH = 0.01
DEFAULT_RANDOM_WALK_LEVY_ALPHA = 1.35
DEFAULT_RANDOM_WALK_MIN_STEP = 1.0
DEFAULT_RANDOM_WALK_MAX_STEP = 4.0
DEFAULT_RANDOMIZE_HAND_START = True
MAX_STEPS = 200
RGB_CAMERA_NAME = "corner2"
DEPTH_CAMERA_NAME = RGB_CAMERA_NAME
RGB_IMAGE_WIDTH = 128
RGB_IMAGE_HEIGHT = 128
DEPTH_IMAGE_WIDTH = 128
DEPTH_IMAGE_HEIGHT = 128
TACTILE_RESOLUTION = 64


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for dataset HDF5 writing. "
            "Install it to use full_data_collection or worker_process."
        ) from exc
    return h5py


def get_policy_for_task(task_name):
    """
    Finds the correct scripted policy for a given task name.
    """
    base_name = task_name.split("-v")[0]
    camel_case = "".join(x.title() for x in base_name.split("-"))
    policy_name = f"Sawyer{camel_case}V3Policy"

    policy_cls = getattr(mw_policies, policy_name, None)
    return policy_cls() if policy_cls is not None else None


def _parse_task_names(raw_tasks: list[str] | None) -> list[str] | None:
    if raw_tasks is None:
        return None

    task_names = []
    for item in raw_tasks:
        for task_name in item.split(","):
            task_name = task_name.strip()
            if task_name and task_name not in task_names:
                task_names.append(task_name)
    return task_names


def _set_randomized_resets(env, enabled: bool) -> None:
    """Allow fresh object/goal placement samples on every reset."""
    env._freeze_rand_vec = not enabled


def _sample_hand_start_position(env) -> np.ndarray:
    """Sample a reachable initial end-effector position from the task workspace."""
    hand_low = getattr(env, "hand_low", None)
    hand_high = getattr(env, "hand_high", None)
    if hand_low is None or hand_high is None:
        raise ValueError(
            "Cannot randomize hand start because this environment does not expose "
            "hand_low/hand_high workspace bounds."
        )

    hand_low = np.asarray(hand_low, dtype=np.float32)
    hand_high = np.asarray(hand_high, dtype=np.float32)
    if hand_low.shape != (3,) or hand_high.shape != (3,):
        raise ValueError(
            "Expected hand_low/hand_high to be 3D workspace bounds, got "
            f"{hand_low.shape} and {hand_high.shape}."
        )

    return np.random.default_rng().uniform(hand_low, hand_high).astype(np.float32)


def _set_hand_start_position(env, hand_start_pos: np.ndarray) -> None:
    """Set the next reset's Sawyer hand start position."""
    hand_start_pos = np.asarray(hand_start_pos, dtype=np.float32)
    env.hand_init_pos = hand_start_pos
    if hasattr(env, "init_config") and "hand_init_pos" in env.init_config:
        env.init_config["hand_init_pos"] = hand_start_pos


def _build_episode_policy_schedule(
    num_episodes: int,
    policy_mode: str = "mixed",
) -> list[str]:
    """Build the episode-level policy schedule for one environment."""
    if policy_mode == "expert_only":
        return ["noisy_expert"] * num_episodes

    if policy_mode == "random_walk_only":
        return ["random_walk"] * num_episodes

    if policy_mode != "mixed":
        raise ValueError(f"Unknown policy_mode: {policy_mode}")

    # Default mixed schedule: roughly 50/50 noisy expert and random walk.
    num_noisy_expert = (num_episodes + 1) // 2
    num_random_walk = num_episodes // 2
    schedule = (
        ["noisy_expert"] * num_noisy_expert
        + ["random_walk"] * num_random_walk
    )
    random.shuffle(schedule)
    return schedule


def _make_random_walk_policy(args, episode_seed: int | None = None) -> RandomWalk:
    """Create a broad random-walk controller for workspace exploration."""
    return RandomWalk(
        direction_policy="gravity",
        step_length_policy="levy",
        direction_kwargs={
            "stddev": args.random_walk_direction_stddev,
            "gravity_strength": args.random_walk_gravity_strength,
        },
        step_length_kwargs={
            "alpha": args.random_walk_levy_alpha,
            "min_step": args.random_walk_min_step,
            "max_step": args.random_walk_max_step,
        },
        seed=episode_seed,
    )


def _make_policy_for_episode(env_name: str, args, policy_type: str):
    if policy_type == "noisy_expert":
        expert_policy = get_policy_for_task(env_name)
        if expert_policy is None:
            raise ValueError(f"No scripted policy available for {env_name}")
        noise_scale = random.uniform(
            args.expert_noise_min,
            args.expert_noise_max,
        )
        return (
            NoisyExpertPolicy(expert_policy, noise_scale=noise_scale),
            {
                "policy_type": "noisy_expert",
                "noise_scale": noise_scale,
            },
        )

    if policy_type == "random_walk":
        episode_seed = random.randint(0, 2**31 - 1)
        return (
            _make_random_walk_policy(args, episode_seed=episode_seed),
            {
                "policy_type": "random_walk",
                "direction_policy": "gravity",
                "step_length_policy": "levy",
                "direction_stddev": args.random_walk_direction_stddev,
                "gravity_strength": args.random_walk_gravity_strength,
                "levy_alpha": args.random_walk_levy_alpha,
                "min_step": args.random_walk_min_step,
                "max_step": args.random_walk_max_step,
                "random_walk_seed": episode_seed,
            },
        )

    raise ValueError(f"Unknown policy_type: {policy_type}")


def _initialize_policy_reference_position(
    env,
    policy: MetaWorldPolicy,
    obs: np.ndarray,
) -> None:
    if not hasattr(policy, "set_reference_position"):
        return

    reference_position = np.asarray(obs[0:3], dtype=np.float32)
    if isinstance(policy, RandomWalk):
        hand_low = getattr(env, "hand_low", None)
        hand_high = getattr(env, "hand_high", None)
        if hand_low is not None and hand_high is not None:
            reference_position = (
                np.asarray(hand_low, dtype=np.float32)
                + np.asarray(hand_high, dtype=np.float32)
            ) / 2.0

    policy.set_reference_position(reference_position)


def _get_object_xyzs(env) -> tuple[np.ndarray, np.ndarray]:
    """Return (object_1_xyz, object_2_xyz) from the env if available."""
    obj1 = np.full(3, np.nan, dtype=np.float32)
    obj2 = np.full(3, np.nan, dtype=np.float32)
    try:
        obj_pos = np.asarray(env._get_pos_objects(), dtype=np.float32).reshape(-1)
    except Exception:
        return obj1, obj2

    if obj_pos.size < 3:
        return obj1, obj2

    count = obj_pos.size // 3
    obj_pos = obj_pos[: count * 3]
    objs = np.split(obj_pos, count)
    obj1 = objs[0].copy()
    if count > 1:
        obj2 = objs[1].copy()
    return obj1, obj2


def _gripper_touching_any(env) -> bool:
    """Check if either gripper pad is touching any non-gripper geometry."""
    try:
        data = env.unwrapped.data
        leftpad_geom_id = data.geom("leftpad_geom").id
        rightpad_geom_id = data.geom("rightpad_geom").id
        pad_ids = {leftpad_geom_id, rightpad_geom_id}
        for contact in data.contact:
            if contact.geom1 in pad_ids or contact.geom2 in pad_ids:
                other = contact.geom2 if contact.geom1 in pad_ids else contact.geom1
                if other in pad_ids:
                    continue
                if data.efc_force[contact.efc_address] > 0:
                    return True
        return False
    except Exception:
        try:
            return bool(env.touching_main_object)
        except Exception:
            return False


def _render_rgb_frame(env) -> np.ndarray:
    """Render one RGB frame from the environment's configured camera."""
    frame = None
    renderer = getattr(env.unwrapped, "mujoco_renderer", None)

    if renderer is not None:
        get_viewer = getattr(renderer, "_get_viewer", None)
        if callable(get_viewer):
            try:
                viewer = get_viewer(render_mode="rgb_array")
                # The depth sensor uses its own offscreen OpenGL context. Make the
                # RGB viewer current again before reading pixels or MuJoCo can
                # return all-black frames after the first depth render.
                viewer.make_context_current()
                frame = viewer.render(
                    render_mode="rgb_array",
                    camera_id=renderer.camera_id,
                )
            except Exception:
                frame = None

    if frame is None:
        frame = env.render()
    if frame is None:
        frame = env.render(
            offscreen=True,
            resolution=(RGB_IMAGE_WIDTH, RGB_IMAGE_HEIGHT),
        )
    frame = cv2.resize(frame, (RGB_IMAGE_WIDTH, RGB_IMAGE_HEIGHT))
    return cv2.flip(frame, 0)


def collect_episode(
    env,
    policy: MetaWorldPolicy,
    task_name: str,
    depth_sensor: DepthCameraSensor,
    tactile_sensor: TactileDigitSensor,
    force_torque_sensor: ForceTorqueSensor,
    max_steps: int = MAX_STEPS,
    reset_seed: int | None = None,
    policy_name: str = "policy",
    randomize_hand_start: bool = DEFAULT_RANDOMIZE_HAND_START,
):
    """
    Runs one episode and returns a DICTIONARY.
    Always returns data, even if expert fails.
    """
    if randomize_hand_start:
        hand_start_pos = _sample_hand_start_position(env)
        _set_hand_start_position(env, hand_start_pos)
    else:
        hand_start_pos = np.asarray(env.hand_init_pos, dtype=np.float32).copy()

    obs, info = env.reset(seed=reset_seed)
    depth_sensor.reset(env)
    tactile_sensor.reset(env)
    force_torque_sensor.reset(env)
    _initialize_policy_reference_position(env, policy, obs)
    current_info = dict(info or {})

    rgb_frames = []
    depth_frames = []
    tactile_frames = []
    proprioception_list = []
    gripper_list = []
    ee_xyz_list = []
    object_1_xyz_list = []
    object_2_xyz_list = []
    bool_contact_list = []
    success_list = []
    force_torque_list = []
    actions = []
    expert_actions_raw = []
    expert_actions_clipped = []
    action_noises = []

    success = False

    for _ in range(max_steps):
        rgb_frames.append(_render_rgb_frame(env))

        depth_sensor.update(env)
        depth_frames.append(depth_sensor.get_depth_as_image().copy())

        tactile_sensor.update(env)
        left_tactile, right_tactile = tactile_sensor.get_finger_images()
        tactile_frames.append(np.stack((left_tactile, right_tactile), axis=0))

        proprioception_list.append(env.data.qpos[:7].copy())
        gripper_list.append(np.float32(obs[3]))
        ee_xyz_list.append(np.asarray(env.get_endeff_pos(), dtype=np.float32).copy())
        obj1, obj2 = _get_object_xyzs(env)
        object_1_xyz_list.append(obj1)
        object_2_xyz_list.append(obj2)
        bool_contact_list.append(_gripper_touching_any(env))
        current_success = bool(current_info.get("success", 0.0) > 0.0)
        success_list.append(current_success)
        success = success or current_success
        force_torque_sensor.update(env)
        force_torque_list.append(
            np.asarray(force_torque_sensor.read(), dtype=np.float32)
        )

        action = np.asarray(policy.get_action(obs), dtype=np.float32)
        actions.append(action)
        if getattr(policy, "last_expert_action_raw", None) is not None:
            expert_actions_raw.append(policy.last_expert_action_raw.copy())
        if getattr(policy, "last_expert_action_clipped", None) is not None:
            expert_actions_clipped.append(policy.last_expert_action_clipped.copy())
        if getattr(policy, "last_noise", None) is not None:
            action_noises.append(policy.last_noise.copy())
        obs, _reward, terminated, truncated, info = env.step(action)
        current_info = dict(info or {})

        if info.get("success", 0.0) > 0.0:
            success = True

        if terminated or truncated:
            break

    if not success and policy_name != "random_walk":
        print(
            f"  [Warn] {policy_name} rollout failed on {task_name} but saving anyway.",
            flush=True,
        )

    data_dict = {
        "pixels": np.array(rgb_frames, dtype=np.uint8),
        "depth": np.array(depth_frames, dtype=np.float32),
        "proprio": np.array(proprioception_list, dtype=np.float32),
        "tactile": np.array(tactile_frames, dtype=np.float32),
        "force_torque": np.array(force_torque_list, dtype=np.float32),
        "gripper": np.array(gripper_list, dtype=np.float32),
        "ee_xyz": np.array(ee_xyz_list, dtype=np.float32),
        "object_1_xyz": np.array(object_1_xyz_list, dtype=np.float32),
        "object_2_xyz": np.array(object_2_xyz_list, dtype=np.float32),
        "bool_contact": np.array(bool_contact_list, dtype=np.bool_),
        "success": np.array(success_list, dtype=np.bool_),
        "action": np.array(actions, dtype=np.float32),
        "hand_start_pos": hand_start_pos,
    }
    if len(expert_actions_raw) == len(actions):
        data_dict["expert_action_raw"] = np.array(expert_actions_raw, dtype=np.float32)
    if len(expert_actions_clipped) == len(actions):
        data_dict["expert_action_clipped"] = np.array(
            expert_actions_clipped, dtype=np.float32
        )
    if len(action_noises) == len(actions):
        data_dict["action_noise"] = np.array(action_noises, dtype=np.float32)
    return data_dict


def worker_process(worker_id, env_names, args):
    """
    Worker function to process a subset of environments.
    """
    h5py = _require_h5py()
    temp_path = Path(args.temp_dir) / f"temp_worker_{worker_id}.hdf5"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[Worker {worker_id}] Started. Processing {len(env_names)} environments.",
        flush=True,
    )

    # Initialize Meta-World locally
    mt50 = metaworld.MT50()

    tasks_by_env = {}
    for task in mt50.train_tasks:
        if task.env_name in env_names:
            if task.env_name not in tasks_by_env:
                tasks_by_env[task.env_name] = []
            tasks_by_env[task.env_name].append(task)

    with h5py.File(temp_path, "w") as f:
        # We finish one environment completely before loading the assets for the next.
        for i, env_name in enumerate(env_names):
            print(
                f"[Worker {worker_id}] Processing {env_name} ({i + 1}/{len(env_names)})",
                flush=True,
            )

            env_cls = mt50.train_classes[env_name]
            if get_policy_for_task(env_name) is None:
                print(
                    f"[Worker {worker_id}] Skipping {env_name}: no scripted policy available.",
                    flush=True,
                )
                continue

            available_tasks = tasks_by_env.get(env_name, [])
            if not available_tasks:
                print(
                    f"[Worker {worker_id}] Skipping {env_name}: no tasks available.",
                    flush=True,
                )
                continue

            # Create the Group ONCE per environment
            task_group = f.create_group(env_name)
            task_group.attrs["task_name"] = env_name
            task_group.attrs["randomize_every_reset"] = bool(
                args.randomize_every_reset
            )
            task_group.attrs["randomize_hand_start"] = bool(
                args.randomize_hand_start
            )

            episode_global_idx = 0
            env = env_cls(render_mode="rgb_array", camera_name=RGB_CAMERA_NAME)
            depth_sensor = DepthCameraSensor(
                camera_name=DEPTH_CAMERA_NAME,
                height=DEPTH_IMAGE_HEIGHT,
                width=DEPTH_IMAGE_WIDTH,
                normalize=False,
            )
            tactile_sensor = TactileDigitSensor(
                resolution=TACTILE_RESOLUTION,
                noise_std=0.0,
                base_texture=False,
                normalize=False,
                photometric_render=False,
            )
            force_torque_sensor = ForceTorqueSensor(
                geom_names=None,
                origin_site="endEffector",
                output_frame="world",
                lowpass_alpha=0.2,
            )

            episode_policy_schedule = _build_episode_policy_schedule(
                args.num_episodes,
                policy_mode=args.policy_mode,
            )
            for policy_type in episode_policy_schedule:
                random_task = random.choice(available_tasks)
                env.set_task(random_task)
                _set_randomized_resets(env, args.randomize_every_reset)

                policy, policy_metadata = _make_policy_for_episode(
                    env_name=env_name,
                    args=args,
                    policy_type=policy_type,
                )

                data_dict = collect_episode(
                    env=env,
                    policy=policy,
                    task_name=env_name,
                    depth_sensor=depth_sensor,
                    tactile_sensor=tactile_sensor,
                    force_torque_sensor=force_torque_sensor,
                    policy_name=policy_metadata["policy_type"],
                    randomize_hand_start=args.randomize_hand_start,
                )

                ep_group = task_group.create_group(f"episode_{episode_global_idx}")
                ep_group.create_dataset(
                    "pixels", data=data_dict["pixels"], compression="gzip"
                )
                ep_group.create_dataset(
                    "depth",
                    data=data_dict["depth"],
                    compression="gzip",
                )
                ep_group.create_dataset(
                    "proprio", data=data_dict["proprio"]
                )
                ep_group.create_dataset(
                    "tactile",
                    data=data_dict["tactile"],
                    compression="gzip",
                )
                ep_group.create_dataset("gripper", data=data_dict["gripper"])
                ep_group.create_dataset("ee_xyz", data=data_dict["ee_xyz"])
                ep_group.create_dataset("object_1_xyz", data=data_dict["object_1_xyz"])
                ep_group.create_dataset("object_2_xyz", data=data_dict["object_2_xyz"])
                ep_group.create_dataset("bool_contact", data=data_dict["bool_contact"])
                ep_group.create_dataset("success", data=data_dict["success"])
                ep_group.create_dataset("force_torque", data=data_dict["force_torque"])
                ep_group.create_dataset("action", data=data_dict["action"])

                ep_group.attrs["policy_type"] = policy_metadata["policy_type"]
                ep_group.attrs["task_name"] = env_name
                ep_group.attrs["randomize_every_reset"] = bool(
                    args.randomize_every_reset
                )
                ep_group.attrs["randomize_hand_start"] = bool(
                    args.randomize_hand_start
                )
                ep_group.attrs["hand_start_pos"] = data_dict["hand_start_pos"]
                ep_group.attrs["hand_low"] = np.asarray(env.hand_low, dtype=np.float32)
                ep_group.attrs["hand_high"] = np.asarray(
                    env.hand_high,
                    dtype=np.float32,
                )
                success_steps = np.flatnonzero(data_dict["success"])
                ep_group.attrs["episode_success"] = bool(success_steps.size > 0)
                ep_group.attrs["first_success_step"] = (
                    int(success_steps[0]) if success_steps.size > 0 else -1
                )
                for key, value in policy_metadata.items():
                    if key == "policy_type":
                        continue
                    ep_group.attrs[key] = value

                episode_global_idx += 1

            env.close()

    print(f"[Worker {worker_id}] Finished.", flush=True)


def explore_qpos():
    # Initialize MT50 and pick an environment
    mt50 = metaworld.MT50()
    env_name = random.choice(list(mt50.train_classes.keys()))
    print(f"Inspecting Environment: {env_name}")

    env_cls = mt50.train_classes[env_name]
    env = env_cls()

    # Set a task before resetting (Crucial step)
    possible_tasks = [t for t in mt50.train_tasks if t.env_name == env_name]
    if possible_tasks:
        env.set_task(random.choice(possible_tasks))
    else:
        print(f"Warning: No tasks found for {env_name}")
        return

    env.reset()

    # Access MuJoCo Model & Data
    # In newer Gymnasium/MuJoCo stacks, these are usually direct attributes
    try:
        model = env.model
        data = env.data
    except AttributeError:
        model = env.unwrapped.model
        data = env.unwrapped.data

    print(f"\nTotal qpos length: {len(data.qpos)}")
    print(f"Number of joints (njnt): {model.njnt}")
    print(f"Number of proprioceptive state (env.model.nq): {model.nq}")
    print(f"{'Index':<6} | {'Joint Name':<30} | {'Value':<10}")
    print("-" * 55)

    # Iterate over joints
    for i in range(model.njnt):
        # mjOBJ_JOINT is the enum for joints
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)

        # Fallback if name is None (sometimes happens for unnamed internal joints)
        if joint_name is None:
            joint_name = f"joint_{i}"

        qpos_addr = model.jnt_qposadr[i]

        # Determine size (1 for hinge/slide, 7 for free joints)
        # We calculate size by looking at the address of the NEXT joint
        if i < model.njnt - 1:
            next_addr = model.jnt_qposadr[i + 1]
            size = next_addr - qpos_addr
        else:
            size = len(data.qpos) - qpos_addr

        # Slice the qpos array
        values = data.qpos[qpos_addr : qpos_addr + size]

        # Formatting
        val_str = ", ".join([f"{v:.3f}" for v in values])
        print(f"{qpos_addr:<6} | {joint_name:<30} | {val_str} | {env.data.qpos[i]:.3f}")


def full_data_collection(args):
    h5py = _require_h5py()
    mt50 = metaworld.MT50()
    all_env_names = list(mt50.train_classes.keys())

    requested_tasks = _parse_task_names(args.tasks)
    if requested_tasks is not None:
        unknown_tasks = sorted(set(requested_tasks) - set(all_env_names))
        if unknown_tasks:
            available = ", ".join(sorted(all_env_names))
            raise ValueError(
                "Unknown MetaWorld task(s): "
                f"{', '.join(unknown_tasks)}. Available tasks: {available}"
            )
        all_env_names = requested_tasks

    if not all_env_names:
        raise ValueError("No MetaWorld tasks selected for collection.")

    # Divide work
    num_workers = min(args.cpus, len(all_env_names))

    # Initialize empty lists for each worker
    chunks = [[] for _ in range(num_workers)]

    # Round-Robin Distribution
    # Env 0 -> Worker 0
    # Env 1 -> Worker 1
    # ...
    # Env 48 -> Worker 0 (Wraps around)
    for i, env_name in enumerate(all_env_names):
        worker_idx = i % num_workers
        chunks[worker_idx].append(env_name)

    print(
        f"Spawning {len(chunks)} workers for {len(all_env_names)} environments: "
        f"{', '.join(all_env_names)}",
        flush=True,
    )

    processes = []
    for i, chunk in enumerate(chunks):
        p = multiprocessing.Process(target=worker_process, args=(i, chunk, args))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("\nAll workers finished. Merging temporary files...", flush=True)

    # Merge Logic
    with h5py.File(args.dataset_path, "w") as final_f:
        for i in range(len(chunks)):
            temp_path = Path(args.temp_dir) / f"temp_worker_{i}.hdf5"

            if os.path.exists(temp_path):
                print(f"Merging {temp_path}...", flush=True)
                with h5py.File(temp_path, "r") as temp_f:
                    for env_key in temp_f.keys():
                        temp_f.copy(env_key, final_f)
                os.remove(temp_path)
            else:
                print(f"Warning: {temp_path} not found!", flush=True)

    print(f"\nDataset successfully saved to {args.dataset_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Parallel Meta-World Data Collection")
    parser.add_argument(
        "--cpus", type=int, default=1, help="Number of CPU cores/workers"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Total episodes to collect per environment across noisy expert and random walk behaviors",
    )
    parser.add_argument(
        "--policy_mode",
        type=str,
        choices=("mixed", "expert_only", "random_walk_only"),
        default="mixed",
        help="Choose whether episodes are split between noisy expert and random walk, or collected from only one policy family.",
    )
    parser.add_argument(
        "--expert_noise_min",
        type=float,
        default=DEFAULT_EXPERT_NOISE_MIN,
        help="Minimum per-episode Gaussian noise scale for expert actions",
    )
    parser.add_argument(
        "--expert_noise_max",
        type=float,
        default=DEFAULT_EXPERT_NOISE_MAX,
        help="Maximum per-episode Gaussian noise scale for expert actions",
    )
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--temp_dir", type=str, default=DEFAULT_TEMP_DIR)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=(
            "MetaWorld task names to collect. Accepts either space-separated "
            "or comma-separated values, e.g. --tasks door-open-v3 reach-v3 "
            "or --tasks door-open-v3,reach-v3. Defaults to all MT50 tasks."
        ),
    )
    parser.add_argument(
        "--randomize_every_reset",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RANDOMIZE_EVERY_RESET,
        help="If enabled, sample a fresh object/goal placement on every reset instead of using the frozen MT50 task layout",
    )
    parser.add_argument(
        "--randomize_hand_start",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RANDOMIZE_HAND_START,
        help=(
            "If enabled, sample the robot hand start position uniformly from "
            "the task's hand_low/hand_high workspace before every episode reset."
        ),
    )
    parser.add_argument(
        "--random_walk_direction_stddev",
        type=float,
        default=DEFAULT_RANDOM_WALK_DIRECTION_STDDEV,
        help="Angular perturbation scale for the random-walk direction policy",
    )
    parser.add_argument(
        "--random_walk_gravity_strength",
        type=float,
        default=DEFAULT_RANDOM_WALK_GRAVITY_STRENGTH,
        help="Soft pull toward the workspace center for the random-walk direction policy",
    )
    parser.add_argument(
        "--random_walk_levy_alpha",
        type=float,
        default=DEFAULT_RANDOM_WALK_LEVY_ALPHA,
        help="Pareto alpha for the random-walk step-length policy",
    )
    parser.add_argument(
        "--random_walk_min_step",
        type=float,
        default=DEFAULT_RANDOM_WALK_MIN_STEP,
        help="Minimum random-walk step magnitude before clipping into MetaWorld action space",
    )
    parser.add_argument(
        "--random_walk_max_step",
        type=float,
        default=DEFAULT_RANDOM_WALK_MAX_STEP,
        help="Maximum random-walk step magnitude before clipping into MetaWorld action space",
    )
    parser.add_argument(
        "--explore-qpos",
        action="store_true",
        help="Print joint positions during collection",
    )
    args = parser.parse_args()
    if args.expert_noise_min < 0.0 or args.expert_noise_max < 0.0:
        parser.error("expert noise bounds must be non-negative")
    if args.expert_noise_min > args.expert_noise_max:
        parser.error("expert_noise_min must be less than or equal to expert_noise_max")
    if args.random_walk_direction_stddev <= 0.0:
        parser.error("random_walk_direction_stddev must be positive")
    if args.random_walk_gravity_strength < 0.0:
        parser.error("random_walk_gravity_strength must be non-negative")
    if args.random_walk_levy_alpha <= 0.0:
        parser.error("random_walk_levy_alpha must be positive")
    if args.random_walk_min_step <= 0.0:
        parser.error("random_walk_min_step must be positive")
    if args.random_walk_max_step <= args.random_walk_min_step:
        parser.error("random_walk_max_step must be greater than random_walk_min_step")

    print(f"Initializing Meta-World MT50 (Main Process)...", flush=True)

    if args.explore_qpos:
        explore_qpos()
    else:
        full_data_collection(args)


if __name__ == "__main__":
    main()
