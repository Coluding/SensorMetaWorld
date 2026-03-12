import metaworld
import numpy as np
import h5py
import cv2
import random
import argparse
import multiprocessing
import os
import glob
from math import ceil
from typing import Any
import mujoco
from metaworld.policies import *
from metaworld_policies import (
    MetaWorldPolicy,
    RandomWalk,
    NoisyExpertPolicy,
)

from pathlib import Path

# Defaults
DEFAULT_DATASET_PATH = "data/metaworld/metaworld_corner.hdf5"
DEFAULT_TEMP_DIR = "data/metaworld/temp/"
DEFAULT_EPISODES_EXPERT = 1
DEFAULT_EPISODES_RANDOM = 1
MAX_STEPS = 256
# Image size for rendering (width, height) - Dynamicrafter default is 320x512
IMG_SIZE = (256, 256)
# Kinematics-only IK defaults (no dynamics / no collision simulation).
KIN_IK_ITERS = 12
KIN_IK_STEP_SIZE = 0.6
KIN_IK_DAMPING = 1e-4
KIN_IK_TOL = 1e-4


def get_policy_for_task(task_name):
    """
    Finds the correct scripted policy for a given task name.
    """
    base_name = task_name.split("-v")[0]
    camel_case = "".join(x.title() for x in base_name.split("-"))
    policy_name = f"Sawyer{camel_case}V3Policy"

    if policy_name in globals():
        return globals()[policy_name]()
    return None


def _build_arm_kinematics_cache(env) -> dict[str, Any]:
    """Pre-compute index lookups for Sawyer arm joints and TCP sites."""
    model = env.model
    arm_joint_names = [f"right_j{i}" for i in range(7)]
    qpos_indices = []
    dof_indices = []
    joint_lower = []
    joint_upper = []

    for joint_name in arm_joint_names:
        joint_id = model.joint(joint_name).id
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        dof_indices.append(int(model.jnt_dofadr[joint_id]))
        joint_lower.append(float(model.jnt_range[joint_id, 0]))
        joint_upper.append(float(model.jnt_range[joint_id, 1]))

    return {
        "qpos_indices": np.array(qpos_indices, dtype=np.int64),
        "dof_indices": np.array(dof_indices, dtype=np.int64),
        "joint_lower": np.array(joint_lower, dtype=np.float64),
        "joint_upper": np.array(joint_upper, dtype=np.float64),
        "right_site_id": env.model.site("rightEndEffector").id,
        "left_site_id": env.model.site("leftEndEffector").id,
    }


def _predict_kinematics_no_sim(
    env,
    kin_data: mujoco.MjData,
    kin_cache: dict[str, Any],
    action: np.ndarray,
) -> np.ndarray:
    """Predict next arm joint positions from action using kinematics only.

    This intentionally does not run MuJoCo dynamics (`env.step` / `mj_step`).
    """
    model = env.model
    arm_qpos_idx = kin_cache["qpos_indices"]
    arm_dof_idx = kin_cache["dof_indices"]
    lower = kin_cache["joint_lower"]
    upper = kin_cache["joint_upper"]
    right_site_id = kin_cache["right_site_id"]
    left_site_id = kin_cache["left_site_id"]

    # Copy current state into standalone kinematics data.
    kin_data.qpos[:] = env.data.qpos
    kin_data.qvel[:] = env.data.qvel
    if model.nmocap > 0:
        kin_data.mocap_pos[:] = env.data.mocap_pos
        kin_data.mocap_quat[:] = env.data.mocap_quat

    mujoco.mj_fwdPosition(model, kin_data)

    action_xyz = np.clip(np.asarray(action[:3], dtype=np.float64), -1.0, 1.0)
    pos_delta = action_xyz * float(env.action_scale)

    if model.nmocap > 0:
        target_tcp = np.clip(
            kin_data.mocap_pos[0] + pos_delta,
            env.mocap_low,
            env.mocap_high,
        )
    else:
        right_pos = kin_data.site_xpos[right_site_id]
        left_pos = kin_data.site_xpos[left_site_id]
        target_tcp = 0.5 * (right_pos + left_pos) + pos_delta

    qpos_work = kin_data.qpos.copy()
    jac_right = np.zeros((3, model.nv), dtype=np.float64)
    jac_left = np.zeros((3, model.nv), dtype=np.float64)

    for _ in range(KIN_IK_ITERS):
        kin_data.qpos[:] = qpos_work
        mujoco.mj_fwdPosition(model, kin_data)

        right_pos = kin_data.site_xpos[right_site_id]
        left_pos = kin_data.site_xpos[left_site_id]
        tcp_pos = 0.5 * (right_pos + left_pos)
        pos_error = target_tcp - tcp_pos

        if np.linalg.norm(pos_error) <= KIN_IK_TOL:
            break

        jac_right.fill(0.0)
        jac_left.fill(0.0)
        mujoco.mj_jacSite(model, kin_data, jac_right, None, right_site_id)
        mujoco.mj_jacSite(model, kin_data, jac_left, None, left_site_id)

        jac_tcp = 0.5 * (jac_right + jac_left)
        j_arm = jac_tcp[:, arm_dof_idx]
        # Damped least-squares IK update in joint space.
        jj_t = j_arm @ j_arm.T
        damped = jj_t + KIN_IK_DAMPING * np.eye(3, dtype=np.float64)
        dq = j_arm.T @ np.linalg.solve(damped, pos_error)

        qpos_work[arm_qpos_idx] += KIN_IK_STEP_SIZE * dq
        qpos_work[arm_qpos_idx] = np.clip(qpos_work[arm_qpos_idx], lower, upper)

    return qpos_work[arm_qpos_idx].copy()


def collect_episode(env, policy: MetaWorldPolicy, policy_type: str, task_name: str):
    """
    Runs one episode and returns a DICTIONARY.
    Always returns data, even if expert fails.
    """
    obs, _ = env.reset()
    if hasattr(policy, "set_reference_position"):
        policy.set_reference_position(obs[0:3])

    frames = []
    actions = []

    # State buffers
    robot_xyz_list, gripper_list = [], []
    proprioception_list = []
    kinematics_prediction_list = []
    obj1_xyz_list, obj1_quat_list = [], []
    obj2_xyz_list, obj2_quat_list = [], []

    kin_data = mujoco.MjData(env.model)
    kin_cache = _build_arm_kinematics_cache(env)

    success = False

    for _ in range(MAX_STEPS):
        # Render
        frame = env.render()
        if frame is None:
            frame = env.render(offscreen=True, resolution=IMG_SIZE)
        else:
            frame = cv2.resize(frame, IMG_SIZE)
        frame = cv2.flip(frame, 0)
        frames.append(frame)

        # Split State
        robot_xyz_list.append(obs[0:3])
        gripper_list.append(obs[3])
        obj1_xyz_list.append(obs[4:7])
        obj1_quat_list.append(obs[7:11])
        obj2_xyz_list.append(obs[11:14])
        obj2_quat_list.append(obs[14:18])

        action = policy.get_action(obs)

        # Proprioception is the first 7 elements of qpos (robot joints, no gripper state).
        proprioception_list.append(env.data.qpos[:7].copy())
        # Kinematics-only joint prediction for the SAME action (no dynamics/collision step).
        kinematics_prediction_list.append(
            _predict_kinematics_no_sim(env, kin_data, kin_cache, action)
        )

        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)

        if info.get("success", 0.0) > 0.0:
            success = True

        if terminated or truncated:
            break

    if policy_type == "expert" and not success:
        print(f"  [Warn] Expert failed on {task_name} but saving anyway.", flush=True)

    return {
        "images": np.array(frames, dtype=np.uint8),
        "robot_xyz": np.array(robot_xyz_list, dtype=np.float32),
        "proprioception": np.array(proprioception_list, dtype=np.float32),
        "kinematics_prediction": np.array(kinematics_prediction_list, dtype=np.float32),
        "gripper": np.array(gripper_list, dtype=np.float32),
        "obj1_xyz": np.array(obj1_xyz_list, dtype=np.float32),
        "obj1_quat": np.array(obj1_quat_list, dtype=np.float32),
        "obj2_xyz": np.array(obj2_xyz_list, dtype=np.float32),
        "obj2_quat": np.array(obj2_quat_list, dtype=np.float32),
        "action": np.array(actions, dtype=np.float32),
        "success": success,
        "policy": policy_type,
    }


def worker_process(worker_id, env_names, args):
    """
    Worker function to process a subset of environments.
    """
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

    camera_views = ["corner", "corner2", "corner3", "topview"]

    with h5py.File(temp_path, "w") as f:
        # We finish one environment completely before loading the assets for the next.
        for i, env_name in enumerate(env_names):
            print(
                f"[Worker {worker_id}] Processing {env_name} ({i + 1}/{len(env_names)})",
                flush=True,
            )

            # Create the Group ONCE per environment
            task_group = f.create_group(env_name)
            task_group.attrs["task_name"] = env_name

            env_cls = mt50.train_classes[env_name]
            expert_policy = get_policy_for_task(env_name)
            available_tasks = tasks_by_env.get(env_name, [])

            # Global counter for this environment (across all cameras)
            episode_global_idx = 0

            # Iterate Cameras
            for cam in camera_views:
                # We must re-init the env to change the camera in standard MetaWorld
                # (This is safer than dynamic rendering which can be version-dependent)
                env = env_cls(render_mode="rgb_array", camera_name=cam)

                modes = [("random", args.num_episodes_random)]
                if expert_policy is not None:
                    modes.append(("expert", args.num_episodes_expert))

                for mode, target_count in modes:
                    collected_count = 0
                    while collected_count < target_count:
                        random_task = random.choice(available_tasks)
                        env.set_task(random_task)

                        if mode == "expert":
                            policy = NoisyExpertPolicy(
                                expert_policy, noise_scale=args.expert_noise
                            )
                        else:
                            policy = RandomWalk(
                                direction_policy=args.random_direction_policy,
                                step_length_policy=args.random_step_length_policy,
                            )

                        data_dict = collect_episode(env, policy, mode, env_name)

                        # We use a flat structure: episode_0, episode_1, but tag them with the camera name.
                        ep_group = task_group.create_group(
                            f"episode_{episode_global_idx}"
                        )

                        ep_group.create_dataset(
                            "images", data=data_dict["images"], compression="gzip"
                        )
                        ep_group.create_dataset(
                            "robot_xyz", data=data_dict["robot_xyz"]
                        )
                        ep_group.create_dataset(
                            "proprioception", data=data_dict["proprioception"]
                        )
                        ep_group.create_dataset(
                            "kinematics_prediction",
                            data=data_dict["kinematics_prediction"],
                        )
                        ep_group.create_dataset("gripper", data=data_dict["gripper"])
                        ep_group.create_dataset("obj1_xyz", data=data_dict["obj1_xyz"])
                        ep_group.create_dataset(
                            "obj1_quat", data=data_dict["obj1_quat"]
                        )
                        ep_group.create_dataset("obj2_xyz", data=data_dict["obj2_xyz"])
                        ep_group.create_dataset(
                            "obj2_quat", data=data_dict["obj2_quat"]
                        )
                        ep_group.create_dataset("action", data=data_dict["action"])

                        ep_group.attrs["success"] = data_dict["success"]
                        ep_group.attrs["policy_type"] = data_dict["policy"]
                        ep_group.attrs["task_name"] = env_name

                        ep_group.attrs["camera_name"] = cam

                        collected_count += 1
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
    mt50 = metaworld.MT50()
    all_env_names = list(mt50.train_classes.keys())
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
        f"Spawning {len(chunks)} workers for {len(all_env_names)} environments...",
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
        "--num_episodes_expert", type=int, default=DEFAULT_EPISODES_EXPERT
    )
    parser.add_argument(
        "--expert_noise",
        type=float,
        default=0.1,
        help="Stddev of Gaussian noise for expert actions",
    )
    parser.add_argument(
        "--num_episodes_random", type=int, default=DEFAULT_EPISODES_RANDOM
    )
    parser.add_argument(
        "--random_direction_policy",
        type=str,
        default="gravity",
        help="Random walk policy for random episodes",
        choices=["uniform", "gaussian", "gravity"],
    )
    parser.add_argument(
        "--random_step_length_policy",
        type=str,
        default="levy",
        help="Step length policy for random walk",
        choices=["constant", "levy"],
    )
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--temp_dir", type=str, default=DEFAULT_TEMP_DIR)
    parser.add_argument(
        "--explore-qpos",
        action="store_true",
        help="Print joint positions during collection",
    )
    args = parser.parse_args()

    print(f"Initializing Meta-World MT50 (Main Process)...", flush=True)

    if args.explore_qpos:
        explore_qpos()
    else:
        full_data_collection(args)


if __name__ == "__main__":
    main()
