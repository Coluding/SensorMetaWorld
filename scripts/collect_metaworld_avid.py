import metaworld
import metaworld.policies as mw_policies
import numpy as np
import cv2
import random
import argparse
import multiprocessing
import os
os.environ.setdefault("MUJOCO_GL", "egl")
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
DEFAULT_DATASET_PATH = "data/metaworld/mw_test_zoom.hdf5"
DEFAULT_TEMP_DIR = "data/metaworld/temp/"
DEFAULT_NUM_EPISODES = 1
DEFAULT_EXPERT_NOISE_MIN = 0.1
DEFAULT_EXPERT_NOISE_MAX = 0.5
DEFAULT_RANDOMIZE_EVERY_RESET = True
DEFAULT_RANDOM_WALK_DIRECTION_STDDEV = 0.35
DEFAULT_RANDOM_WALK_GRAVITY_STRENGTH = 0.01
DEFAULT_RANDOM_WALK_LEVY_ALPHA = 1.35
DEFAULT_RANDOM_WALK_MIN_STEP = 1.0
DEFAULT_RANDOM_WALK_MAX_STEP = 4.0
MAX_STEPS = 500
DEFAULT_CAMERA_NAMES = ("corner2", )
AVAILABLE_CAMERA_NAMES = (
    "corner",
    "corner2",
    "corner3",
    "corner4",
    "topview",
    "behindGripper",
    "gripperPOV",
)
RGB_IMAGE_WIDTH = 128
RGB_IMAGE_HEIGHT = 128
DEPTH_IMAGE_WIDTH = 128
DEPTH_IMAGE_HEIGHT = 128
TACTILE_RESOLUTION = 64
FRAME_AGGREGATIONS = ("first", "mean", "max", "sum")


def _aggregate_window(values: list, mode: str) -> np.ndarray:
    """Reduce a list of per-tick sensor readings into a single frame.

    ``first`` returns the value at the window boundary (no aggregation);
    ``mean``/``max``/``sum`` reduce element-wise across the stride window.
    Boolean readings (e.g. bool_contact) collapse to "any in window" for
    max/sum/mean once cast back to bool downstream.
    """
    arr = np.stack(values, axis=0)
    if mode == "first":
        return arr[0]
    if mode == "mean":
        return arr.mean(axis=0)
    if mode == "max":
        return arr.max(axis=0)
    if mode == "sum":
        return arr.sum(axis=0)
    raise ValueError(f"Unknown frame_aggregation: {mode}")


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


def _set_randomized_resets(env, enabled: bool) -> None:
    """Allow fresh object/goal placement samples on every reset."""
    env._freeze_rand_vec = not enabled


def _build_episode_policy_schedule(num_episodes: int) -> list[str]:
    """Return a shuffled 50/50-ish split of noisy expert and random walk."""
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


def _resolve_camera_id(env, camera_name: str) -> int:
    """Resolve a MuJoCo camera name to its integer id for the given env."""
    try:
        return int(env.unwrapped.model.camera(camera_name).id)
    except KeyError:
        available = [
            env.unwrapped.model.camera(i).name
            for i in range(env.unwrapped.model.ncam)
        ]
        raise RuntimeError(
            f"Camera '{camera_name}' not found in MuJoCo model. "
            f"Available cameras: {available}"
        )


def _apply_camera_zoom(env, camera_names, zoom_factor: float) -> None:
    """Optically zoom the given cameras by narrowing their field of view.

    Dividing a MuJoCo camera's ``fovy`` by ``zoom_factor`` concentrates the
    full render resolution onto a smaller central region, so objects appear
    larger and finer detail is resolved with no interpolation/cropping loss.
    Because RGB and depth render through the same model camera, both views are
    zoomed identically; depth *distances* are unaffected (only the projection
    changes). ``zoom_factor == 1.0`` is a no-op.
    """
    if zoom_factor == 1.0:
        return
    model = env.unwrapped.model
    for camera_name in camera_names:
        camera_id = _resolve_camera_id(env, camera_name)
        model.cam_fovy[camera_id] = model.cam_fovy[camera_id] / zoom_factor


def _render_rgb_frame(env, camera_id: int) -> np.ndarray:
    """Render one RGB frame from the given camera id."""
    frame = None
    renderer = getattr(env.unwrapped, "mujoco_renderer", None)

    if renderer is not None:
        get_viewer = getattr(renderer, "_get_viewer", None)
        if callable(get_viewer):
            try:
                viewer = get_viewer(render_mode="rgb_array")
                # The depth sensor renders through this same offscreen viewer
                # (with render_mode="depth_array"). Make the context current
                # again before reading pixels or MuJoCo can return all-black
                # frames after the first depth render.
                viewer.make_context_current()
                frame = viewer.render(
                    render_mode="rgb_array",
                    camera_id=camera_id,
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
    depth_sensors: dict[str, DepthCameraSensor],
    tactile_sensor: TactileDigitSensor,
    force_torque_sensor: ForceTorqueSensor,
    camera_names: list[str],
    max_steps: int = MAX_STEPS,
    reset_seed: int | None = None,
    policy_name: str = "policy",
    frame_stride: int = 1,
    frame_aggregation: str = "first",
    action_repeat: int = 1,
):
    """
    Runs one episode and returns a DICTIONARY.
    Always returns data, even if expert fails.

    The single rollout is rendered from every camera in ``camera_names`` (each
    gets its own RGB ``pixels`` and ``depth`` frame); ``depth_sensors`` maps
    each camera name to a DepthCameraSensor bound to it. The camera-independent
    sensors (proprio, tactile, force_torque, gripper, ee/object xyz,
    bool_contact, action) are collected once and shared across all views.

    Two distinct mechanisms downsample the saved trajectory:

    * ``frame_stride``: the policy is re-queried with a fresh observation on
      every control tick (so the controller reacts at full rate and dynamics
      stay smooth), but only every ``frame_stride``-th frame is stored.
    * ``action_repeat``: the policy is queried once per block of
      ``action_repeat`` ticks; that single action is *committed* (re-applied
      unchanged) for the whole block, and only the block's initial frame is
      stored. This is classic action-repeat / frame-skip — it produces larger
      inter-frame motion (better dynamics for video prediction) because the
      controller cannot correct mid-block.

    They are mutually exclusive (enforced by the CLI); whichever is >1 sets the
    storage window length. Numeric sensors are buffered every tick and reduced
    per window via ``frame_aggregation`` (first/mean/max/sum) — with
    ``action_repeat`` the action is constant over the window, so ``first``
    yields the committed action. RGB/depth are not aggregated: one frame per
    camera is sampled at each window boundary, avoiding image blurring and
    per-tick render cost. ``frame_stride=1`` and ``action_repeat=1`` keep the
    original behaviour regardless of ``frame_aggregation``.
    """
    obs, _ = env.reset(seed=reset_seed)
    for depth_sensor in depth_sensors.values():
        depth_sensor.reset(env)
    tactile_sensor.reset(env)
    force_torque_sensor.reset(env)
    _initialize_policy_reference_position(env, policy, obs)

    rgb_camera_ids = {cam: _resolve_camera_id(env, cam) for cam in camera_names}
    rgb_frames = {cam: [] for cam in camera_names}
    depth_frames = {cam: [] for cam in camera_names}
    tactile_frames = []
    proprioception_list = []
    gripper_list = []
    ee_xyz_list = []
    object_1_xyz_list = []
    object_2_xyz_list = []
    bool_contact_list = []
    force_torque_list = []
    actions = []
    expert_actions_raw = []
    expert_actions_clipped = []
    action_noises = []

    # Per-tick buffers for the current frame_stride window. Numeric sensors are
    # accumulated every control tick and reduced into one frame per window via
    # `frame_aggregation`; RGB/depth images are sampled once at each window
    # boundary (see below) rather than aggregated.
    window = {
        "tactile": [],
        "proprio": [],
        "gripper": [],
        "ee_xyz": [],
        "object_1_xyz": [],
        "object_2_xyz": [],
        "bool_contact": [],
        "force_torque": [],
        "action": [],
        "expert_action_raw": [],
        "expert_action_clipped": [],
        "action_noise": [],
    }

    def _flush_window():
        """Reduce the buffered window into one frame per numeric sensor."""
        if not window["action"]:
            return
        tactile_frames.append(_aggregate_window(window["tactile"], frame_aggregation))
        proprioception_list.append(
            _aggregate_window(window["proprio"], frame_aggregation)
        )
        gripper_list.append(_aggregate_window(window["gripper"], frame_aggregation))
        ee_xyz_list.append(_aggregate_window(window["ee_xyz"], frame_aggregation))
        object_1_xyz_list.append(
            _aggregate_window(window["object_1_xyz"], frame_aggregation)
        )
        object_2_xyz_list.append(
            _aggregate_window(window["object_2_xyz"], frame_aggregation)
        )
        bool_contact_list.append(
            _aggregate_window(window["bool_contact"], frame_aggregation)
        )
        force_torque_list.append(
            _aggregate_window(window["force_torque"], frame_aggregation)
        )
        actions.append(_aggregate_window(window["action"], frame_aggregation))
        if window["expert_action_raw"]:
            expert_actions_raw.append(
                _aggregate_window(window["expert_action_raw"], frame_aggregation)
            )
        if window["expert_action_clipped"]:
            expert_actions_clipped.append(
                _aggregate_window(window["expert_action_clipped"], frame_aggregation)
            )
        if window["action_noise"]:
            action_noises.append(
                _aggregate_window(window["action_noise"], frame_aggregation)
            )
        for buf in window.values():
            buf.clear()

    success = False

    # Whichever of frame_stride / action_repeat is >1 defines the storage
    # window (they are mutually exclusive). With action_repeat the policy
    # commits one action per window instead of being re-queried every tick.
    stride = max(frame_stride, action_repeat)
    commit_action = action_repeat > 1
    cached_action = None

    for step_idx in range(max_steps):
        # Force-torque low-pass filter must see every control tick.
        force_torque_sensor.update(env)

        is_window_start = (step_idx % stride) == 0
        if is_window_start:
            # Start of a new window: flush the previous window, then sample the
            # images that represent this window at its boundary, once per camera.
            _flush_window()
            for cam in camera_names:
                rgb_frames[cam].append(
                    _render_rgb_frame(env, rgb_camera_ids[cam])
                )
                depth_sensors[cam].update(env)
                depth_frames[cam].append(
                    depth_sensors[cam].get_depth_as_image().copy()
                )

        tactile_sensor.update(env)
        left_tactile, right_tactile = tactile_sensor.get_finger_images()
        window["tactile"].append(np.stack((left_tactile, right_tactile), axis=0))
        window["proprio"].append(env.data.qpos[:7].copy())
        window["gripper"].append(np.float32(obs[3]))
        window["ee_xyz"].append(
            np.asarray(env.get_endeff_pos(), dtype=np.float32).copy()
        )
        obj1, obj2 = _get_object_xyzs(env)
        window["object_1_xyz"].append(obj1)
        window["object_2_xyz"].append(obj2)
        window["bool_contact"].append(_gripper_touching_any(env))
        window["force_torque"].append(
            np.asarray(force_torque_sensor.read(), dtype=np.float32)
        )

        # In action_repeat mode the policy is queried only at the window start
        # (using that block's initial observation) and the action is held
        # constant for the rest of the block; otherwise it reacts every tick.
        if (not commit_action) or (cached_action is None) or is_window_start:
            cached_action = np.asarray(policy.get_action(obs), dtype=np.float32)
        action = cached_action
        window["action"].append(action)
        if getattr(policy, "last_expert_action_raw", None) is not None:
            window["expert_action_raw"].append(policy.last_expert_action_raw.copy())
        if getattr(policy, "last_expert_action_clipped", None) is not None:
            window["expert_action_clipped"].append(
                policy.last_expert_action_clipped.copy()
            )
        if getattr(policy, "last_noise", None) is not None:
            window["action_noise"].append(policy.last_noise.copy())
        obs, _reward, terminated, truncated, info = env.step(action)

        if info.get("success", 0.0) > 0.0:
            success = True

        if terminated or truncated:
            break

    # Reduce the final (possibly partial) window.
    _flush_window()

    if not success and policy_name != "random_walk":
        print(
            f"  [Warn] {policy_name} rollout failed on {task_name} but saving anyway.",
            flush=True,
        )

    # Per-camera image views (pixels + depth), keyed by camera name.
    cameras = {
        cam: {
            "pixels": np.array(rgb_frames[cam], dtype=np.uint8),
            "depth": np.array(depth_frames[cam], dtype=np.float32),
        }
        for cam in camera_names
    }
    # Camera-independent sensors, shared across all views.
    sensors = {
        "proprio": np.array(proprioception_list, dtype=np.float32),
        "tactile": np.array(tactile_frames, dtype=np.float32),
        "force_torque": np.array(force_torque_list, dtype=np.float32),
        "gripper": np.array(gripper_list, dtype=np.float32),
        "ee_xyz": np.array(ee_xyz_list, dtype=np.float32),
        "object_1_xyz": np.array(object_1_xyz_list, dtype=np.float32),
        "object_2_xyz": np.array(object_2_xyz_list, dtype=np.float32),
        "bool_contact": np.array(bool_contact_list, dtype=np.bool_),
        "action": np.array(actions, dtype=np.float32),
    }
    if len(expert_actions_raw) == len(actions):
        sensors["expert_action_raw"] = np.array(expert_actions_raw, dtype=np.float32)
    if len(expert_actions_clipped) == len(actions):
        sensors["expert_action_clipped"] = np.array(
            expert_actions_clipped, dtype=np.float32
        )
    if len(action_noises) == len(actions):
        sensors["action_noise"] = np.array(action_noises, dtype=np.float32)
    return {"cameras": cameras, "sensors": sensors}


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
            task_group.attrs["frame_stride"] = int(args.frame_stride)
            task_group.attrs["frame_aggregation"] = str(args.frame_aggregation)
            task_group.attrs["action_repeat"] = int(args.action_repeat)
            task_group.attrs["zoom_factor"] = float(args.zoom_factor)
            task_group.attrs["camera_names"] = list(args.cameras)

            episode_global_idx = 0
            env = env_cls(render_mode="rgb_array", camera_name=args.cameras[0])
            _apply_camera_zoom(env, args.cameras, args.zoom_factor)
            depth_sensors = {
                cam: DepthCameraSensor(
                    camera_name=cam,
                    height=DEPTH_IMAGE_HEIGHT,
                    width=DEPTH_IMAGE_WIDTH,
                    normalize=False,
                )
                for cam in args.cameras
            }
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

            episode_policy_schedule = _build_episode_policy_schedule(args.num_episodes)
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
                    depth_sensors=depth_sensors,
                    tactile_sensor=tactile_sensor,
                    force_torque_sensor=force_torque_sensor,
                    camera_names=list(args.cameras),
                    policy_name=policy_metadata["policy_type"],
                    frame_stride=args.frame_stride,
                    frame_aggregation=args.frame_aggregation,
                    action_repeat=args.action_repeat,
                )

                episode_key = f"episode_{episode_global_idx}"

                # Per-camera image views: <env>/<camera>/episode_N/{pixels, depth}
                for cam, cam_data in data_dict["cameras"].items():
                    cam_group = task_group.require_group(cam)
                    cam_ep_group = cam_group.create_group(episode_key)
                    cam_ep_group.create_dataset(
                        "pixels", data=cam_data["pixels"], compression="gzip"
                    )
                    cam_ep_group.create_dataset(
                        "depth", data=cam_data["depth"], compression="gzip"
                    )

                # Shared, camera-independent sensors: <env>/sensors/episode_N/*
                sensors_group = task_group.require_group("sensors")
                sensor_ep_group = sensors_group.create_group(episode_key)
                sensor_data = data_dict["sensors"]
                sensor_ep_group.create_dataset("proprio", data=sensor_data["proprio"])
                sensor_ep_group.create_dataset(
                    "tactile", data=sensor_data["tactile"], compression="gzip"
                )
                sensor_ep_group.create_dataset("gripper", data=sensor_data["gripper"])
                sensor_ep_group.create_dataset("ee_xyz", data=sensor_data["ee_xyz"])
                sensor_ep_group.create_dataset(
                    "object_1_xyz", data=sensor_data["object_1_xyz"]
                )
                sensor_ep_group.create_dataset(
                    "object_2_xyz", data=sensor_data["object_2_xyz"]
                )
                sensor_ep_group.create_dataset(
                    "bool_contact", data=sensor_data["bool_contact"]
                )
                sensor_ep_group.create_dataset(
                    "force_torque", data=sensor_data["force_torque"]
                )
                sensor_ep_group.create_dataset("action", data=sensor_data["action"])
                for optional_key in (
                    "expert_action_raw",
                    "expert_action_clipped",
                    "action_noise",
                ):
                    if optional_key in sensor_data:
                        sensor_ep_group.create_dataset(
                            optional_key, data=sensor_data[optional_key]
                        )

                sensor_ep_group.attrs["policy_type"] = policy_metadata["policy_type"]
                sensor_ep_group.attrs["task_name"] = env_name
                sensor_ep_group.attrs["randomize_every_reset"] = bool(
                    args.randomize_every_reset
                )
                for key, value in policy_metadata.items():
                    if key == "policy_type":
                        continue
                    sensor_ep_group.attrs[key] = value

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
        "--num_episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Total episodes to collect per environment across noisy expert and random walk behaviors",
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
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Record one frame every N control steps. The policy still steps "
        "every tick, but only every Nth frame is saved, lowering the effective "
        "fps of the saved trajectory by this factor (1 = record every step).",
    )
    parser.add_argument(
        "--frame_aggregation",
        type=str,
        default="first",
        choices=FRAME_AGGREGATIONS,
        help="How to reduce numeric sensors over each storage window: "
        "'first' samples the boundary value (no aggregation), 'mean'/'max'/'sum' "
        "reduce element-wise. RGB/depth images are always sampled, never aggregated.",
    )
    parser.add_argument(
        "--action_repeat",
        type=int,
        default=1,
        help="Action-repeat / frame-skip factor. The policy is queried once per "
        "block of N control ticks and that single action is re-applied unchanged "
        "for the whole block; only the block's initial frame is stored. This "
        "yields larger inter-frame motion (better dynamics for video prediction). "
        "Mutually exclusive with --frame_stride (1 = disabled).",
    )
    parser.add_argument(
        "--zoom_factor",
        type=float,
        default=1.0,
        help="Optical zoom for all rendered cameras: divides each camera's "
        "field-of-view (fovy) by this factor so the view narrows onto the scene "
        "center, enlarging objects and resolving finer detail with no resolution "
        "loss. Applies to both RGB and depth (>1 zooms in, 1.0 = disabled).",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=list(DEFAULT_CAMERA_NAMES),
        choices=AVAILABLE_CAMERA_NAMES,
        metavar="CAMERA",
        help="One or more camera names to render each rollout from. Each camera "
        "produces its own pixels+depth under <env>/<camera>/episode_N/; the "
        f"default is {list(DEFAULT_CAMERA_NAMES)}. Choices: {list(AVAILABLE_CAMERA_NAMES)}.",
    )
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--temp_dir", type=str, default=DEFAULT_TEMP_DIR)
    parser.add_argument(
        "--randomize_every_reset",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RANDOMIZE_EVERY_RESET,
        help="If enabled, sample a fresh object/goal placement on every reset instead of using the frozen MT50 task layout",
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
    if args.frame_stride < 1:
        parser.error("frame_stride must be a positive integer")
    if args.action_repeat < 1:
        parser.error("action_repeat must be a positive integer")
    if args.frame_stride > 1 and args.action_repeat > 1:
        parser.error(
            "--frame_stride and --action_repeat are mutually exclusive; "
            "set at most one of them above 1"
        )
    if args.zoom_factor <= 0.0:
        parser.error("zoom_factor must be positive")
    if len(set(args.cameras)) != len(args.cameras):
        parser.error("--cameras must not contain duplicate camera names")

    print(f"Initializing Meta-World MT50 (Main Process)...", flush=True)

    if args.explore_qpos:
        explore_qpos()
    else:
        full_data_collection(args)


if __name__ == "__main__":
    main()
