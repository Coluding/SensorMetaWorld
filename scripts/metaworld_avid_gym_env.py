"""Gymnasium adapter for the AVID-style MetaWorld observations.

This mirrors the sensor setup used by ``collect_metaworld_avid.py`` while
presenting a normal Gymnasium ``Env`` interface for planners and RL code.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import metaworld
import numpy as np
from gymnasium import spaces

from metaworld.sensors.force_torque_sensor import ForceTorqueSensor
from metaworld.sensors.tactile_digit_sensor import TactileDigitSensor
from metaworld.sensors.visual import DepthCameraSensor
from metaworld.types import Task


DEFAULT_MAX_STEPS = 200
DEFAULT_CAMERA_NAME = "corner2"
DEFAULT_RGB_SIZE = (128, 128)
DEFAULT_DEPTH_SIZE = (128, 128)
DEFAULT_TACTILE_RESOLUTION = 64

try:
    import cv2 as cv
except ImportError:
    cv = None


def _resize_array(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize with OpenCV when available, otherwise nearest-neighbor NumPy."""
    if cv is not None:
        return cv.resize(image, (width, height))
    if image.shape[0] == height and image.shape[1] == width:
        return image

    y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
    if image.ndim == 2:
        return image[y_idx][:, x_idx]
    return image[y_idx][:, x_idx, ...]


def _set_randomized_resets(env: gym.Env, enabled: bool) -> None:
    """Allow fresh object/goal placement samples on every reset."""
    env.unwrapped._freeze_rand_vec = not enabled


def _render_rgb_frame(env: gym.Env, width: int, height: int) -> np.ndarray:
    """Render one RGB frame from the environment's configured camera."""
    frame = None
    renderer = getattr(env.unwrapped, "mujoco_renderer", None)

    if renderer is not None:
        get_viewer = getattr(renderer, "_get_viewer", None)
        if callable(get_viewer):
            try:
                viewer = get_viewer(render_mode="rgb_array")
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
        frame = env.render(offscreen=True, resolution=(width, height))

    frame = _resize_array(frame, width=width, height=height)
    return np.flipud(frame).astype(np.uint8)


def _get_object_xyzs(env: gym.Env) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Return two object slots and validity masks without NaNs."""
    obj1 = np.zeros(3, dtype=np.float32)
    obj2 = np.zeros(3, dtype=np.float32)
    obj1_present = False
    obj2_present = False

    try:
        obj_pos = np.asarray(
            env.unwrapped._get_pos_objects(),
            dtype=np.float32,
        ).reshape(-1)
    except Exception:
        return obj1, obj2, obj1_present, obj2_present

    if obj_pos.size < 3:
        return obj1, obj2, obj1_present, obj2_present

    count = obj_pos.size // 3
    obj_pos = obj_pos[: count * 3]
    objs = np.split(obj_pos, count)
    obj1 = objs[0].copy()
    obj1_present = bool(np.all(np.isfinite(obj1)))
    if count > 1:
        obj2 = objs[1].copy()
        obj2_present = bool(np.all(np.isfinite(obj2)))

    if not obj1_present:
        obj1.fill(0.0)
    if not obj2_present:
        obj2.fill(0.0)

    return obj1, obj2, obj1_present, obj2_present


def _gripper_touching_any(env: gym.Env) -> bool:
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
            return bool(env.unwrapped.touching_main_object)
        except Exception:
            return False


class MetaWorldAvidEnv(gym.Env):
    """Gymnasium Env exposing MetaWorld state plus AVID sensor observations."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env_name: str,
        *,
        seed: int | None = None,
        task: Task | None = None,
        task_index: int | None = None,
        sample_task_on_reset: bool = False,
        randomize_every_reset: bool = True,
        max_episode_steps: int = DEFAULT_MAX_STEPS,
        camera_name: str = DEFAULT_CAMERA_NAME,
        rgb_size: tuple[int, int] = DEFAULT_RGB_SIZE,
        depth_size: tuple[int, int] = DEFAULT_DEPTH_SIZE,
        tactile_resolution: int = DEFAULT_TACTILE_RESOLUTION,
        tactile_photometric: bool = False,
        tactile_normalize: bool = False,
        depth_normalize: bool = False,
    ) -> None:
        super().__init__()
        self.env_name = env_name
        self.max_episode_steps = int(max_episode_steps)
        self.sample_task_on_reset = bool(sample_task_on_reset)
        self.randomize_every_reset = bool(randomize_every_reset)
        self.rgb_width, self.rgb_height = rgb_size
        self.depth_width, self.depth_height = depth_size
        self._elapsed_steps = 0

        self.benchmark = metaworld.MT1(env_name, seed=seed)
        env_cls = self.benchmark.train_classes[env_name]
        self.metaworld_env = env_cls(
            render_mode="rgb_array",
            camera_name=camera_name,
        )
        self.tasks = [
            benchmark_task
            for benchmark_task in self.benchmark.train_tasks
            if benchmark_task.env_name == env_name
        ]
        if not self.tasks:
            raise ValueError(f"No MetaWorld tasks available for {env_name!r}.")

        if task is not None:
            self._current_task = task
        elif task_index is not None:
            self._current_task = self.tasks[task_index]
        else:
            self._current_task = self.tasks[0]
        self.metaworld_env.set_task(self._current_task)
        self._sync_state_observation_space()

        self.depth_sensor = DepthCameraSensor(
            camera_name=camera_name,
            height=self.depth_height,
            width=self.depth_width,
            normalize=depth_normalize,
        )
        self.tactile_sensor = TactileDigitSensor(
            resolution=tactile_resolution,
            noise_std=0.0,
            base_texture=False,
            normalize=tactile_normalize,
            photometric_render=tactile_photometric,
        )
        self.force_torque_sensor = ForceTorqueSensor(
            geom_names=None,
            origin_site="endEffector",
            output_frame="world",
            lowpass_alpha=0.2,
        )

        self.action_space = self.metaworld_env.action_space
        self.observation_space = self._make_observation_space()

    def _sync_state_observation_space(self) -> None:
        """Keep the wrapped env's public space aligned after ``set_task``."""
        self.metaworld_env.observation_space = (
            self.metaworld_env.sawyer_observation_space
        )

    def _make_observation_space(self) -> spaces.Dict:
        depth_flat_space = self.depth_sensor.get_observation_space()
        tactile_flat_space = self.tactile_sensor.get_observation_space()
        force_flat_space = self.force_torque_sensor.get_observation_space()

        if self.tactile_sensor.photometric_render:
            tactile_shape = (
                2,
                self.tactile_sensor.resolution,
                self.tactile_sensor.resolution,
                3,
            )
        else:
            tactile_shape = (
                2,
                self.tactile_sensor.resolution,
                self.tactile_sensor.resolution,
            )

        return spaces.Dict(
            {
                "state": self.metaworld_env.sawyer_observation_space,
                "pixels": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.rgb_height, self.rgb_width, 3),
                    dtype=np.uint8,
                ),
                "depth": spaces.Box(
                    low=float(np.min(depth_flat_space.low)),
                    high=float(np.max(depth_flat_space.high)),
                    shape=(self.depth_height, self.depth_width),
                    dtype=np.float32,
                ),
                "proprio": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7,),
                    dtype=np.float32,
                ),
                "tactile": spaces.Box(
                    low=float(np.min(tactile_flat_space.low)),
                    high=float(np.max(tactile_flat_space.high)),
                    shape=tactile_shape,
                    dtype=np.float32,
                ),
                "force_torque": spaces.Box(
                    low=float(np.min(force_flat_space.low)),
                    high=float(np.max(force_flat_space.high)),
                    shape=(6,),
                    dtype=np.float32,
                ),
                "gripper": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "ee_xyz": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "object_1_xyz": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "object_2_xyz": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "object_1_present": spaces.MultiBinary(1),
                "object_2_present": spaces.MultiBinary(1),
                "contact": spaces.MultiBinary(1),
            }
        )

    def _sample_task(self) -> Task:
        task_idx = int(self.np_random.integers(len(self.tasks)))
        return self.tasks[task_idx]

    def set_task(self, task: Task) -> None:
        self._current_task = task
        self.metaworld_env.set_task(task)
        self._sync_state_observation_space()
        if hasattr(self, "observation_space"):
            self.observation_space.spaces["state"] = (
                self.metaworld_env.sawyer_observation_space
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if self.sample_task_on_reset:
            self.set_task(self._sample_task())

        _set_randomized_resets(self.metaworld_env, self.randomize_every_reset)
        state, info = self.metaworld_env.reset(seed=seed, options=options)
        self.depth_sensor.reset(self.metaworld_env)
        self.tactile_sensor.reset(self.metaworld_env)
        self.force_torque_sensor.reset(self.metaworld_env)
        self._elapsed_steps = 0
        return self._get_observation(state), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        state, reward, terminated, truncated, info = self.metaworld_env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_episode_steps:
            truncated = True
        return (
            self._get_observation(state),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def _get_observation(self, state: np.ndarray) -> dict[str, np.ndarray]:
        pixels = _render_rgb_frame(
            self.metaworld_env,
            width=self.rgb_width,
            height=self.rgb_height,
        )

        self.depth_sensor.update(self.metaworld_env)
        depth = self.depth_sensor.get_depth_as_image().astype(np.float32)

        self.tactile_sensor.update(self.metaworld_env)
        left_tactile, right_tactile = self.tactile_sensor.get_finger_images()
        tactile = np.stack((left_tactile, right_tactile), axis=0).astype(np.float32)

        self.force_torque_sensor.update(self.metaworld_env)
        force_torque = np.asarray(self.force_torque_sensor.read(), dtype=np.float32)

        obj1, obj2, obj1_present, obj2_present = _get_object_xyzs(self.metaworld_env)

        return {
            "state": np.asarray(
                state,
                dtype=self.metaworld_env.observation_space.dtype,
            ),
            "pixels": pixels,
            "depth": depth,
            "proprio": np.asarray(
                self.metaworld_env.unwrapped.data.qpos[:7],
                dtype=np.float32,
            ).copy(),
            "tactile": tactile,
            "force_torque": force_torque,
            "gripper": np.asarray([state[3]], dtype=np.float32),
            "ee_xyz": np.asarray(
                self.metaworld_env.unwrapped.get_endeff_pos(),
                dtype=np.float32,
            ).copy(),
            "object_1_xyz": obj1,
            "object_2_xyz": obj2,
            "object_1_present": np.asarray([obj1_present], dtype=np.int8),
            "object_2_present": np.asarray([obj2_present], dtype=np.int8),
            "contact": np.asarray(
                [_gripper_touching_any(self.metaworld_env)],
                dtype=np.int8,
            ),
        }

    def render(self) -> np.ndarray:
        return _render_rgb_frame(
            self.metaworld_env,
            width=self.rgb_width,
            height=self.rgb_height,
        )

    def close(self) -> None:
        self.metaworld_env.close()


def make_metaworld_avid_env(env_name: str, **kwargs: Any) -> MetaWorldAvidEnv:
    """Create a Gymnasium-compatible AVID MetaWorld environment."""
    return MetaWorldAvidEnv(env_name, **kwargs)
