"""Test script for ForceTorqueSensor.

This script tests the 6-axis contact wrench sensor on the gripper.

Usage:
    python -m pytest tests/sensors/test_force_torque_sensor.py -v
    # Or run directly:
    python tests/sensors/test_force_torque_sensor.py
    python tests/sensors/test_force_torque_sensor.py --env-name basketball-v3
"""

import argparse
import csv
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from PIL import Image, ImageDraw

from metaworld.envs.sawyer_force_torque_validation_v3 import (
    SawyerForceTorqueValidationEnvV3,
)
from metaworld.policies import ENV_POLICY_MAP
from metaworld.policies.sawyer_button_press_v3_policy import SawyerButtonPressV3Policy
from metaworld.sensors.force_torque_sensor import ForceTorqueSensor

LOGGER = logging.getLogger(__name__)

BUTTON_PRESS_ENV_NAME = "button-press-v3"
DEFAULT_VIS_ENV_NAME = "basketball-v3"
VIS_CAMERA_NAME = "corner2"
VIS_WIDTH = 640
VIS_HEIGHT = 480


def _make_env(
    env_name: str,
    render_mode: str | None = None,
    camera_name: str | None = None,
    width: int = VIS_WIDTH,
    height: int = VIS_HEIGHT,
):
    return gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        render_mode=render_mode,
        camera_name=camera_name,
        width=width,
        height=height,
    )


def _make_button_press_env(
    render_mode: str | None = None,
    camera_name: str | None = None,
    width: int = VIS_WIDTH,
    height: int = VIS_HEIGHT,
):
    return _make_env(
        BUTTON_PRESS_ENV_NAME,
        render_mode=render_mode,
        camera_name=camera_name,
        width=width,
        height=height,
    )


def _make_validation_env(
    render_mode: str | None = None,
    camera_name: str | None = None,
    width: int = VIS_WIDTH,
    height: int = VIS_HEIGHT,
):
    env = SawyerForceTorqueValidationEnvV3(
        render_mode=render_mode,
        camera_name=camera_name,
        width=width,
        height=height,
    )
    env._set_task_called = True
    return env


def _make_policy(env_name: str):
    try:
        policy_cls = ENV_POLICY_MAP[env_name]
    except KeyError as exc:
        raise ValueError(
            f"No expert policy is registered for env '{env_name}'. "
            f"Supported envs: {sorted(ENV_POLICY_MAP)}"
        ) from exc
    return policy_cls()


def _button_contact_detected(env, sensor: ForceTorqueSensor) -> bool:
    button_geom_id = env.unwrapped.model.geom("btnGeom").id
    for contact_idx in range(env.unwrapped.data.ncon):
        contact = env.unwrapped.data.contact[contact_idx]
        geoms = {int(contact.geom1), int(contact.geom2)}
        if button_geom_id in geoms and any(
            geom_id in geoms for geom_id in sensor._geom_ids
        ):
            return True
    return False


def _rollout_button_press_policy(
    env,
    sensor: ForceTorqueSensor,
    policy: SawyerButtonPressV3Policy,
    *,
    steps: int,
):
    obs, _ = env.reset(seed=42)
    sensor.reset(env)

    max_force_norm = 0.0
    max_torque_norm = 0.0
    max_button_travel = 0.0
    contact_detected = False
    initial_button_y = float(env.unwrapped._get_site_pos("buttonStart")[1])

    for step_idx in range(steps):
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        sensor.update(env)
        wrench = sensor.read()

        force_norm = float(np.linalg.norm(wrench[:3]))
        torque_norm = float(np.linalg.norm(wrench[3:]))
        button_y = float(env.unwrapped._get_pos_objects()[1])
        button_travel = abs(button_y - initial_button_y)

        max_force_norm = max(max_force_norm, force_norm)
        max_torque_norm = max(max_torque_norm, torque_norm)
        max_button_travel = max(max_button_travel, button_travel)
        contact_detected = contact_detected or _button_contact_detected(env, sensor)

        yield {
            "step_idx": step_idx,
            "obs": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "wrench": wrench,
            "force_norm": force_norm,
            "torque_norm": torque_norm,
            "button_travel": button_travel,
            "contact_detected": contact_detected,
            "max_force_norm": max_force_norm,
            "max_torque_norm": max_torque_norm,
            "max_button_travel": max_button_travel,
        }

        if terminated or truncated:
            break


def _rollout_policy(
    env,
    sensor: ForceTorqueSensor,
    policy,
    *,
    steps: int,
):
    obs, _ = env.reset(seed=42)
    sensor.reset(env)

    max_force_norm = 0.0
    max_torque_norm = 0.0

    for step_idx in range(steps):
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        sensor.update(env)
        wrench = sensor.read()
        force_norm = float(np.linalg.norm(wrench[:3]))
        torque_norm = float(np.linalg.norm(wrench[3:]))
        max_force_norm = max(max_force_norm, force_norm)
        max_torque_norm = max(max_torque_norm, torque_norm)

        yield {
            "step_idx": step_idx,
            "obs": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "wrench": wrench,
            "force_norm": force_norm,
            "torque_norm": torque_norm,
            "max_force_norm": max_force_norm,
            "max_torque_norm": max_torque_norm,
        }

        if terminated or truncated:
            break


def _format_wrench_overlay(wrench: np.ndarray) -> list[str]:
    return [
        f"Fx: {wrench[0]: .4f} N",
        f"Fy: {wrench[1]: .4f} N",
        f"Fz: {wrench[2]: .4f} N",
        f"Tx: {wrench[3]: .4f} N*m",
        f"Ty: {wrench[4]: .4f} N*m",
        f"Tz: {wrench[5]: .4f} N*m",
    ]


def _format_pose_overlay(position: np.ndarray, quat: np.ndarray) -> list[str]:
    return [
        f"Hx: {position[0]: .4f} m",
        f"Hy: {position[1]: .4f} m",
        f"Hz: {position[2]: .4f} m",
        f"Qw: {quat[0]: .4f}",
        f"Qx: {quat[1]: .4f}",
        f"Qy: {quat[2]: .4f}",
        f"Qz: {quat[3]: .4f}",
    ]


def _format_compact_wrench_overlay(label: str, wrench: np.ndarray) -> list[str]:
    return [
        f"{label} Fx:{wrench[0]: .2f} Fy:{wrench[1]: .2f} Fz:{wrench[2]: .2f}",
        f"{label} Tx:{wrench[3]: .2f} Ty:{wrench[4]: .2f} Tz:{wrench[5]: .2f}",
    ]


def _annotate_frame(
    frame: np.ndarray,
    wrench: np.ndarray,
    *,
    extra_lines: list[str] | None = None,
) -> np.ndarray:
    flipped_frame = np.flipud(frame)
    image = Image.fromarray(flipped_frame)
    draw = ImageDraw.Draw(image, "RGBA")

    lines = _format_wrench_overlay(wrench)
    if extra_lines:
        lines.extend(extra_lines)
    line_height = 18
    box_height = 12 + line_height * len(lines)
    box_width = 480 if extra_lines else 320

    draw.rectangle((12, 12, 12 + box_width, 12 + box_height), fill=(0, 0, 0, 170))
    for idx, line in enumerate(lines):
        draw.text((20, 18 + idx * line_height), line, fill=(255, 255, 255, 255))

    return np.asarray(image)


def _default_output_path_for_env(env_name: str) -> Path:
    safe_env_name = env_name.replace("/", "_")
    return Path(f"force_torque_sensor_demo_{safe_env_name}.mp4")


def _encode_frames_to_video(
    frames_dir: Path,
    output_path: str | Path,
    *,
    fps: int,
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to generate the headless video output.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(
        ffmpeg_cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return output_path


def generate_force_torque_sensor_video(
    output_path: str | Path | None = None,
    *,
    env_name: str = DEFAULT_VIS_ENV_NAME,
    steps: int = 240,
    fps: int = 20,
    camera_name: str = VIS_CAMERA_NAME,
) -> Path:
    """Record an expert-policy demo video with force/torque measurements."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to generate the headless video output.")

    if output_path is None:
        output_path = _default_output_path_for_env(env_name)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = _make_env(
        env_name,
        render_mode="rgb_array",
        camera_name=camera_name,
        width=VIS_WIDTH,
        height=VIS_HEIGHT,
    )
    sensor = ForceTorqueSensor()
    policy = _make_policy(env_name)

    try:
        with tempfile.TemporaryDirectory(prefix="ft_sensor_frames_") as tmp_dir_name:
            frames_dir = Path(tmp_dir_name)
            for rollout_step in _rollout_policy(env, sensor, policy, steps=steps):
                frame = env.render()
                if frame is None:
                    raise RuntimeError("rgb_array rendering returned no frame.")

                frame_idx = int(rollout_step["step_idx"])
                wrench = rollout_step["wrench"]
                annotated_frame = _annotate_frame(frame, wrench)
                Image.fromarray(annotated_frame).save(
                    frames_dir / f"frame_{frame_idx:04d}.png"
                )
                LOGGER.info(
                    "env=%s video frame=%d force=%0.4f torque=%0.4f",
                    env_name,
                    frame_idx,
                    rollout_step["force_norm"],
                    rollout_step["torque_norm"],
                )
            _encode_frames_to_video(frames_dir, output_path, fps=fps)
    finally:
        env.close()

    return output_path


def visualize_force_torque_sensor(
    *,
    env_name: str = DEFAULT_VIS_ENV_NAME,
    steps: int = 250,
    fps: int = 30,
    camera_name: str = VIS_CAMERA_NAME,
    output_path: str | Path | None = None,
) -> Path | None:
    """Render the chosen expert-policy rollout to video."""
    output_path = generate_force_torque_sensor_video(
        output_path=output_path,
        env_name=env_name,
        steps=steps,
        fps=fps,
        camera_name=camera_name,
    )
    print(f"Video written to: {output_path}")
    return output_path


def _read_reference_wrench(
    env: SawyerForceTorqueValidationEnvV3, *, output_frame: str = "sensor"
) -> np.ndarray:
    force_sensor = env.unwrapped.model.sensor("wrist_force_ref")
    torque_sensor = env.unwrapped.model.sensor("wrist_torque_ref")
    data = env.unwrapped.data
    force_adr = int(force_sensor.adr[0])
    force_dim = int(force_sensor.dim[0])
    torque_adr = int(torque_sensor.adr[0])
    torque_dim = int(torque_sensor.dim[0])

    force = np.asarray(
        data.sensordata[force_adr : force_adr + force_dim],
        dtype=np.float64,
    )
    torque = np.asarray(
        data.sensordata[torque_adr : torque_adr + torque_dim],
        dtype=np.float64,
    )
    wrench = np.concatenate((force, torque))
    if output_frame == "world":
        site_id = env.unwrapped.model.site("endEffector").id
        rot_sensor_to_world = data.site_xmat[site_id].reshape(3, 3)
        wrench[:3] = rot_sensor_to_world @ wrench[:3]
        wrench[3:] = rot_sensor_to_world @ wrench[3:]
    return wrench


def _validation_wall_contact_detected(env: SawyerForceTorqueValidationEnvV3) -> bool:
    wall_geom_id = env.unwrapped.model.geom("validation_wall").id
    payload_geom_id = env.unwrapped.model.geom("payload_geom").id
    for contact_idx in range(env.unwrapped.data.ncon):
        contact = env.unwrapped.data.contact[contact_idx]
        geoms = {int(contact.geom1), int(contact.geom2)}
        if wall_geom_id in geoms and payload_geom_id in geoms:
            return True
    return False


def generate_force_torque_validation_debug_video(
    output_path: str | Path | None = None,
    *,
    fps: int = 20,
    camera_name: str = VIS_CAMERA_NAME,
    static_steps: int = 20,
    max_approach_steps: int = 40,
    press_steps_after_contact: int = 40,
) -> Path:
    """Render the validation env with static and contact-aware wall-push phases."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to generate the headless video output.")

    if output_path is None:
        output_path = Path("force_torque_validation_debug.mp4")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = _make_validation_env(
        render_mode="rgb_array",
        camera_name=camera_name,
        width=VIS_WIDTH,
        height=VIS_HEIGHT,
    )
    sensor_world = ForceTorqueSensor(output_frame="world")
    sensor_sensor = ForceTorqueSensor(output_frame="sensor")

    try:
        env.reset(seed=42)
        sensor_world.reset(env)
        sensor_sensor.reset(env)

        payload_pos = np.asarray(
            env.unwrapped.get_body_com("payload"), dtype=np.float64
        )
        env.unwrapped.model.geom("validation_wall").pos = np.array(
            [payload_pos[0] - 0.11, payload_pos[1], payload_pos[2]],
            dtype=np.float64,
        )
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

        with tempfile.TemporaryDirectory(
            prefix="ft_validation_frames_"
        ) as tmp_dir_name:
            frames_dir = Path(tmp_dir_name)
            frame_idx = 0
            contact_frame_idx: int | None = None
            baseline_world_wrench: np.ndarray | None = None
            baseline_sensor_wrench: np.ndarray | None = None

            def record_frame(phase: str, action: np.ndarray) -> bool:
                nonlocal frame_idx, contact_frame_idx
                nonlocal baseline_world_wrench, baseline_sensor_wrench

                sensor_world.update(env)
                sensor_sensor.update(env)

                wrench_world = sensor_world.read()
                wrench_sensor = sensor_sensor.read()
                ref_world = _read_reference_wrench(env, output_frame="world")
                ref_sensor = _read_reference_wrench(env, output_frame="sensor")
                frame = env.render()
                if frame is None:
                    raise RuntimeError("rgb_array rendering returned no frame.")

                contact_detected = _validation_wall_contact_detected(env)
                if phase == "static" and baseline_world_wrench is None:
                    baseline_world_wrench = wrench_world.copy()
                    baseline_sensor_wrench = wrench_sensor.copy()
                if contact_detected and contact_frame_idx is None:
                    contact_frame_idx = frame_idx

                extra_lines = [
                    f"Phase: {phase}",
                    f"Action: [{action[0]: .2f}, {action[1]: .2f}, {action[2]: .2f}, {action[3]: .2f}]",
                    f"Wall contact: {contact_detected}",
                ]
                extra_lines.extend(_format_compact_wrench_overlay("CW", wrench_world))
                extra_lines.extend(_format_compact_wrench_overlay("RW", ref_world))
                extra_lines.extend(_format_compact_wrench_overlay("CS", wrench_sensor))
                extra_lines.extend(_format_compact_wrench_overlay("RS", ref_sensor))
                if (
                    baseline_world_wrench is not None
                    and baseline_sensor_wrench is not None
                ):
                    extra_lines.extend(
                        _format_compact_wrench_overlay(
                            "dW", wrench_world - baseline_world_wrench
                        )
                    )
                    extra_lines.extend(
                        _format_compact_wrench_overlay(
                            "dS", wrench_sensor - baseline_sensor_wrench
                        )
                    )
                extra_lines.append(
                    f"|W-R| force:{np.linalg.norm(wrench_world[:3] - ref_world[:3]): .2f} "
                    f"torque:{np.linalg.norm(wrench_world[3:] - ref_world[3:]): .2f}"
                )

                annotated_frame = _annotate_frame(
                    frame,
                    wrench_world,
                    extra_lines=extra_lines,
                )
                Image.fromarray(annotated_frame).save(
                    frames_dir / f"frame_{frame_idx:04d}.png"
                )
                frame_idx += 1
                return contact_detected

            static_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            for _ in range(static_steps):
                env.step(static_action)
                record_frame("static", static_action)

            approach_action = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            contact_detected = False
            for _ in range(max_approach_steps):
                env.step(approach_action)
                contact_detected = record_frame("approach-left", approach_action)
                if contact_detected:
                    break

            for _ in range(press_steps_after_contact):
                env.step(approach_action)
                record_frame(
                    "press-left-after-contact"
                    if contact_detected
                    else "continue-approach-left",
                    approach_action,
                )

            _encode_frames_to_video(frames_dir, output_path, fps=fps)
    finally:
        env.close()

    return output_path


class TestForceTorqueSensor:
    """Test suite for ForceTorqueSensor."""

    def test_sensor_creation(self):
        """Test that we can create a ForceTorqueSensor."""
        sensor = ForceTorqueSensor()
        assert sensor.geom_names is None
        assert sensor.origin_site == "endEffector"
        assert sensor.output_frame == "world"
        assert not sensor.invert_sign

    def test_observation_space(self):
        """Test that observation space is correctly defined."""
        sensor = ForceTorqueSensor()
        obs_space = sensor.get_observation_space()
        assert obs_space.shape == (6,)
        assert obs_space.dtype == np.float64

    def test_metadata(self):
        """Test that sensor metadata is populated."""
        sensor = ForceTorqueSensor(
            geom_names=("leftpad_geom",),
            origin_site="rightEndEffector",
            output_frame="sensor",
            invert_sign=True,
        )
        metadata = sensor.get_metadata()
        assert metadata["type"] == "force"
        assert metadata["subtype"] == "contact_wrench"
        assert metadata["geom_names"] == ("leftpad_geom",)
        assert metadata["resolved_geom_names"] == ()
        assert metadata["origin_site"] == "rightEndEffector"
        assert metadata["output_frame"] == "sensor"
        assert metadata["invert_sign"] is True

    def test_read_before_reset_raises_error(self):
        """Test that reading before reset raises an error."""
        sensor = ForceTorqueSensor()
        with pytest.raises(RuntimeError, match="read\\(\\) called before reset\\(\\)"):
            sensor.read()

    def test_validate_and_reset(self):
        """Test validate and reset with valid/invalid names."""
        env = gym.make("Meta-World/MT1", env_name="pick-place-v3")
        env.reset()

        sensor_valid = ForceTorqueSensor()
        assert sensor_valid.validate(env)
        sensor_valid.reset(env)
        assert {
            "rail",
            "leftclaw_it",
            "rightclaw_it",
            "leftpad_geom",
            "rightpad_geom",
        }.issubset(set(sensor_valid.get_metadata()["resolved_geom_names"]))

        sensor_bad_site = ForceTorqueSensor(origin_site="not_a_site")
        assert not sensor_bad_site.validate(env)
        with pytest.raises(RuntimeError, match="Site 'not_a_site' not found"):
            sensor_bad_site.reset(env)

        sensor_bad_geom = ForceTorqueSensor(geom_names=("not_a_geom",))
        assert not sensor_bad_geom.validate(env)
        with pytest.raises(RuntimeError, match="One or more geometries not found"):
            sensor_bad_geom.reset(env)

        env.close()

    def test_sensor_with_environment(self):
        """Test sensor output shape and numerical sanity."""
        env = _make_button_press_env()
        sensor = ForceTorqueSensor()

        env.reset(seed=42)
        sensor.reset(env)
        sensor.update(env)
        wrench = sensor.read()

        assert wrench.shape == (6,)
        assert wrench.dtype == np.float64
        assert np.all(np.isfinite(wrench))

        env.close()

    def test_button_press_policy_reports_force_on_button_contact(self):
        """Test expert rollout produces button motion and measurable wrench."""
        env = _make_button_press_env()
        sensor = ForceTorqueSensor()
        policy = SawyerButtonPressV3Policy()

        max_force_norm = 0.0
        max_torque_norm = 0.0
        max_button_travel = 0.0
        contact_detected = False

        for rollout_step in _rollout_button_press_policy(
            env, sensor, policy, steps=250
        ):
            max_force_norm = max(max_force_norm, float(rollout_step["force_norm"]))
            max_torque_norm = max(max_torque_norm, float(rollout_step["torque_norm"]))
            max_button_travel = max(
                max_button_travel, float(rollout_step["button_travel"])
            )
            contact_detected = contact_detected or bool(
                rollout_step["contact_detected"]
            )
            LOGGER.info(
                "button step=%d force=%0.4f torque=%0.4f travel=%0.4f success=%s",
                int(rollout_step["step_idx"]),
                float(rollout_step["force_norm"]),
                float(rollout_step["torque_norm"]),
                float(rollout_step["button_travel"]),
                rollout_step["info"].get("success"),
            )

        env.close()
        assert contact_detected
        assert max_button_travel > 0.01
        assert max_force_norm > 1e-3
        assert max_torque_norm >= 0.0

    def test_static_gravity_wrench_matches_reference_and_expected_magnitude(self):
        """Static payload load should produce the expected gravity wrench."""
        env = _make_validation_env()
        sensor_world = ForceTorqueSensor(output_frame="world")
        sensor_sensor = ForceTorqueSensor(output_frame="sensor")

        try:
            env.reset(seed=42)
            sensor_world.reset(env)
            sensor_sensor.reset(env)

            for _ in range(20):
                env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

            sensor_world.update(env)
            sensor_sensor.update(env)
            wrench_world = sensor_world.read()
            wrench_sensor = sensor_sensor.read()
            ref_sensor = _read_reference_wrench(env, output_frame="sensor")
            ref_world = _read_reference_wrench(env, output_frame="world")

            LOGGER.info(
                "Static world wrench: %s", np.array2string(wrench_world, precision=4)
            )
            LOGGER.info(
                "Static sensor wrench: %s", np.array2string(wrench_sensor, precision=4)
            )
            LOGGER.info(
                "Static reference world wrench: %s",
                np.array2string(ref_world, precision=4),
            )
            LOGGER.info(
                "Static reference sensor wrench: %s",
                np.array2string(ref_sensor, precision=4),
            )

            np.testing.assert_allclose(wrench_sensor, ref_sensor, atol=0.3, rtol=0.05)
            np.testing.assert_allclose(wrench_world, ref_world, atol=0.3, rtol=0.05)
            assert abs(wrench_world[2]) > abs(wrench_world[0]) + 10.0
            assert abs(wrench_world[2]) > abs(wrench_world[1]) + 5.0
            assert np.linalg.norm(wrench_world[3:]) > 0.1
        finally:
            env.close()

    def test_single_known_wall_contact_produces_significant_additional_load(self):
        """Pushing the welded payload into a fixed wall should create a clear contact wrench increase."""
        env = _make_validation_env()
        sensor_world = ForceTorqueSensor(output_frame="world")

        try:
            env.reset(seed=42)
            sensor_world.reset(env)
            payload_pos = np.asarray(
                env.unwrapped.get_body_com("payload"), dtype=np.float64
            )
            env.unwrapped.model.geom("validation_wall").pos = np.array(
                [payload_pos[0] + 0.11, payload_pos[1], payload_pos[2]],
                dtype=np.float64,
            )
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

            baseline_samples: list[np.ndarray] = []
            for _ in range(10):
                env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
                sensor_world.update(env)
                baseline_samples.append(sensor_world.read())

            wall_geom_id = env.unwrapped.model.geom("validation_wall").id
            payload_geom_id = env.unwrapped.model.geom("payload_geom").id
            post_contact_samples: list[np.ndarray] = []
            post_contact_ref_samples: list[np.ndarray] = []
            touched_wall = False

            for _ in range(20):
                env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
                sensor_world.update(env)
                wrench = sensor_world.read()

                for contact_idx in range(env.unwrapped.data.ncon):
                    contact = env.unwrapped.data.contact[contact_idx]
                    geoms = {int(contact.geom1), int(contact.geom2)}
                    if wall_geom_id in geoms and payload_geom_id in geoms:
                        touched_wall = True
                        post_contact_samples.append(wrench.copy())
                        post_contact_ref_samples.append(
                            _read_reference_wrench(env, output_frame="world")
                        )
                        break

            assert touched_wall
            assert len(post_contact_samples) > 0

            baseline_mean = np.mean(np.asarray(baseline_samples), axis=0)
            contact_mean = np.mean(np.asarray(post_contact_samples), axis=0)
            delta_force = contact_mean[:3] - baseline_mean[:3]
            ref_world = np.mean(np.asarray(post_contact_ref_samples), axis=0)

            LOGGER.info(
                "Baseline world wrench: %s", np.array2string(baseline_mean, precision=4)
            )
            LOGGER.info(
                "Contact world wrench: %s", np.array2string(contact_mean, precision=4)
            )
            LOGGER.info(
                "Delta force from wall contact: %s",
                np.array2string(delta_force, precision=4),
            )
            LOGGER.info(
                "Reference world wrench at contact: %s",
                np.array2string(ref_world, precision=4),
            )

            np.testing.assert_allclose(contact_mean, ref_world, atol=0.5, rtol=0.05)
            assert np.linalg.norm(delta_force) > 20.0
        finally:
            env.close()

    def test_world_and_sensor_frames_are_rotation_consistent(self):
        """World-frame and sensor-frame wrenches should agree via the site rotation."""
        env = _make_validation_env()
        sensor_world = ForceTorqueSensor(output_frame="world")
        sensor_sensor = ForceTorqueSensor(output_frame="sensor")

        try:
            env.reset(seed=42)
            sensor_world.reset(env)
            sensor_sensor.reset(env)

            for _ in range(20):
                env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

            sensor_world.update(env)
            sensor_sensor.update(env)
            wrench_world = sensor_world.read()
            wrench_sensor = sensor_sensor.read()

            site_id = env.unwrapped.model.site("endEffector").id
            rot_sensor_to_world = env.unwrapped.data.site_xmat[site_id].reshape(3, 3)
            expected_sensor_force = rot_sensor_to_world.T @ wrench_world[:3]
            expected_sensor_torque = rot_sensor_to_world.T @ wrench_world[3:]

            np.testing.assert_allclose(
                wrench_sensor[:3], expected_sensor_force, atol=0.3, rtol=0.05
            )
            np.testing.assert_allclose(
                wrench_sensor[3:], expected_sensor_torque, atol=0.3, rtol=0.05
            )
        finally:
            env.close()

    def test_directional_probe_logs_forces(self, tmp_path):
        """Probe workspace boundaries and log wrench data for inspection.

        The sequence sweeps the end-effector in the open pick-place workspace toward:
        - table/downward contact
        - slight raise from the table
        - left edge
        - left edge press
        - right edge
        - forward edge
        """
        env = gym.make(
            "Meta-World/MT1",
            env_name="pick-place-v3",
            render_mode="rgb_array",
            camera_name=VIS_CAMERA_NAME,
            width=VIS_WIDTH,
            height=VIS_HEIGHT,
        )
        sensor = ForceTorqueSensor()
        env.reset(seed=7)
        sensor.reset(env)

        hand_low = np.asarray(env.unwrapped.hand_low, dtype=np.float64)
        hand_high = np.asarray(env.unwrapped.hand_high, dtype=np.float64)
        boundary_tol = 0.01
        stall_tol = 5e-4
        stall_steps_required = 5
        max_sweep_steps = 200
        press_steps = 20
        raise_steps = 15

        records: list[dict[str, float | str | int]] = []
        max_force_norm_by_phase: dict[str, float] = {}
        step_idx = 0
        video_path = tmp_path / "force_torque_directional_probe.mp4"

        try:
            with tempfile.TemporaryDirectory(
                prefix="ft_directional_probe_frames_"
            ) as tmp_dir_name:
                frames_dir = Path(tmp_dir_name)

                def record_probe_step(phase: str, action: np.ndarray) -> np.ndarray:
                    nonlocal step_idx

                    sensor.update(env)
                    wrench = sensor.read()
                    force_norm = float(np.linalg.norm(wrench[:3]))
                    torque_norm = float(np.linalg.norm(wrench[3:]))
                    tcp_after = np.asarray(env.unwrapped.tcp_center, dtype=np.float64)
                    ncon = int(env.unwrapped.data.ncon)

                    records.append(
                        {
                            "step": step_idx,
                            "phase": phase,
                            "tcp_x": float(tcp_after[0]),
                            "tcp_y": float(tcp_after[1]),
                            "tcp_z": float(tcp_after[2]),
                            "fx": float(wrench[0]),
                            "fy": float(wrench[1]),
                            "fz": float(wrench[2]),
                            "tx": float(wrench[3]),
                            "ty": float(wrench[4]),
                            "tz": float(wrench[5]),
                            "force_norm": force_norm,
                            "torque_norm": torque_norm,
                            "ncon": ncon,
                        }
                    )

                    frame = env.render()
                    if frame is None:
                        raise RuntimeError("rgb_array rendering returned no frame.")
                    extra_lines = [
                        f"Probe phase: {phase}",
                        f"Action: [{action[0]: .2f}, {action[1]: .2f}, {action[2]: .2f}, {action[3]: .2f}]",
                        f"TCP: [{tcp_after[0]: .3f}, {tcp_after[1]: .3f}, {tcp_after[2]: .3f}]",
                        f"Force norm: {force_norm: .3f} N",
                        f"Torque norm: {torque_norm: .3f} N*m",
                        f"Contacts: {ncon}",
                    ]
                    annotated_frame = _annotate_frame(
                        frame,
                        wrench,
                        extra_lines=extra_lines,
                    )
                    Image.fromarray(annotated_frame).save(
                        frames_dir / f"frame_{step_idx:04d}.png"
                    )
                    step_idx += 1
                    return tcp_after

                def run_constant_action_phase(
                    phase: str,
                    action: np.ndarray,
                    *,
                    max_steps: int,
                    stop_when,
                ) -> float:
                    max_force_norm = 0.0
                    stalled_steps = 0
                    prev_tcp = np.asarray(env.unwrapped.tcp_center, dtype=np.float64)

                    for _ in range(max_steps):
                        env.step(action)
                        tcp_after = record_probe_step(phase, action)
                        force_norm = float(records[-1]["force_norm"])
                        max_force_norm = max(max_force_norm, force_norm)

                        if np.linalg.norm(tcp_after - prev_tcp) < stall_tol:
                            stalled_steps += 1
                        else:
                            stalled_steps = 0

                        if (
                            stop_when(tcp_after)
                            or stalled_steps >= stall_steps_required
                        ):
                            break
                        prev_tcp = tcp_after

                    return max_force_norm

                # Let dynamics settle first.
                for _ in range(10):
                    env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
                    sensor.update(env)

                phase_specs = [
                    (
                        "down_to_table",
                        np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32),
                        lambda tcp: tcp[2] <= hand_low[2] + boundary_tol,
                        max_sweep_steps,
                    ),
                    (
                        "raise_from_table",
                        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
                        lambda tcp: False,
                        raise_steps,
                    ),
                    (
                        "left",
                        np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        lambda tcp: tcp[0] <= hand_low[0] + boundary_tol,
                        max_sweep_steps,
                    ),
                    (
                        "press_left",
                        np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        lambda tcp: False,
                        press_steps,
                    ),
                    (
                        "right",
                        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        lambda tcp: tcp[0] >= hand_high[0] - boundary_tol,
                        max_sweep_steps * 2,
                    ),
                    (
                        "straight",
                        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                        lambda tcp: tcp[1] >= hand_high[1] - boundary_tol,
                        max_sweep_steps,
                    ),
                ]

                for phase, action, stop_when, max_steps in phase_specs:
                    max_force_norm_by_phase[phase] = run_constant_action_phase(
                        phase,
                        action,
                        max_steps=max_steps,
                        stop_when=stop_when,
                    )

                _encode_frames_to_video(frames_dir, video_path, fps=20)
        finally:
            env.close()

        csv_path = tmp_path / "force_torque_directional_probe.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        LOGGER.info("Directional force/torque log written to: %s", csv_path)
        LOGGER.info("Directional probe video written to: %s", video_path)
        LOGGER.info("Peak force norm by phase: %s", max_force_norm_by_phase)

        assert len(records) > 0
        assert csv_path.exists()
        assert video_path.exists()
        assert np.all(np.isfinite([r["force_norm"] for r in records]))
        assert max_force_norm_by_phase["down_to_table"] > 1e-3
        assert max_force_norm_by_phase["left"] > 1e-3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a force/torque demo video.")
    parser.add_argument(
        "--validation-debug-video",
        action="store_true",
        help="Render the dedicated validation debug video instead of an expert-policy demo.",
    )
    parser.add_argument(
        "--env-name",
        default=DEFAULT_VIS_ENV_NAME,
        help=f"MetaWorld env to render. Default: {DEFAULT_VIS_ENV_NAME}",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=250,
        help="Number of rollout steps to render.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video frame rate.",
    )
    parser.add_argument(
        "--camera-name",
        default=VIS_CAMERA_NAME,
        help=f"MuJoCo camera to render from. Default: {VIS_CAMERA_NAME}",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional explicit output path for the rendered mp4.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.validation_debug_video:
        output_path = generate_force_torque_validation_debug_video(
            output_path=args.output_path,
            fps=args.fps,
            camera_name=args.camera_name,
        )
        print(f"Validation debug video written to: {output_path}")
    else:
        visualize_force_torque_sensor(
            env_name=args.env_name,
            steps=args.steps,
            fps=args.fps,
            camera_name=args.camera_name,
            output_path=args.output_path,
        )
