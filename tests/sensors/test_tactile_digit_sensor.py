"""Tests and qualitative visualization for TactileDigitSensor.

Usage:
    python -m pytest tests/sensors/test_tactile_digit_sensor.py -v
    python tests/sensors/test_tactile_digit_sensor.py
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import gymnasium as gym
import matplotlib
import numpy as np
import pytest

from metaworld.policies.sawyer_basketball_v3_policy import SawyerBasketballV3Policy
from metaworld.sensors.tactile_digit_sensor import TactileDigitSensor

# Use a non-interactive backend so qualitative tests work in headless runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


QUAL_VIS_ENV_NAME = "basketball-v3"
QUAL_VIS_CAMERA = "corner2"


def _split_tactile_reading(
    sensor: TactileDigitSensor, reading: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape the flattened reading back into left/right RGB images."""
    image_size = sensor.resolution * sensor.resolution * 3
    left = reading[:image_size].reshape(sensor.resolution, sensor.resolution, 3)
    right = reading[image_size:].reshape(sensor.resolution, sensor.resolution, 3)
    return left, right


def _activation_score(sensor: TactileDigitSensor, image: np.ndarray) -> float:
    """Measure deviation from the nominal gel background."""
    baseline = sensor._base_image
    scale = 255.0 if not sensor.normalize else 1.0
    return float(np.mean(np.abs(image / scale - baseline)))


def _collect_contact_rollout(
    sensor: TactileDigitSensor,
    *,
    env_name: str = QUAL_VIS_ENV_NAME,
    steps: int = 200,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Run an expert policy and record tactile snapshots through contact."""
    env = gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        render_mode="rgb_array",
        camera_name=QUAL_VIS_CAMERA,
        width=640,
        height=480,
    )
    policy = SawyerBasketballV3Policy()
    frames: list[dict[str, object]] = []

    try:
        obs, _ = env.reset(seed=seed)
        sensor.reset(env)
        sensor.update(env)

        initial_reading = sensor.read().copy()
        initial_left, initial_right = _split_tactile_reading(sensor, initial_reading)
        frames.append(
            {
                "step": 0,
                "scene": _safe_render_scene(env),
                "left": initial_left,
                "right": initial_right,
                "left_activation": _activation_score(sensor, initial_left),
                "right_activation": _activation_score(sensor, initial_right),
            }
        )

        for step_idx in range(1, steps + 1):
            action = policy.get_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            sensor.update(env)
            reading = sensor.read().copy()
            left, right = _split_tactile_reading(sensor, reading)
            frames.append(
                {
                    "step": step_idx,
                    "scene": _safe_render_scene(env),
                    "left": left,
                    "right": right,
                    "left_activation": _activation_score(sensor, left),
                    "right_activation": _activation_score(sensor, right),
                }
            )

            if terminated or truncated:
                break
    finally:
        env.close()

    return frames


def _safe_render_scene(env) -> np.ndarray:
    """Render the scene when possible, otherwise synthesize a useful overview panel."""
    try:
        frame = env.render()
    except Exception:
        return _make_headless_scene_panel(env)

    if frame is None:
        return _make_headless_scene_panel(env)
    return np.flipud(np.asarray(frame))


def _make_headless_scene_panel(env) -> np.ndarray:
    """Create a lightweight scene summary image for headless qualitative tests."""
    width = 640
    height = 480
    canvas = np.ones((height, width, 3), dtype=np.float32)
    canvas[..., 0] *= 0.96
    canvas[..., 1] *= 0.97
    canvas[..., 2] *= 0.99

    title_band = slice(0, 70)
    canvas[title_band, :, :] = np.array([0.16, 0.20, 0.28], dtype=np.float32)

    obj_pos = np.asarray(env.unwrapped._get_pos_objects(), dtype=np.float32)
    tcp_pos = np.asarray(env.unwrapped.tcp_center, dtype=np.float32)
    left_pad = np.asarray(
        env.unwrapped.data.geom_xpos[env.unwrapped.model.geom("leftpad_geom").id],
        dtype=np.float32,
    )
    right_pad = np.asarray(
        env.unwrapped.data.geom_xpos[env.unwrapped.model.geom("rightpad_geom").id],
        dtype=np.float32,
    )

    positions = np.vstack([obj_pos, tcp_pos, left_pad, right_pad])
    xy = positions[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)

    x_min = float(mins[0] - 0.08)
    x_max = float(maxs[0] + 0.08)
    y_min = float(mins[1] - 0.08)
    y_max = float(maxs[1] + 0.08)

    plot_x0 = 40
    plot_x1 = width - 40
    plot_y0 = 110
    plot_y1 = height - 40
    canvas[plot_y0:plot_y1, plot_x0:plot_x1, :] = np.array(
        [0.92, 0.94, 0.97], dtype=np.float32
    )

    def world_to_px(pos_xy: np.ndarray) -> tuple[int, int]:
        x = (float(pos_xy[0]) - x_min) / max(x_max - x_min, 1e-6)
        y = (float(pos_xy[1]) - y_min) / max(y_max - y_min, 1e-6)
        px = int(plot_x0 + np.clip(x, 0.0, 1.0) * (plot_x1 - plot_x0))
        py = int(plot_y1 - np.clip(y, 0.0, 1.0) * (plot_y1 - plot_y0))
        return px, py

    def draw_disk(center_xy: np.ndarray, radius: int, color: tuple[float, float, float]) -> None:
        px, py = world_to_px(center_xy)
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - px) ** 2 + (yy - py) ** 2 <= radius**2
        canvas[mask] = np.array(color, dtype=np.float32)

    def draw_line(
        p0_xy: np.ndarray,
        p1_xy: np.ndarray,
        color: tuple[float, float, float],
        thickness: int = 2,
    ) -> None:
        p0 = np.array(world_to_px(p0_xy), dtype=np.float32)
        p1 = np.array(world_to_px(p1_xy), dtype=np.float32)
        seg = p1 - p0
        denom = float(np.dot(seg, seg))
        yy, xx = np.mgrid[:height, :width]
        pts = np.stack([xx, yy], axis=-1).astype(np.float32)
        if denom < 1e-6:
            dist2 = np.sum((pts - p0) ** 2, axis=-1)
        else:
            t = np.clip(np.sum((pts - p0) * seg, axis=-1) / denom, 0.0, 1.0)
            proj = p0 + t[..., None] * seg
            dist2 = np.sum((pts - proj) ** 2, axis=-1)
        mask = dist2 <= float(thickness**2)
        canvas[mask] = np.array(color, dtype=np.float32)

    draw_line(left_pad[:2], tcp_pos[:2], (0.55, 0.63, 0.80), thickness=2)
    draw_line(right_pad[:2], tcp_pos[:2], (0.55, 0.63, 0.80), thickness=2)
    draw_line(left_pad[:2], obj_pos[:2], (0.88, 0.78, 0.52), thickness=1)
    draw_line(right_pad[:2], obj_pos[:2], (0.88, 0.78, 0.52), thickness=1)

    draw_disk(obj_pos[:2], radius=16, color=(0.84, 0.50, 0.26))
    draw_disk(tcp_pos[:2], radius=14, color=(0.18, 0.24, 0.78))
    draw_disk(left_pad[:2], radius=10, color=(0.16, 0.62, 0.34))
    draw_disk(right_pad[:2], radius=10, color=(0.72, 0.18, 0.32))

    # Simple legend swatches.
    canvas[18:34, 26:42, :] = np.array([0.84, 0.50, 0.26], dtype=np.float32)
    canvas[18:34, 190:206, :] = np.array([0.18, 0.24, 0.78], dtype=np.float32)
    canvas[42:58, 26:42, :] = np.array([0.16, 0.62, 0.34], dtype=np.float32)
    canvas[42:58, 190:206, :] = np.array([0.72, 0.18, 0.32], dtype=np.float32)

    return np.clip(canvas * 255.0, 0.0, 255.0).astype(np.uint8)


def visualize_tactile_digit_rollout(
    output_path: str | Path,
    *,
    env_name: str = QUAL_VIS_ENV_NAME,
    steps: int = 200,
) -> Path:
    """Save an MP4 showing scene, left tactile, and right tactile over time."""
    sensor = TactileDigitSensor(resolution=64, sigma_px=2.0, noise_std=0.0, seed=7)
    frames = _collect_contact_rollout(sensor, env_name=env_name, steps=steps)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (1440, 480),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for '{output_path}'.")

    try:
        for frame in frames:
            composed = _compose_rollout_video_frame(sensor, frame)
            writer.write(cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return output_path


def _compose_rollout_video_frame(
    sensor: TactileDigitSensor, frame: dict[str, object]
) -> np.ndarray:
    """Compose one video frame with scene, left tactile, and right tactile columns."""
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), dpi=100)

    scene_ax, left_ax, right_ax = axes
    scene_ax.imshow(np.asarray(frame["scene"]))
    scene_ax.set_title(f"Scene step {frame['step']}")
    scene_ax.axis("off")

    tactile_high = 1.0 if sensor.normalize else 255.0
    left_ax.imshow(np.clip(np.asarray(frame["left"]), 0.0, tactile_high))
    left_ax.set_title(f"Left tactile\nact={frame['left_activation']:.3f}")
    left_ax.axis("off")

    right_ax.imshow(np.clip(np.asarray(frame["right"]), 0.0, tactile_high))
    right_ax.set_title(f"Right tactile\nact={frame['right_activation']:.3f}")
    right_ax.axis("off")

    fig.suptitle(
        "Basketball grasp rollout: scene and tactile evolution",
        fontsize=14,
    )
    fig.tight_layout()
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
        height, width, 4
    )[..., :3]
    plt.close(fig)
    return image


class TestTactileDigitSensor:
    """Test suite for the DIGIT/GelSight-style tactile sensor."""

    def test_sensor_creation(self):
        """Test that the sensor can be instantiated."""
        sensor = TactileDigitSensor(
            resolution=32,
            sigma_px=1.5,
            noise_std=0.0,
            base_texture=True,
            seed=123,
            normalize=True,
        )

        assert sensor.name == "tactile_digit"
        assert sensor.resolution == 32
        assert sensor.sigma_px == 1.5
        assert sensor.noise_std == 0.0
        assert sensor.base_texture
        assert sensor.seed == 123
        assert sensor.normalize

    def test_observation_space(self):
        """Test that observation space matches two flattened RGB images."""
        sensor = TactileDigitSensor(resolution=32, normalize=True)
        obs_space = sensor.get_observation_space()

        assert obs_space.shape == (2 * 32 * 32 * 3,)
        assert obs_space.dtype == np.float32
        assert np.all(obs_space.low == 0.0)
        assert np.all(obs_space.high == 1.0)

    def test_observation_space_unnormalized(self):
        """Test observation bounds for [0, 255] output mode."""
        sensor = TactileDigitSensor(resolution=16, normalize=False)
        obs_space = sensor.get_observation_space()

        assert obs_space.shape == (2 * 16 * 16 * 3,)
        assert obs_space.high[0] == 255.0

    def test_metadata(self):
        """Test that metadata exposes tactile-image properties."""
        sensor = TactileDigitSensor(resolution=64, sigma_px=2.5)
        metadata = sensor.get_metadata()

        assert metadata["type"] == "tactile"
        assert metadata["modality"] == "vision_tactile"
        assert metadata["resolution"] == 64
        assert metadata["channels"] == 3
        assert metadata["fingers"] == 2
        assert metadata["sigma_px"] == 2.5

    def test_read_before_reset_raises_error(self):
        """Test that reading before reset raises an informative error."""
        sensor = TactileDigitSensor()
        with pytest.raises(RuntimeError, match="read\\(\\) called before reset\\(\\)"):
            sensor.read()

    def test_seed_makes_base_texture_deterministic(self):
        """Test seeded sensors produce identical base tactile images."""
        sensor_a = TactileDigitSensor(seed=123, base_texture=True)
        sensor_b = TactileDigitSensor(seed=123, base_texture=True)
        sensor_c = TactileDigitSensor(seed=456, base_texture=True)

        assert np.allclose(sensor_a._base_image, sensor_b._base_image)
        assert not np.allclose(sensor_a._base_image, sensor_c._base_image)

    def test_render_tactile_image_zero_pressure_returns_base(self):
        """Test that zero pressure produces the nominal gel image."""
        sensor = TactileDigitSensor(resolution=32, noise_std=0.0, seed=0)
        pressure = np.zeros((32, 32), dtype=np.float32)
        image = sensor._render_tactile_image(pressure)

        assert image.shape == (32, 32, 3)
        assert image.dtype == np.float32
        assert np.allclose(image, sensor._base_image)

    def test_add_gaussian_splat_increases_local_pressure(self):
        """Test Gaussian pressure accumulation is localized and positive."""
        sensor = TactileDigitSensor(resolution=32, sigma_px=1.5, seed=0)
        pressure = np.zeros((32, 32), dtype=np.float32)

        sensor._add_gaussian_splat(pressure, px=16.0, py=16.0, weight=0.75)

        assert pressure[16, 16] > 0.7
        assert pressure[16, 16] == pytest.approx(float(pressure.max()), rel=1e-5)
        assert pressure[0, 0] < 1e-6

    def test_coord_to_pixel_maps_pad_center_to_image_center(self):
        """Test local tactile coordinates map consistently to image coordinates."""
        sensor = TactileDigitSensor(resolution=64)

        center = sensor._coord_to_pixel(0.0, 0.01)
        left_edge = sensor._coord_to_pixel(-0.01, 0.01)
        right_edge = sensor._coord_to_pixel(0.01, 0.01)

        assert center == pytest.approx((sensor.resolution - 1) / 2.0, rel=1e-5)
        assert left_edge == pytest.approx(0.0, abs=1e-5)
        assert right_edge == pytest.approx(sensor.resolution - 1, abs=1e-5)

    def test_geom_matching_prefers_pad_geometries(self):
        """Test robust left/right finger geom matching with fallbacks."""
        sensor = TactileDigitSensor()
        geom_names = [
            "robot0:left_finger_link_geom",
            "robot0:leftpad_geom",
            "robot0:right_finger_link_geom",
            "robot0:rightpad_geom",
        ]

        left_matches = sensor._match_geom_names(geom_names, "left")
        right_matches = sensor._match_geom_names(geom_names, "right")

        assert left_matches[0] == "robot0:leftpad_geom"
        assert right_matches[0] == "robot0:rightpad_geom"

    def test_resolve_finger_attachment_anchors_to_pad_geom(self):
        """Test the tactile frame is anchored on the pad geom, not an end-effector site."""
        sensor = TactileDigitSensor()

        class MockModel:
            def __init__(self):
                self.ngeom = 2
                self.nsite = 2
                self.geom_size = np.array(
                    [
                        [0.045, 0.003, 0.015],
                        [0.045, 0.003, 0.015],
                    ],
                    dtype=np.float64,
                )

            def geom(self, key):
                if isinstance(key, int):
                    names = ("leftpad_geom", "rightpad_geom")
                    return SimpleNamespace(name=names[key], id=key)
                mapping = {"leftpad_geom": 0, "rightpad_geom": 1}
                return SimpleNamespace(name=key, id=mapping[key])

            def site(self, key):
                if isinstance(key, int):
                    names = ("leftEndEffector", "rightEndEffector")
                    return SimpleNamespace(name=names[key], id=key)
                mapping = {"leftEndEffector": 0, "rightEndEffector": 1}
                return SimpleNamespace(name=key, id=mapping[key])

        env = SimpleNamespace(unwrapped=SimpleNamespace(model=MockModel()))

        left = sensor._resolve_finger_attachment(env, "left")
        right = sensor._resolve_finger_attachment(env, "right")

        assert left.anchor_name == "leftpad_geom"
        assert right.anchor_name == "rightpad_geom"
        assert left.anchor_site_id is None
        assert right.anchor_site_id is None
        assert left.half_extent_u == pytest.approx(0.045)
        assert left.half_extent_v == pytest.approx(0.015)

    def test_internal_contact_filter_uses_pad_y_normal(self):
        """Test internal contact classification follows the pad y-axis normal."""
        sensor = TactileDigitSensor()
        attachment = SimpleNamespace(geom_ids=(0,))
        frame = (np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64))
        opposite_frame = (np.array([0.0, 1.0, 0.0], dtype=np.float64), np.eye(3))

        internal_contact = SimpleNamespace(
            geom1=0,
            geom2=2,
            pos=np.array([0.0, 0.001, 0.0], dtype=np.float64),
        )
        external_contact = SimpleNamespace(
            geom1=0,
            geom2=2,
            pos=np.array([0.0, -0.01, 0.0], dtype=np.float64),
        )

        assert sensor._is_internal_pad_contact(
            contact=internal_contact,
            attachment=attachment,
            frame=frame,
            opposite_frame=opposite_frame,
            opposite_geom_ids=(1,),
        )
        assert not sensor._is_internal_pad_contact(
            contact=external_contact,
            attachment=attachment,
            frame=frame,
            opposite_frame=opposite_frame,
            opposite_geom_ids=(1,),
        )

    def test_sensor_with_environment(self):
        """Test reset, validate, update, and read on a real MetaWorld env."""
        env = gym.make("Meta-World/MT1", env_name="basketball-v3")
        sensor = TactileDigitSensor(resolution=32, noise_std=0.0, seed=5)

        try:
            env.reset(seed=42)
            assert sensor.validate(env)

            sensor.reset(env)
            sensor.update(env)
            reading = sensor.read()
            left, right = _split_tactile_reading(sensor, reading)

            assert reading.shape == (2 * 32 * 32 * 3,)
            assert reading.dtype == np.float32
            assert np.all(reading >= 0.0)
            assert np.all(reading <= 1.0)
            assert left.shape == (32, 32, 3)
            assert right.shape == (32, 32, 3)
            assert sensor._left_finger is not None
            assert sensor._right_finger is not None
        finally:
            env.close()

    def test_reset_initializes_reading_to_base_texture(self):
        """Test reset caches the no-contact tactile image."""
        env = gym.make("Meta-World/MT1", env_name="basketball-v3")
        sensor = TactileDigitSensor(resolution=16, noise_std=0.0, seed=11)

        try:
            env.reset(seed=42)
            sensor.reset(env)
            reading = sensor.read()
            left, right = _split_tactile_reading(sensor, reading)

            assert np.allclose(left, sensor._base_image)
            assert np.allclose(right, sensor._base_image)
        finally:
            env.close()

    def test_policy_rollout_produces_tactile_activation(self):
        """Test that a contact-rich rollout produces visible tactile changes."""
        sensor = TactileDigitSensor(resolution=32, noise_std=0.0, seed=7)
        frames = _collect_contact_rollout(sensor, steps=200)
        activations = np.array(
            [max(frame["left_activation"], frame["right_activation"]) for frame in frames],
            dtype=np.float32,
        )

        assert frames
        assert float(activations.max()) > 0.02

    def test_qualitative_visualization_file_creation(self, tmp_path: Path):
        """Create a qualitative MP4 for human inspection."""
        output_path = visualize_tactile_digit_rollout(
            tmp_path / "tactile_digit_rollout.mp4",
            steps=200,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0


if __name__ == "__main__":
    print("MetaWorld Tactile Digit Sensor Test\n")

    test_suite = TestTactileDigitSensor()

    for test_name in (
        "test_sensor_creation",
        "test_observation_space",
        "test_observation_space_unnormalized",
        "test_metadata",
        "test_read_before_reset_raises_error",
        "test_seed_makes_base_texture_deterministic",
        "test_render_tactile_image_zero_pressure_returns_base",
        "test_add_gaussian_splat_increases_local_pressure",
        "test_coord_to_pixel_maps_pad_center_to_image_center",
        "test_geom_matching_prefers_pad_geometries",
    ):
        try:
            getattr(test_suite, test_name)()
            print(f"✓ {test_name} passed")
        except Exception as exc:  # pragma: no cover - convenience entry point.
            print(f"✗ {test_name} failed: {exc}")

    print("\nCreating qualitative video...")
    output = visualize_tactile_digit_rollout("tactile_digit_rollout.mp4", steps=200)
    print(f"✓ Saved tactile video to {output}")
