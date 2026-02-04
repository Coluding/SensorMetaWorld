"""Test script for Lidar2DSensor.

This script tests the 2D LiDAR sensor with raycasting in MuJoCo environments.

Usage:
    python -m pytest tests/sensors/test_lidar.py -v
    # Or run directly:
    python tests/sensors/test_lidar.py
"""

import gymnasium as gym
import numpy as np
import pytest

from metaworld.sensors.laser import Lidar2DSensor


class TestLidar2DSensor:
    """Test suite for Lidar2DSensor."""

    def test_sensor_creation(self):
        """Test that we can create a Lidar2DSensor."""
        sensor = Lidar2DSensor(
            origin_site="lidar_origin",
            num_rays=64,
            max_range=2.0,
            fov_degrees=360.0,
        )

        assert sensor.origin_site == "lidar_origin"
        assert sensor.num_rays == 64
        assert sensor.max_range == 2.0
        assert sensor.fov_degrees == 360.0
        assert sensor.name == "lidar2d_lidar_origin_64rays"

    def test_observation_space(self):
        """Test that observation space is correctly defined."""
        sensor = Lidar2DSensor(
            origin_site="lidar_origin",
            num_rays=128,
            max_range=5.0,
        )

        obs_space = sensor.get_observation_space()
        assert obs_space.shape == (128,)
        assert obs_space.dtype == np.float32
        assert obs_space.low[0] == 0.0
        assert obs_space.high[0] == 5.0
        assert np.all(obs_space.low == 0.0)
        assert np.all(obs_space.high == 5.0)

    def test_metadata(self):
        """Test that sensor metadata is populated."""
        sensor = Lidar2DSensor(
            origin_site="lidar_origin",
            num_rays=64,
            max_range=2.0,
            fov_degrees=270.0,
        )

        metadata = sensor.get_metadata()
        assert metadata["type"] == "lidar"
        assert metadata["subtype"] == "2d"
        assert metadata["num_rays"] == 64
        assert metadata["max_range"] == 2.0
        assert metadata["fov_degrees"] == 270.0
        assert metadata["origin_site"] == "lidar_origin"

    def test_read_before_reset_raises_error(self):
        """Test that reading before reset raises an error."""
        sensor = Lidar2DSensor(origin_site="lidar_origin")

        with pytest.raises(RuntimeError, match="read\\(\\) called before reset\\(\\)"):
            sensor.read()

    def test_sensor_with_environment(self):
        """Test sensor with actual environment."""
        # Create environment
        env = gym.make("Meta-World/MT1", env_name="reach-v3")

        # Create sensor
        sensor = Lidar2DSensor(
            origin_site="lidar_origin",
            num_rays=64,
            max_range=2.0,
        )

        # Reset environment and sensor
        env.reset()
        sensor.reset(env)

        # Validate site exists
        assert sensor.validate(env)

        # Update sensor (cast rays)
        sensor.update(env)

        # Read sensor data
        distances = sensor.read()
        assert distances.shape == (64,)
        assert distances.dtype == np.float64
        assert np.all(distances >= 0.0)
        assert np.all(distances <= 2.0)

        env.close()

    def test_ray_directions_generation(self):
        """Test that ray directions are generated correctly."""
        sensor = Lidar2DSensor(
            origin_site="lidar_origin",
            num_rays=4,
            max_range=1.0,
            fov_degrees=360.0,
        )

        env = gym.make("Meta-World/MT1", env_name="reach-v3")
        env.reset()
        sensor.reset(env)

        # For 4 rays at 0°, 90°, 180°, 270°, the local directions should be:
        # Ray 0: [1, 0, 0]   (0°)
        # Ray 1: [0, 1, 0]   (90°)
        # Ray 2: [-1, 0, 0]  (180°)
        # Ray 3: [0, -1, 0]  (270°)

        expected_dirs = np.array([
            [1.0, 0.0, 0.0],   # 0°
            [0.0, 1.0, 0.0],   # 90°
            [-1.0, 0.0, 0.0],  # 180°
            [0.0, -1.0, 0.0],  # 270°
        ])

        np.testing.assert_array_almost_equal(
            sensor._ray_dirs_local,
            expected_dirs,
            decimal=5,
        )

        env.close()

    def test_partial_fov(self):
        """Test sensor with partial field of view (not full 360°)."""
        sensor = Lidar2DSensor(
            origin_site="lidar_origin",
            num_rays=8,
            max_range=2.0,
            fov_degrees=180.0,  # Half circle
        )

        env = gym.make("Meta-World/MT1", env_name="reach-v3")
        env.reset()
        sensor.reset(env)

        # For 180° FOV with 8 rays, should span from 0° to 180°
        # First ray: 0°, last ray: 157.5° (180° / 8 = 22.5° spacing)
        angles = np.linspace(0, np.deg2rad(180.0), 8, endpoint=False)

        expected_x = np.cos(angles)
        expected_y = np.sin(angles)

        np.testing.assert_array_almost_equal(
            sensor._ray_dirs_local[:, 0],
            expected_x,
            decimal=5,
        )
        np.testing.assert_array_almost_equal(
            sensor._ray_dirs_local[:, 1],
            expected_y,
            decimal=5,
        )

        # Z component should be zero (2D lidar)
        assert np.all(sensor._ray_dirs_local[:, 2] == 0.0)

        env.close()

    def test_different_ray_counts(self):
        """Test sensor with different ray counts."""
        env = gym.make("Meta-World/MT1", env_name="reach-v3")
        env.reset()

        for num_rays in [16, 32, 64, 128, 256]:
            sensor = Lidar2DSensor(
                origin_site="lidar_origin",
                num_rays=num_rays,
                max_range=2.0,
            )

            sensor.reset(env)
            sensor.update(env)
            distances = sensor.read()

            assert distances.shape == (num_rays,)
            assert np.all(distances >= 0.0)
            assert np.all(distances <= 2.0)

        env.close()

    def test_max_range_behavior(self):
        """Test that distances are clamped to max_range."""
        env = gym.make("Meta-World/MT1", env_name="reach-v3")
        env.reset()

        sensor = Lidar2DSensor(
            origin_site="lidar_origin",
            num_rays=64,
            max_range=0.5,  # Very short range
        )

        sensor.reset(env)
        sensor.update(env)
        distances = sensor.read()

        # All distances should be <= max_range
        assert np.all(distances <= 0.5)

        env.close()

    def test_site_validation(self):
        """Test that sensor validates site existence."""
        env = gym.make("Meta-World/MT1", env_name="reach-v3")
        env.reset()

        # Valid site
        sensor_valid = Lidar2DSensor(origin_site="lidar_origin")
        assert sensor_valid.validate(env)

        # Invalid site
        sensor_invalid = Lidar2DSensor(origin_site="nonexistent_site")
        assert not sensor_invalid.validate(env)

        # Reset with invalid site should raise error
        with pytest.raises(RuntimeError, match="Site .* not found"):
            sensor_invalid.reset(env)

        env.close()

    def test_buffers_reused(self):
        """Test that internal buffers are reused, not reallocated."""
        env = gym.make("Meta-World/MT1", env_name="reach-v3")
        env.reset()

        sensor = Lidar2DSensor(
            origin_site="lidar_origin",
            num_rays=64,
        )

        sensor.reset(env)

        # Store buffer IDs
        distances_id = id(sensor._distances)
        geom_ids_id = id(sensor._geom_ids)
        ray_dirs_id = id(sensor._ray_dirs_local)

        # Update multiple times
        for _ in range(10):
            sensor.update(env)
            sensor.read()

        # Buffer IDs should remain the same (no reallocation)
        assert id(sensor._distances) == distances_id
        assert id(sensor._geom_ids) == geom_ids_id
        assert id(sensor._ray_dirs_local) == ray_dirs_id

        env.close()

    def test_consistency_across_updates(self):
        """Test that sensor gives consistent results for static scene."""
        env = gym.make("Meta-World/MT1", env_name="reach-v3")
        env.reset(seed=42)  # Fixed seed for determinism

        sensor = Lidar2DSensor(
            origin_site="lidar_origin",
            num_rays=64,
        )

        sensor.reset(env)

        # Take first measurement
        sensor.update(env)
        distances_1 = sensor.read().copy()

        # Take second measurement (no step, scene unchanged)
        sensor.update(env)
        distances_2 = sensor.read().copy()

        # Should be identical for static scene
        np.testing.assert_array_almost_equal(distances_1, distances_2, decimal=6)

        env.close()

