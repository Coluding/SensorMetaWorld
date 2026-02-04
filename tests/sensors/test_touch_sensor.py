"""Test script for GripperTouchSensor.

This script tests the binary touch sensor on the gripper fingers.

Usage:
    python -m pytest tests/sensors/test_touch_sensor.py -v
    # Or run directly:
    python tests/sensors/test_touch_sensor.py
"""

import gymnasium as gym
import numpy as np
import pytest
from metaworld.policies.sawyer_basketball_v3_policy import SawyerBasketballV3Policy
from metaworld.sensors.tactile import GripperTouchSensor


class TestGripperTouchSensor:
    """Test suite for GripperTouchSensor."""

    def test_sensor_creation(self):
        """Test that we can create a GripperTouchSensor."""
        sensor = GripperTouchSensor()

        assert sensor.name == "gripper_touch_leftpad_geom_rightpad_geom"
        assert sensor._left_geom_name == "leftpad_geom"
        assert sensor._right_geom_name == "rightpad_geom"

    def test_observation_space(self):
        """Test that observation space is correctly defined."""
        sensor = GripperTouchSensor()

        obs_space = sensor.get_observation_space()
        assert obs_space.shape == (2,)
        assert obs_space.dtype == np.float32
        assert obs_space.low[0] == 0.0
        assert obs_space.high[0] == 1.0

    def test_metadata(self):
        """Test that sensor metadata is populated."""
        sensor = GripperTouchSensor()

        metadata = sensor.get_metadata()
        assert metadata["type"] == "tactile"
        assert metadata["subtype"] == "binary_touch"
        assert metadata["left_finger_geom"] == "leftpad_geom"
        assert metadata["right_finger_geom"] == "rightpad_geom"
        assert metadata["channels"] == 2
        assert metadata["units"] == "binary"

    def test_sensor_with_environment(self):
        """Test sensor with actual pick-place-v3 environment."""
        # Create environment
        env = gym.make("Meta-World/MT1", env_name="pick-place-v3")

        # Create sensor
        sensor = GripperTouchSensor()

        # Reset environment and sensor
        env.reset()
        sensor.reset(env)

        # Validate geometries exist
        assert sensor.validate(env)

        # Update sensor (check for contacts)
        sensor.update(env)

        # Read sensor data
        touch_data = sensor.read()
        assert touch_data.shape == (2,)
        assert touch_data.dtype == np.float64
        assert np.all(touch_data >= 0.0)
        assert np.all(touch_data <= 1.0)

        env.close()

    def test_read_before_reset_raises_error(self):
        """Test that reading before reset raises an error."""
        sensor = GripperTouchSensor()

        with pytest.raises(RuntimeError, match="read\\(\\) called before reset\\(\\)"):
            sensor.read()

    def test_custom_geometry_names(self):
        """Test sensor with custom geometry names."""
        sensor = GripperTouchSensor(
            left_geom_name="leftpad_geom", right_geom_name="rightpad_geom"
        )

        assert sensor._left_geom_name == "leftpad_geom"
        assert sensor._right_geom_name == "rightpad_geom"

    def test_touch_triggered_once(self):
        env = gym.make("Meta-World/MT1", env_name="basketball-v3")

        # Create sensor
        sensor = GripperTouchSensor()
        sensor.reset(env)
        policy = SawyerBasketballV3Policy()

        done = False
        obs, info = env.reset()
        touch_triggered = False
        while not done and not touch_triggered:
            a = policy.get_action(obs)
            obs, _, _, _, info = env.step(a)
            sensor.update(env)
            touch_triggered = np.any(sensor.read())
            done = int(info['success']) == 1

        assert touch_triggered
