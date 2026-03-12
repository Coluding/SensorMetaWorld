"""Tests for FutureStateIKSensor.

Usage:
    python -m pytest tests/sensors/test_kinematic_sensor.py -v
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from metaworld.sensors.kinematic import FutureStateIKSensor


class TestFutureStateIKSensor:
    """Test suite for FutureStateIKSensor."""

    def test_sensor_creation(self):
        sensor = FutureStateIKSensor(prediction_horizon=5, ee_action_scale=0.5)

        assert sensor.prediction_horizon == 5
        assert sensor.ee_action_scale == 0.5
        assert sensor.name == "future_state_ik_h5"

    def test_observation_space(self):
        horizon = 4
        sensor = FutureStateIKSensor(prediction_horizon=horizon)

        obs_space = sensor.get_observation_space()
        # [arm 7 + ee 3 + converged 1] per step
        assert obs_space.shape == (horizon * (7 + 3 + 1),)
        assert obs_space.dtype == np.float64

    def test_metadata(self):
        sensor = FutureStateIKSensor(prediction_horizon=3)
        metadata = sensor.get_metadata()

        assert metadata["type"] == "kinematic"
        assert metadata["subtype"] == "future_state_ik"
        assert metadata["prediction_horizon"] == 3

    def test_set_action_sequence_shapes(self):
        sensor = FutureStateIKSensor(prediction_horizon=3)

        actions_xyz = np.array(
            [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], dtype=np.float64
        )
        sensor.set_action_sequence(actions_xyz)

        actions_mw = np.array(
            [[0.01, 0.0, 0.0, 0.0], [0.0, 0.01, 0.0, 0.0], [0.0, 0.0, 0.01, 0.0]], dtype=np.float64
        )
        sensor.set_action_sequence(actions_mw)

        with pytest.raises(ValueError, match="Expected actions shape"):
            sensor.set_action_sequence(np.zeros((2, 3), dtype=np.float64))

    def test_validate(self):
        env = gym.make("Meta-World/MT1", env_name="reach-v3")
        env.reset(seed=0)

        sensor_ok = FutureStateIKSensor(prediction_horizon=2)
        assert sensor_ok.validate(env)

        sensor_bad = FutureStateIKSensor(prediction_horizon=2, ee_body_name="not_a_body")
        assert not sensor_bad.validate(env)

        env.close()

    def test_prediction_on_environment(self):
        env = gym.make("Meta-World/MT1", env_name="reach-v3")
        env.reset(seed=1)

        horizon = 5
        sensor = FutureStateIKSensor(prediction_horizon=horizon)
        sensor.reset(env)

        actions = np.array(
            [
                [0.020, 0.000, 0.000],
                [0.010, 0.010, 0.005],
                [0.000, 0.015, 0.000],
                [-0.010, 0.000, 0.010],
                [0.000, -0.010, 0.000],
            ],
            dtype=np.float64,
        )

        sensor.set_action_sequence(actions)
        sensor.update(env)

        flat = sensor.read()
        pred = sensor.get_last_prediction()

        assert flat.shape == (horizon * (7 + 3 + 1),)

        assert pred["target_ee_positions"].shape == (horizon, 3)
        assert pred["achieved_ee_positions"].shape == (horizon, 3)
        assert pred["arm_joint_angles"].shape == (horizon, 7)
        assert pred["qpos_trajectory"].shape[0] == horizon
        assert pred["converged"].shape == (horizon,)
        assert pred["converged"].dtype == np.bool_

        ee_errors = np.linalg.norm(
            pred["target_ee_positions"] - pred["achieved_ee_positions"], axis=1
        )
        assert np.all(np.isfinite(ee_errors))
        assert np.max(ee_errors) < 1e-2

        env.close()
