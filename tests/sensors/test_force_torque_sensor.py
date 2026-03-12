"""Test script for ForceTorqueSensor.

This script tests the 6-axis contact wrench sensor on the gripper.

Usage:
    python -m pytest tests/sensors/test_force_torque_sensor.py -v
    # Or run directly:
    python tests/sensors/test_force_torque_sensor.py
"""

import csv
import logging
import gymnasium as gym
import numpy as np
import pytest

from metaworld.policies.sawyer_basketball_v3_policy import SawyerBasketballV3Policy
from metaworld.sensors.force_torque_sensor import ForceTorqueSensor

LOGGER = logging.getLogger(__name__)


class TestForceTorqueSensor:
    """Test suite for ForceTorqueSensor."""

    def test_sensor_creation(self):
        """Test that we can create a ForceTorqueSensor."""
        sensor = ForceTorqueSensor()
        assert sensor.geom_names == ("leftpad_geom", "rightpad_geom")
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
        env = gym.make("Meta-World/MT1", env_name="pick-place-v3")
        sensor = ForceTorqueSensor()

        env.reset(seed=42)
        sensor.reset(env)
        sensor.update(env)
        wrench = sensor.read()

        assert wrench.shape == (6,)
        assert wrench.dtype == np.float64
        assert np.all(np.isfinite(wrench))

        env.close()

    def test_wrench_nonzero_on_contact(self):
        """Test wrench becomes non-zero when monitored geoms make contact."""
        env = gym.make("Meta-World/MT1", env_name="basketball-v3")
        sensor = ForceTorqueSensor()
        policy = SawyerBasketballV3Policy()

        obs, _ = env.reset()
        sensor.reset(env)


        touched = False
        non_zero_wrench = False

        for step_idx in range(250):
            action = policy.get_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            sensor.update(env)
            wrench = sensor.read()

            for contact_idx in range(env.unwrapped.data.ncon):
                c = env.unwrapped.data.contact[contact_idx]
                if c.geom1 in sensor._geom_ids or c.geom2 in sensor._geom_ids:
                    touched = True
                    non_zero_wrench = np.linalg.norm(wrench[:3]) > 1e-8 or np.linalg.norm(wrench[3:]) > 1e-10
                    break

            if non_zero_wrench or terminated or truncated:
                assert (
                    step_idx > 0
                ), "Expected to have at least one step before contact occurs."
                LOGGER.info(
                    "Contact detected at step=%d wrench=%s",
                    step_idx,
                    np.array2string(wrench, precision=6),
                )
                break

        env.close()
        assert touched
        assert non_zero_wrench

    def test_directional_probe_logs_forces(self, tmp_path):
        """Probe workspace boundaries and log wrench data for inspection.

        The sequence pushes the end-effector toward:
        - table/downward contact
        - left edge
        - right edge
        - bottom edge
        - top edge
        """
        env = gym.make("Meta-World/MT1", env_name="reach-wall-v3")
        sensor = ForceTorqueSensor()
        env.reset(seed=7)
        sensor.reset(env)

        action_scale = float(env.unwrapped.action_scale)
        hand_low = np.asarray(env.unwrapped.hand_low, dtype=np.float64)
        hand_high = np.asarray(env.unwrapped.hand_high, dtype=np.float64)
        center = (hand_low + hand_high) / 2.0
        z_down = float(hand_low[2])

        targets = [
            ("down", np.array([center[0], center[1], z_down], dtype=np.float64)),
            ("left", np.array([hand_low[0], center[1], z_down], dtype=np.float64)),
            ("right", np.array([hand_high[0], center[1], z_down], dtype=np.float64)),
            ("bottom", np.array([center[0], hand_low[1], z_down], dtype=np.float64)),
            ("top", np.array([center[0], hand_high[1], z_down], dtype=np.float64)),
        ]

        records: list[dict[str, float | str | int]] = []
        max_force_norm_by_phase: dict[str, float] = {}
        step_idx = 0

        # Let dynamics settle first.
        for _ in range(10):
            env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
            sensor.update(env)

        for phase, target in targets:
            max_force_norm = 0.0
            for _ in range(70):
                tcp = np.asarray(env.unwrapped.tcp_center, dtype=np.float64)
                action_xyz = np.clip(
                    (target - tcp) / max(action_scale, 1e-8),
                    -1.0,
                    1.0,
                )
                action = np.array(
                    [action_xyz[0], action_xyz[1], action_xyz[2], 0.0],
                    dtype=np.float32,
                )
                env.step(action)
                sensor.update(env)
                wrench = sensor.read()

                force_norm = float(np.linalg.norm(wrench[:3]))
                torque_norm = float(np.linalg.norm(wrench[3:]))
                max_force_norm = max(max_force_norm, force_norm)

                tcp_after = np.asarray(env.unwrapped.tcp_center, dtype=np.float64)
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
                        "ncon": int(env.unwrapped.data.ncon),
                    }
                )
                step_idx += 1
            max_force_norm_by_phase[phase] = max_force_norm

        csv_path = tmp_path / "force_torque_directional_probe.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        env.close()
        LOGGER.info("Directional force/torque log written to: %s", csv_path)
        LOGGER.info("Peak force norm by phase: %s", max_force_norm_by_phase)

        assert len(records) > 0
        assert csv_path.exists()
        assert np.all(np.isfinite([r["force_norm"] for r in records]))
        # Downward push should create contact with the supporting surface.
        assert max_force_norm_by_phase["down"] > 1e-3
