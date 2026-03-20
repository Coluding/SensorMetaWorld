"""Tests for ImageTactileSensor."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest

from metaworld.policies.sawyer_basketball_v3_policy import SawyerBasketballV3Policy
from metaworld.sensors.tactile import ImageTactileSensor


class _MockModel:
    def __init__(
        self,
        geom_name_to_id: dict[str, int],
        geom_size: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self._geom_name_to_id = geom_name_to_id
        if geom_size is not None:
            self.geom_size = geom_size

    def geom(self, key: str | int) -> SimpleNamespace:
        if isinstance(key, str):
            if key not in self._geom_name_to_id:
                raise KeyError(key)
            return SimpleNamespace(name=key, id=self._geom_name_to_id[key])

        for name, geom_id in self._geom_name_to_id.items():
            if geom_id == key:
                return SimpleNamespace(name=name, id=geom_id)
        raise KeyError(key)


class _MockData:
    def __init__(
        self,
        contacts: list[SimpleNamespace],
        geom_xpos: npt.NDArray[np.float64],
        geom_xmat: npt.NDArray[np.float64],
    ) -> None:
        self.contact = contacts
        self.ncon = len(contacts)
        self.geom_xpos = geom_xpos
        self.geom_xmat = geom_xmat


class _MockEnv:
    def __init__(
        self,
        geom_name_to_id: dict[str, int],
        contacts: list[SimpleNamespace],
        geom_xpos: npt.NDArray[np.float64],
        geom_xmat: npt.NDArray[np.float64],
        geom_size: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.model = _MockModel(geom_name_to_id, geom_size=geom_size)
        self.data = _MockData(contacts, geom_xpos, geom_xmat)
        self.unwrapped = self


class TestImageTactileSensor:
    def test_sensor_creation_and_space(self) -> None:
        sensor = ImageTactileSensor(image_size=(32, 32))

        assert sensor.name == "gripper_image_tactile_32x32"
        obs_space = sensor.get_observation_space()
        assert obs_space.shape == (2 * 32 * 32,)
        assert obs_space.dtype == np.float32
        assert np.all(obs_space.low == 0.0)
        assert np.all(obs_space.high == 1.0)

    def test_metadata(self) -> None:
        sensor = ImageTactileSensor(image_size=(16, 20))
        metadata = sensor.get_metadata()
        assert metadata["type"] == "tactile"
        assert metadata["modality"] == "image_based_tactile"
        assert metadata["image_size"] == (16, 20)
        assert metadata["channels"] == 2

    def test_read_before_reset_raises(self) -> None:
        sensor = ImageTactileSensor()
        with pytest.raises(RuntimeError, match="read\\(\\) called before reset\\(\\)"):
            sensor.read()

    def test_validate_invalid_configuration_raises(self) -> None:
        sensor = ImageTactileSensor(image_size=(0, 32))
        mock_env = _MockEnv(
            geom_name_to_id={"leftpad_geom": 0, "rightpad_geom": 1},
            contacts=[],
            geom_xpos=np.zeros((2, 3), dtype=np.float64),
            geom_xmat=np.tile(np.eye(3, dtype=np.float64).reshape(1, 9), (2, 1)),
        )

        with pytest.raises(ValueError, match="image_size"):
            sensor.validate(mock_env)

    def test_validate_missing_geom_raises(self) -> None:
        env = gym.make("Meta-World/MT1", env_name="pick-place-v3")
        env.reset()
        sensor = ImageTactileSensor(
            left_geom_names=("missing_left_geom",),
            right_geom_names=("rightpad_geom",),
        )
        with pytest.raises(RuntimeError, match="missing_left_geom"):
            sensor.validate(env)
        env.close()

    def test_rasterization_with_mock_contacts(self) -> None:
        geom_name_to_id = {
            "leftpad_geom": 0,
            "rightpad_geom": 1,
            "obj_geom": 2,
        }
        contacts = [
            SimpleNamespace(
                geom1=0,
                geom2=2,
                pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                dist=-0.001,
            )
        ]
        geom_xpos = np.array(
            [
                [0.0, 0.0, 0.0],  # left
                [0.1, 0.0, 0.0],  # right
                [0.05, 0.0, 0.0],  # object (unused)
            ],
            dtype=np.float64,
        )
        geom_xmat = np.tile(np.eye(3, dtype=np.float64).reshape(1, 9), (3, 1))
        env = _MockEnv(geom_name_to_id, contacts, geom_xpos, geom_xmat)

        sensor = ImageTactileSensor(
            image_size=(16, 16),
            gaussian_sigma=1.2,
            force_scale=500.0,
            use_contact_force=False,
        )
        sensor.reset(env)
        sensor.update(env)

        reading = sensor.read()
        assert reading.shape == (2 * 16 * 16,)
        tactile_maps = reading.reshape(2, 16, 16)
        left_map = tactile_maps[0]
        right_map = tactile_maps[1]

        assert np.max(left_map) > 0.0
        assert np.sum(left_map) > 0.0
        assert np.allclose(right_map, 0.0)

    def test_basketball_env_contact_activation(self) -> None:
        env = gym.make("Meta-World/MT1", env_name="basketball-v3")
        sensor = ImageTactileSensor(image_size=(16, 16), use_contact_force=True)
        policy = SawyerBasketballV3Policy()

        obs, _ = env.reset(seed=42)
        sensor.reset(env)

        activated = False
        for _ in range(250):
            action = policy.get_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            sensor.update(env)
            tactile = sensor.read()
            if np.max(tactile) > 1e-6:
                activated = True
                break
            if terminated or truncated:
                break

        env.close()
        assert activated

    def test_auto_pad_half_extent_uses_geom_size(self) -> None:
        geom_name_to_id = {
            "leftpad_geom": 0,
            "rightpad_geom": 1,
            "obj_geom": 2,
        }
        contacts = [
            SimpleNamespace(
                geom1=0,
                geom2=2,
                # This point is outside configured half-extent x=0.015 but inside geom half-size x=0.045.
                pos=np.array([0.03, 0.0, 0.0], dtype=np.float64),
                dist=-0.001,
            )
        ]
        geom_xpos = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.05, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        geom_xmat = np.tile(np.eye(3, dtype=np.float64).reshape(1, 9), (3, 1))
        geom_size = np.array(
            [
                [0.045, 0.003, 0.015],  # left pad
                [0.045, 0.003, 0.015],  # right pad
                [0.010, 0.010, 0.010],  # object (unused)
            ],
            dtype=np.float64,
        )
        env = _MockEnv(
            geom_name_to_id,
            contacts,
            geom_xpos,
            geom_xmat,
            geom_size=geom_size,
        )

        sensor = ImageTactileSensor(
            image_size=(16, 16),
            pad_half_extent=(0.015, 0.008),
            auto_pad_half_extent=True,
            force_scale=500.0,
            use_contact_force=False,
        )
        sensor.reset(env)
        sensor.update(env)
        assert np.max(sensor.read()) > 0.0

    def test_inside_outside_split_maps(self) -> None:
        geom_name_to_id = {
            "leftpad_geom": 0,
            "rightpad_geom": 1,
            "obj_geom": 2,
        }
        contacts = [
            # Left-side inside contact (default left inside sign is negative y).
            SimpleNamespace(
                geom1=0,
                geom2=2,
                pos=np.array([0.0, -0.001, 0.0], dtype=np.float64),
                dist=-0.001,
            ),
            # Left-side outside contact.
            SimpleNamespace(
                geom1=0,
                geom2=2,
                pos=np.array([0.0, 0.001, 0.0], dtype=np.float64),
                dist=-0.001,
            ),
        ]
        geom_xpos = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.05, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        geom_xmat = np.tile(np.eye(3, dtype=np.float64).reshape(1, 9), (3, 1))
        env = _MockEnv(geom_name_to_id, contacts, geom_xpos, geom_xmat)

        sensor = ImageTactileSensor(
            image_size=(16, 16),
            pad_half_extent=(0.02, 0.02),
            force_scale=500.0,
            use_contact_force=False,
            normalize=True,
        )
        sensor.reset(env)
        sensor.update(env)

        assert float(np.max(sensor.left_inside_image)) > 0.0
        assert float(np.max(sensor.left_outside_image)) > 0.0
        assert np.allclose(sensor.right_inside_image, 0.0)
        assert np.allclose(sensor.right_outside_image, 0.0)
