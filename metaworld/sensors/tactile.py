"""Tactile sensors for MetaWorld environments.

This module contains touch and force sensors for gripper contact detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from metaworld.sensors.base import SensorBase

if TYPE_CHECKING:
    from metaworld.sawyer_xyz_env import SawyerXYZEnv

class GripperTouchSensor(SensorBase):
    """Binary touch sensor for gripper fingers.

    This sensor detects whether the left and right gripper fingers are in
    contact with any object in the environment. It returns a binary signal
    for each finger (1.0 = touching, 0.0 = not touching).

    Args:
        left_geom_name: Name of the left finger geometry in MuJoCo model.
        right_geom_name: Name of the right finger geometry in MuJoCo model.
    """

    def __init__(
        self,
        left_geom_name: str = "leftpad_geom",
        right_geom_name: str = "rightpad_geom",
    ):
        """Initialize the gripper touch sensor.

        Args:
            left_geom_name: Name of left finger geometry in MuJoCo XML.
            right_geom_name: Name of right finger geometry in MuJoCo XML.
        """
        self._left_geom_name = left_geom_name
        self._right_geom_name = right_geom_name

        # Internal state (set during reset)
        self._left_finger_id: int | None = None
        self._right_finger_id: int | None = None
        self._touch_buffer: npt.NDArray[np.float32] | None = None

    @property
    def name(self) -> str:
        """Return unique identifier for this sensor."""
        return f"gripper_touch_{self._left_geom_name}_{self._right_geom_name}"

    def reset(self, env: MujocoEnv) -> None:
        """Reset sensor state and validate geometries exist.

        Args:
            env: The MetaWorld environment.

        Raises:
            RuntimeError: If the specified geometries are not found in the model.
        """
        # Look up geometry IDs
        try:
            self._left_finger_id = env.unwrapped.model.geom(self._left_geom_name).id
        except KeyError:
            raise RuntimeError(
                f"Geometry '{self._left_geom_name}' not found in MuJoCo model. "
                f"Available geometries: {[env.unwrapped.model.geom(i).name for i in range(env.unwrapped.model.ngeom)]}"
            )

        try:
            self._right_finger_id = env.unwrapped.model.geom(self._right_geom_name).id
        except KeyError:
            raise RuntimeError(
                f"Geometry '{self._right_geom_name}' not found in MuJoCo model. "
                f"Available geometries: {[env.unwrapped.model.geom(i).name for i in range(env.unwrapped.model.ngeom)]}"
            )

        # Initialize touch buffer [left, right]
        self._touch_buffer = np.zeros(2, dtype=np.float32)

    def update(self, env: MujocoEnv) -> None:
        """Update touch readings by checking contact state.

        Args:
            env: The MetaWorld environment after physics step.

        Note:
            This iterates through all active contacts and checks if either
            finger geometry is involved.
        """
        # Reset touch state
        self._touch_buffer[0] = 0.0  # left finger
        self._touch_buffer[1] = 0.0  # right finger

        # Check all active contacts
        ncon = env.unwrapped.data.ncon
        for i in range(ncon):
            contact = env.unwrapped.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check if left finger is involved in this contact
            if geom1 == self._left_finger_id or geom2 == self._left_finger_id:
                self._touch_buffer[0] = 1.0

            # Check if right finger is involved in this contact
            if geom1 == self._right_finger_id or geom2 == self._right_finger_id:
                self._touch_buffer[1] = 1.0

            # Early exit if both fingers are touching
            if self._touch_buffer[0] == 1.0 and self._touch_buffer[1] == 1.0:
                break

    def read(self) -> npt.NDArray[np.float64]:
        """Return current touch readings.

        Returns:
            Array of shape (2,) with [left_touching, right_touching].
            Each value is either 0.0 (not touching) or 1.0 (touching).

        Raises:
            RuntimeError: If read() is called before reset().
        """
        if self._touch_buffer is None:
            raise RuntimeError(
                f"Sensor '{self.name}' read() called before reset(). "
                "Call env.reset() first."
            )
        return self._touch_buffer.astype(np.float64)

    def get_observation_space(self) -> spaces.Space:
        """Return observation space for the touch sensor.

        Returns:
            Box space with shape (2,) representing [left, right] touch state.
        """
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    def get_metadata(self) -> dict[str, str | int | float]:
        """Return metadata about this touch sensor."""
        return {
            "type": "tactile",
            "subtype": "binary_touch",
            "left_finger_geom": self._left_geom_name,
            "right_finger_geom": self._right_geom_name,
            "channels": 2,
            "units": "binary",
        }

    def validate(self, env: MujocoEnv) -> bool:
        """Validate that the finger geometries exist in the environment.

        Args:
            env: The environment to validate.

        Returns:
            True if both geometries exist, False otherwise.
        """
        try:
            env.unwrapped.model.geom(self._left_geom_name).id
            env.unwrapped.model.geom(self._right_geom_name).id
            return True
        except KeyError:
            return False