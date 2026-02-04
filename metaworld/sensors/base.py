"""Base sensor abstraction for MetaWorld sensory system.

This module defines the core sensor interface that all sensors must implement.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from gymnasium import spaces

if TYPE_CHECKING:
    from metaworld.sawyer_xyz_env import SawyerXYZEnv


class SensorBase(abc.ABC):
    """Abstract base class for all sensors.

    All sensors in the sensory-aware MetaWorld system must inherit from this class
    and implement its abstract methods. This ensures a consistent interface for
    sensor lifecycle management, reading, and observation space definition.

    The sensor lifecycle consists of:
    1. Initialization (__init__): Configure sensor parameters
    2. Reset (reset): Initialize/reset sensor state when environment resets
    3. Update (update): Update sensor readings after each physics step
    4. Read (read): Return current sensor reading as numpy array

    Example:
        >>> class MyCustomSensor(SensorBase):
        ...     def __init__(self, param1: float):
        ...         self.param1 = param1
        ...         self._reading = None
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_custom_sensor"
        ...
        ...     def reset(self, env: SawyerXYZEnv) -> None:
        ...         self._reading = np.zeros(10)
        ...
        ...     def update(self, env: SawyerXYZEnv) -> None:
        ...         self._reading = self._compute_reading(env)
        ...
        ...     def read(self) -> npt.NDArray[np.float64]:
        ...         return self._reading
        ...
        ...     def get_observation_space(self) -> spaces.Space:
        ...         return spaces.Box(low=-np.inf, high=np.inf, shape=(10,))
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return unique identifier for this sensor.

        This name will be used as the key in the observation dictionary
        when multiple sensors are present.

        Returns:
            Unique string identifier for the sensor.

        Note:
            Two sensors with the same name cannot be used simultaneously
            in the same environment.
        """
        pass

    @abc.abstractmethod
    def reset(self, env: SawyerXYZEnv) -> None:
        """Reset sensor state when the environment resets.

        This method is called once per episode reset. Use it to:
        - Initialize internal state variables
        - Clear sensor histories/buffers
        - Validate sensor configuration against current environment state

        Args:
            env: The MetaWorld environment being reset. Provides access to
                env.data (MuJoCo state), env.model (MuJoCo model), and other
                environment properties.

        Raises:
            RuntimeError: If sensor cannot be properly initialized (e.g., required
                MuJoCo elements missing from XML).

        Note:
            This method should NOT modify the environment state, only read from it.
            Avoid computationally expensive operations here if possible.
        """
        pass

    @abc.abstractmethod
    def update(self, env: SawyerXYZEnv) -> None:
        """Update sensor readings after a physics step.

        This method is called after each env.step() to update the sensor's
        internal state based on the new simulator state. This is where the
        main sensor computation happens.

        Args:
            env: The MetaWorld environment after the physics step. The MuJoCo
                state (env.data) reflects the state AFTER the action was applied
                and mujoco.mj_forward() was called.

        Note:
            - This method should update internal state but not return anything
            - Actual sensor readings are retrieved via read()
            - For performance, consider caching computations and using lazy evaluation
            - This method should NOT modify the environment state
        """
        pass

    @abc.abstractmethod
    def read(self) -> npt.NDArray[np.float64]:
        """Return the current sensor reading.

        This method returns the sensor's current state as a numpy array.
        It should be fast (ideally O(1)) and idempotent - calling it multiple
        times should return the same value until update() is called again.

        Returns:
            Sensor reading as a 1D numpy array of float64. The shape must match
            the shape defined in get_observation_space().

        Note:
            - Always return a 1D flattened array, even for multi-dimensional sensors
            - The returned array should match the observation space bounds
            - For multi-dimensional data (e.g., images), flatten in C-order (row-major)
        """
        pass

    @abc.abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """Return the Gymnasium observation space for this sensor.

        This defines the shape, dtype, and bounds of the sensor's output.
        It will be used to construct the environment's overall observation space.

        Returns:
            A Gymnasium Space object describing valid sensor outputs. Typically
            a spaces.Box for continuous sensors, but can be other space types
            (Discrete, MultiBinary, etc.) if appropriate.

        Example:
            >>> def get_observation_space(self) -> spaces.Space:
            ...     # For a 64x64 depth image
            ...     return spaces.Box(
            ...         low=0.0,
            ...         high=10.0,  # max depth in meters
            ...         shape=(64 * 64,),  # flattened
            ...         dtype=np.float32
            ...     )

        Note:
            - The shape must match the flattened output of read()
            - Consider computational cost: smaller spaces = faster training
            - Use appropriate dtype (float32 often sufficient, saves memory vs float64)
        """
        pass

    def get_metadata(self) -> dict[str, Any]:
        """Return optional metadata about the sensor.

        Override this method to provide additional information about the sensor
        that might be useful for logging, debugging, or analysis. This metadata
        is not used by the core sensor system but can be accessed by users.

        Returns:
            Dictionary with arbitrary metadata. Common keys:
            - 'type': Sensor category (e.g., 'visual', 'tactile', 'force')
            - 'units': Physical units of measurement
            - 'frequency': Update rate in Hz
            - 'description': Human-readable description
            - 'version': Sensor implementation version

        Example:
            >>> def get_metadata(self) -> dict[str, Any]:
            ...     return {
            ...         'type': 'visual',
            ...         'units': 'meters',
            ...         'frequency': 30,
            ...         'description': 'Depth camera mounted on gripper'
            ...     }
        """
        return {}

    def validate(self, env: SawyerXYZEnv) -> bool:
        """Validate that this sensor is compatible with the given environment.

        Override this method to perform checks that the sensor can operate
        correctly with the given environment. Called once during sensor
        registration before any reset() or update() calls.

        Args:
            env: The environment to validate against.

        Returns:
            True if sensor is compatible, False otherwise.

        Example:
            >>> def validate(self, env: SawyerXYZEnv) -> bool:
            ...     # Check if required MuJoCo camera exists
            ...     try:
            ...         camera_id = env.model.camera(self.camera_name).id
            ...         return True
            ...     except KeyError:
            ...         return False

        Note:
            - This is optional; default implementation returns True
            - Returning False will prevent the sensor from being added
            - Prefer raising informative exceptions over returning False
        """
        return True

    def __repr__(self) -> str:
        """Return string representation of the sensor."""
        return f"{self.__class__.__name__}(name='{self.name}')"