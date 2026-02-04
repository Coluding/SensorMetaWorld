"""Sensor system for MetaWorld environments.

This module provides sensor abstractions for augmenting MetaWorld environments
with additional sensory modalities beyond proprioceptive observations.
"""

from metaworld.sensors.base import SensorBase
from metaworld.sensors.tactile import GripperTouchSensor
from metaworld.sensors.visual import DepthCameraSensor

__all__ = [
    "SensorBase",
    "GripperTouchSensor",
    "DepthCameraSensor",
]