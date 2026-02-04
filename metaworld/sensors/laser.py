"""Laser sensors for MetaWorld environments.

This module contains ray casting sensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
import mujoco
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from metaworld.sensors.base import SensorBase

if TYPE_CHECKING:
    from metaworld.sawyer_xyz_env import SawyerXYZEnv

class Lidar2DSensor(SensorBase):
    def __init__(
        self,
        origin_site: str = "lidar_origin",
        num_rays: int = 64,
        max_range: float = 2.0,
        fov_degrees: float = 360.0,
        z_height: float = 0.0,
    ):
        self.origin_site = origin_site
        self.num_rays = num_rays
        self.max_range = max_range
        self.fov_degrees = fov_degrees
        self.z_height = z_height

        self._distances: npt.NDArray[np.float32] | None = None
        self._ray_dirs_local: npt.NDArray[np.float32] | None = None

    @property
    def name(self) -> str:
        """Return unique identifier for this sensor."""
        return f"lidar2d_{self.origin_site}_{self.num_rays}rays"

    def reset(self, env: MujocoEnv) -> None:
        """Reset sensor state and validate geometries exist.

        Args:
            env: The MetaWorld environment.

        Raises:
            RuntimeError: If the specified geometries are not found in the model.
        """
        try:
            self._site_id = env.unwrapped.model.site(self.origin_site).id
        except KeyError:
            raise RuntimeError(f"Site {self.origin_site} not found.")

        angles = np.linspace(
            0,
            np.deg2rad(self.fov_degrees),
            self.num_rays,
            endpoint=False
        )

        self._ray_dirs_local = np.zeros((self.num_rays, 3), dtype=np.float64)
        self._ray_dirs_local[:, 0] = np.cos(angles)
        self._ray_dirs_local[:, 1] = np.sin(angles)
        self._ray_dirs_local[:, 2] = 0.0

        self._distances = np.full(self.num_rays, self.max_range, dtype=np.float64)
        self._geom_ids = np.zeros(self.num_rays, dtype=np.int32)



    def update(self, env: MujocoEnv) -> None:
        model = env.unwrapped.model
        data = env.unwrapped.data
        origin = data.site_xpos[self._site_id].copy()
        rotation_mat = data.site_xmat[self._site_id].reshape(3, 3)
        ray_dirs_world = (rotation_mat @ self._ray_dirs_local.T).T

        # MuJoCo expects vec as 1D flattened array of shape (num_rays*3,)
        vec_flat = np.ascontiguousarray(ray_dirs_world).ravel()

        mujoco.mj_multiRay(
            m=model,
            d=data,
            pnt=origin,
            vec=vec_flat,  # Must be 1D array of shape (num_rays*3,)
            geomgroup=None,
            flg_static=1,
            bodyexclude=-1,
            geomid=self._geom_ids,
            dist=self._distances,
            nray=self.num_rays,
            cutoff=self.max_range,
        )

        # Replace -1 (no hit) with max_range
        self._distances[self._distances < 0] = self.max_range
        np.clip(self._distances, 0.0, self.max_range, out=self._distances)


    def read(self) -> npt.NDArray[np.float64]:
        """Return current distance measurements.

        Returns:
            Array of shape (num_rays,) with distance to nearest obstacle for each ray.

        Raises:
            RuntimeError: If read() is called before reset().
        """
        if self._distances is None:
            raise RuntimeError(
                f"Sensor '{self.name}' read() called before reset(). "
                "Call env.reset() first."
            )
        return self._distances.copy()

    def get_observation_space(self) -> spaces.Space:
        """Return observation space for the touch sensor.

        Returns:
            Box space with shape (2,) representing [left, right] touch state.
        """
        return spaces.Box(
            low=0.0,
            high=self.max_range,
            shape=(self.num_rays,),
            dtype=np.float32,
        )

    def get_metadata(self) -> dict[str, str | int | float]:
        """Return metadata about this sensor."""
        return {
            "type": "lidar",
            "subtype": "2d",
            "num_rays": self.num_rays,
            "max_range": self.max_range,  # Fixed typo: was "max_rangek"
            "fov_degrees": self.fov_degrees,
            "origin_site": self.origin_site,
        }

    def validate(self, env: MujocoEnv) -> bool:
        try:
            env.unwrapped.model.site(self.origin_site).id
            return True
        except KeyError:
            return False

    def visualize(self):
        """Visualize the LiDAR scan as a 2D plot."""
        fig, ax = plt.subplots(figsize=(10, 10))

        vecs = self._ray_dirs_local[:, :2] * self._distances[:, None]

        ax.quiver(
            np.zeros(len(vecs)), np.zeros(len(vecs)),
            vecs[:, 0], vecs[:, 1],
            angles='xy', scale_units='xy', scale=1
        )

        max_len = np.max(np.linalg.norm(vecs, axis=1))
        ax.set_xlim(-1.1 * max_len, 1.1 * max_len)
        ax.set_ylim(-1.1 * max_len, 1.1 * max_len)  # Fixed: was plt.set_ylim

        ax.set_aspect('equal')
        ax.grid(True)
        plt.show()
