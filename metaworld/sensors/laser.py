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
from mpl_toolkits.mplot3d import Axes3D
from numpy import dtype, floating, ndarray
from numpy._typing import _32Bit

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


    def read(self) -> ndarray[tuple[int, ...], dtype[floating[_32Bit]]]:
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



class Lidar3DSensor(SensorBase):
    def __init__(
            self,
            origin_site: str = "lidar_origin",
            num_vertical_layers: int = 16,
            num_horizontal_rays: int = 360,
            vertical_fov_degrees: float = 30.0,
            horizontal_fov_degrees: float = 360.0,
            max_range: float = 100.0,
    ):
        self.num_layers = num_vertical_layers
        self.num_horizontal = num_horizontal_rays
        self.total_rays = num_vertical_layers * num_horizontal_rays
        self.origin_site = origin_site
        self.vertical_fov = vertical_fov_degrees
        self.horizontal_fov = horizontal_fov_degrees
        self.max_range = max_range

        self._distances: npt.NDArray[np.float32] | None = None
        self._ray_dirs_local: npt.NDArray[np.float32] | None = None


    def reset(self, env):
        # Vertical angles: elevation from -15° to +15° (for 30° FOV)
        v_angles = np.linspace(
            -np.deg2rad(self.vertical_fov / 2),
            np.deg2rad(self.vertical_fov / 2),
            self.num_layers
        )

        # Horizontal angles: azimuth 0° to 360°
        h_angles = np.linspace(
            0, np.deg2rad(self.horizontal_fov),
            self.num_horizontal,
            endpoint=False
        )

        # Create ray directions in spherical coordinates
        rays = []
        for v_angle in v_angles:
            for h_angle in h_angles:
                x = np.cos(v_angle) * np.cos(h_angle)
                y = np.cos(v_angle) * np.sin(h_angle)
                z = np.sin(v_angle)
                rays.append([x, y, z])

        self._ray_dirs_local = np.array(rays, dtype=np.float64)
        self._distances = np.full(self.total_rays, self.max_range, dtype=np.float64)

    def read(self) -> ndarray[tuple[int, ...], dtype[floating[_32Bit]]]:
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
        return self._distances.reshape(self.num_layers, self.num_horizontal).copy()


    def get_observation_space(self) -> spaces.Space:
        """Return observation space for the 3D LiDAR sensor.

        Returns:
            Box space with shape (num_layers, num_horizontal) representing distance measurements.
        """
        return spaces.Box(
            low=0.0,
            high=self.max_range,
            shape=(self.num_layers, self.num_horizontal),
            dtype=np.float32,
        )

    def get_metadata(self) -> dict[str, str | int | float]:
        """Return metadata about this sensor."""
        return {
            "type": "lidar",
            "subtype": "3d",
            "total_rays": self.total_rays,
            "num_layers": self.num_layers,
            "num_horizontal": self.num_horizontal,
            "max_range": self.max_range,
            "vertical_fov": self.vertical_fov,
            "horizontal_fov": self.horizontal_fov,
            "origin_site": self.origin_site,
        }

    def validate(self, env: MujocoEnv) -> bool:
        try:
            env.unwrapped.model.site(self.origin_site).id
            return True
        except KeyError:
            return False

    @property
    def name(self) -> str:
        """Return unique identifier for this sensor."""
        return f"lidar3d_{self.origin_site}_{self.num_layers}x{self.num_horizontal}rays"

    def update(self, env: MujocoEnv) -> None:
        """Update sensor by casting rays in the environment.

        Args:
            env: The MetaWorld environment.
        """
        try:
            self._site_id = env.unwrapped.model.site(self.origin_site).id
        except KeyError:
            raise RuntimeError(f"Site {self.origin_site} not found.")

        model = env.unwrapped.model
        data = env.unwrapped.data
        origin = data.site_xpos[self._site_id].copy()
        rotation_mat = data.site_xmat[self._site_id].reshape(3, 3)
        ray_dirs_world = (rotation_mat @ self._ray_dirs_local.T).T

        # MuJoCo expects vec as 1D flattened array of shape (num_rays*3,)
        vec_flat = np.ascontiguousarray(ray_dirs_world).ravel()

        # Allocate geom_ids buffer if not already allocated
        if not hasattr(self, '_geom_ids') or self._geom_ids is None:
            self._geom_ids = np.zeros(self.total_rays, dtype=np.int32)

        mujoco.mj_multiRay(
            m=model,
            d=data,
            pnt=origin,
            vec=vec_flat,
            geomgroup=None,
            flg_static=1,
            bodyexclude=-1,
            geomid=self._geom_ids,
            dist=self._distances,
            nray=self.total_rays,
            cutoff=self.max_range,
        )

        # Replace -1 (no hit) with max_range
        self._distances[self._distances < 0] = self.max_range
        np.clip(self._distances, 0.0, self.max_range, out=self._distances)

    def visualize(self, subsample: int = 1):
        """Visualize the 3D LiDAR scan as a 3D point cloud.

        Args:
            subsample: Subsample factor for visualization (1 = all points, 2 = every other point, etc.)
                      Use higher values for faster rendering with dense scans.
        """
        if self._distances is None or self._ray_dirs_local is None:
            raise RuntimeError("Cannot visualize before update() has been called.")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Compute 3D endpoints of all rays
        endpoints = self._ray_dirs_local * self._distances[:, None]

        # Subsample for performance if needed
        if subsample > 1:
            endpoints = endpoints[::subsample]
            distances = self._distances[::subsample]
        else:
            distances = self._distances

        # Color points by distance (closer = blue, farther = red)
        colors = plt.cm.viridis(distances / self.max_range)

        # Plot as scatter (point cloud)
        ax.scatter(
            endpoints[:, 0],
            endpoints[:, 1],
            endpoints[:, 2],
            c=colors,
            s=1,
            alpha=0.6
        )

        # Plot origin
        ax.scatter([0], [0], [0], c='red', s=100, marker='o', label='LiDAR Origin')

        # Set equal aspect ratio
        max_range = np.max(np.abs(endpoints))
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D LiDAR Scan ({self.num_layers} layers × {self.num_horizontal} rays)')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()