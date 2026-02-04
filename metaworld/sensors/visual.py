"""Visual sensors for MetaWorld environments.

This module contains camera-based sensors including RGB, depth, and segmentation.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Literal
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
import cv2 as cv


from metaworld.sensors.base import SensorBase

if TYPE_CHECKING:
    from metaworld.sawyer_xyz_env import SawyerXYZEnv


class DepthCameraSensor(SensorBase):
    """Depth camera sensor using MuJoCo's offscreen rendering.

    This sensor renders depth images from a specified camera in the MuJoCo scene.
    The camera must be defined in the environment's XML file.

    Depth values represent distance from the camera to the nearest surface along
    each ray. By default, raw depth values are returned, but optional normalization
    and inversion are available.

    Args:
        camera_name: Name of the camera in the MuJoCo XML (must exist in model).
        height: Height of the rendered depth image in pixels.
        width: Width of the rendered depth image in pixels.
        normalize: If True, normalize depth values to [0, 1] using near/far planes.
        invert: If True, invert depth values (useful for some representations).
        near_plane: Near clipping plane distance (for normalization).
        far_plane: Far clipping plane distance (for normalization).

    Example:
        >>> # Add to environment XML:
        >>> # <camera name="gripper_depth" pos="0 0 0.05" quat="1 0 0 0"/>
        >>>
        >>> sensor = DepthCameraSensor(
        ...     camera_name="gripper_depth",
        ...     height=64,
        ...     width=64,
        ...     normalize=False  # raw depth values
        ... )
        >>> env = gym.make('Meta-World/MT1', env_name='reach-v3')
        >>> # TODO: Add sensor to env (requires SensorManager integration)

    Note:
        - Depth rendering is computationally expensive (typical overhead: 30-50%)
        - Consider reducing resolution for training (64x64 or 84x84)
        - Raw depth values are in MuJoCo's internal units (typically meters)
        - The camera must be defined in the XML before environment initialization
    """

    def __init__(
        self,
        camera_name: str,
        height: int = 64,
        width: int = 64,
        normalize: bool = False,
        invert: bool = False,
        near_plane: float = 0.01,
        far_plane: float = 10.0,
    ):
        """Initialize the depth camera sensor.

        Args:
            camera_name: Name of the camera in the MuJoCo XML.
            height: Image height in pixels.
            width: Image width in pixels.
            normalize: Whether to normalize depth to [0, 1].
            invert: Whether to invert depth values.
            near_plane: Near clipping distance for normalization.
            far_plane: Far clipping distance for normalization.
        """
        self.camera_name = camera_name
        self.height = height
        self.width = width
        self.normalize = normalize
        self.invert = invert
        self.near_plane = near_plane
        self.far_plane = far_plane

        # Internal state
        self._depth_buffer: npt.NDArray[np.float32] | None = None
        self._camera_id: int | None = None

    @property
    def name(self) -> str:
        """Return unique identifier for this sensor."""
        return f"depth_camera_{self.camera_name}"

    def reset(self, env: SawyerXYZEnv) -> None:
        """Reset sensor state and validate camera exists.

        Args:
            env: The MetaWorld environment.

        Raises:
            RuntimeError: If the specified camera is not found in the model.
        """
        # Validate camera exists
        try:
            self._camera_id = env.unwrapped.model.camera(self.camera_name).id
        except KeyError:
            raise RuntimeError(
                f"Camera '{self.camera_name}' not found in MuJoCo model. "
                f"Available cameras: {[env.unwrapped.model.camera(i).name for i in range(env.unwrapped.model.ncam)]}"
            )

        # Initialize depth buffer
        self._depth_buffer = np.zeros((self.height * self.width,), dtype=np.float32)

    def update(self, env: SawyerXYZEnv) -> None:
        """Render depth image from the camera.

        Args:
            env: The MetaWorld environment after physics step.

        Note:
            This uses MuJoCo's offscreen rendering which is relatively expensive.
            The depth buffer is cached until the next update() call.
        """
        renderer: mujoco.Renderer = env.unwrapped.mujoco_renderer
        viewer = OffScreenViewer(
            renderer.model,
            renderer.data,
            renderer.width,
            renderer.height,
            renderer.max_geom,
            renderer._vopt,
        )

        _depth_img = viewer.render("depth_array", self._camera_id, True)
        _depth_img = cv.resize(_depth_img, (self.height, self.width))

        if self.normalize:
            _depth_img = np.clip(_depth_img, self.near_plane, self.far_plane)
            _depth_img = (_depth_img - self.near_plane) / (self.far_plane - self.near_plane)

        if self.invert:
            if self.normalize:
                # Already normalized to [0, 1], just flip
                _depth_img = 1.0 - _depth_img
            else:
                # Raw depth: invert within actual range
                max_depth = _depth_img.max()
                _depth_img = max_depth - _depth_img

        self._depth_buffer = np.reshape(_depth_img, -1)

    def read(self) -> npt.NDArray[np.float64]:
        """Return current depth image as flattened array.

        Returns:
            Flattened depth image of shape (height * width,).

        Raises:
            RuntimeError: If read() is called before reset().
        """
        if self._depth_buffer is None:
            raise RuntimeError(
                f"Sensor '{self.name}' read() called before reset(). "
                "Call env.reset() first."
            )
        return self._depth_buffer.astype(np.float64)

    def get_observation_space(self) -> spaces.Space:
        """Return observation space for the depth image.

        Returns:
            Box space with shape (height * width,) representing flattened depth image.
        """
        if self.normalize:
            # Normalized depth is in [0, 1]
            low = 0.0
            high = 1.0
        else:
            # Raw depth can range from near to far plane
            low = self.near_plane
            high = self.far_plane

        return spaces.Box(
            low=low,
            high=high,
            shape=(self.height * self.width,),
            dtype=np.float32,
        )

    def get_metadata(self) -> dict[str, str | int | float]:
        """Return metadata about this depth camera."""
        return {
            "type": "visual",
            "subtype": "depth",
            "camera_name": self.camera_name,
            "resolution": f"{self.width}x{self.height}",
            "normalize": self.normalize,
            "invert": self.invert,
            "units": "meters" if not self.normalize else "normalized",
        }

    def validate(self, env: SawyerXYZEnv) -> bool:
        """Validate that the camera exists in the environment.

        Args:
            env: The environment to validate.

        Returns:
            True if camera exists, False otherwise.
        """
        try:
            env.unwrapped.model.camera(self.camera_name).id
            return True
        except KeyError:
            return False

    def get_depth_as_image(self) -> npt.NDArray[np.float32]:
        """Return depth buffer reshaped as 2D image (for visualization).

        Returns:
            Depth image of shape (height, width).

        Note:
            This is a utility method for debugging/visualization, not part of
            the observation returned to the agent.
        """
        if self._depth_buffer is None:
            raise RuntimeError("No depth data available. Call update() first.")
        return self._depth_buffer.reshape((self.height, self.width))

    def depth_to_color_image(self) -> npt.NDArray[np.uint8]:
        """Convert depth buffer to RGB colormap for visualization.

        Returns:
            RGB image of shape (height, width, 3) with depth encoded as color.
            Closer objects are darker, farther objects are lighter.

        Note:
            This is for visualization only. Uses matplotlib's 'viridis' colormap
            if available, otherwise grayscale.
        """
        if self._depth_buffer is None:
            raise RuntimeError("No depth data available. Call update() first.")

        # Reshape to 2D
        depth_image = self.get_depth_as_image()

        # Normalize to [0, 1] for colormap
        depth_min = depth_image.min()
        depth_max = depth_image.max()
        if depth_max > depth_min:
            depth_normalized = (depth_image - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_image)

        # Apply viridis colormap (or any other matplotlib colormap)
        cmap = plt.cm.viridis
        colored = cmap(depth_normalized)
        # Convert to uint8 RGB (drop alpha channel)
        return (colored[:, :, :3] * 255).astype(np.uint8)

