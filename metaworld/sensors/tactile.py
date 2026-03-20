"""Tactile sensors for MetaWorld environments.

This module contains tactile sensors derived from MuJoCo contact information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import mujoco
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

from metaworld.sensors.base import SensorBase

if TYPE_CHECKING:
    from metaworld.sawyer_xyz_env import SawyerXYZEnv


_Side = Literal["left", "right"]


@dataclass(frozen=True)
class _GeomMissingReport:
    """Container for missing geom names by side."""

    left_missing: tuple[str, ...]
    right_missing: tuple[str, ...]

    @property
    def has_missing(self) -> bool:
        return bool(self.left_missing or self.right_missing)

    def to_error_message(self) -> str:
        missing_chunks: list[str] = []
        if self.left_missing:
            missing_chunks.append(f"left={self.left_missing}")
        if self.right_missing:
            missing_chunks.append(f"right={self.right_missing}")
        return (
            "Configured left/right tactile geom names were not found in the MuJoCo model: "
            + ", ".join(missing_chunks)
        )


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
    ) -> None:
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
        except KeyError as exc:
            raise RuntimeError(
                f"Geometry '{self._left_geom_name}' not found in MuJoCo model. "
                f"Available geometries: {[env.unwrapped.model.geom(i).name for i in range(env.unwrapped.model.ngeom)]}"
            ) from exc

        try:
            self._right_finger_id = env.unwrapped.model.geom(self._right_geom_name).id
        except KeyError as exc:
            raise RuntimeError(
                f"Geometry '{self._right_geom_name}' not found in MuJoCo model. "
                f"Available geometries: {[env.unwrapped.model.geom(i).name for i in range(env.unwrapped.model.ngeom)]}"
            ) from exc

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
        if self._touch_buffer is None:
            raise RuntimeError(
                f"Sensor '{self.name}' update() called before reset(). "
                "Call env.reset() and sensor.reset(env) first."
            )

        # Reset touch state
        self._touch_buffer[0] = 0.0  # left finger
        self._touch_buffer[1] = 0.0  # right finger

        # Check all active contacts
        ncon = env.unwrapped.data.ncon
        for i in range(ncon):
            contact = env.unwrapped.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

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


class ImageTactileSensor(SensorBase):
    """Image-based tactile sensor for Sawyer gripper inner pads.

    The sensor builds two synthetic tactile maps (left/right) from MuJoCo
    contacts involving configured finger geometries. Each contact contributes
    a Gaussian blob in the selected local pad plane of the corresponding finger.

    The final observation is always flattened as:
    `[left_image_flattened, right_image_flattened]`.

    Example:
        >>> sensor = ImageTactileSensor(
        ...     image_size=(32, 32),
        ...     left_geom_names=("leftpad_geom",),
        ...     right_geom_names=("rightpad_geom",),
        ... )
        >>> env = gym.make("Meta-World/MT1", env_name="basketball-v3")
        >>> env.reset()
        >>> sensor.validate(env)
        >>> sensor.reset(env)
        >>> sensor.update(env)
        >>> tactile = sensor.read()  # shape=(2 * 32 * 32,)
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (32, 32),
        left_geom_names: tuple[str, ...] | list[str] = ("leftpad_geom",),
        right_geom_names: tuple[str, ...] | list[str] = ("rightpad_geom",),
        pad_half_extent: tuple[float, float] = (0.015, 0.008),
        auto_pad_half_extent: bool = True,
        gaussian_sigma: float = 1.5,
        force_scale: float = 0.1,
        clip_value: float = 5.0,
        normalize: bool = False,
        use_contact_force: bool = True,
        include_depth_falloff: bool = False,
        tactile_plane_axes: tuple[int, int] = (0, 2),
        tactile_normal_axis: int = 1,
        inside_normal_sign_by_side: tuple[float, float] = (-1.0, 1.0),
        post_smoothing_sigma: float = 0.0,
    ) -> None:
        """Initialize the image-based tactile sensor.

        Args:
            image_size: `(height, width)` per finger tactile map.
            left_geom_names: MuJoCo geom names associated with left finger pad.
            right_geom_names: MuJoCo geom names associated with right finger pad.
            pad_half_extent: Half-extent `(half_width, half_height)` in local
                finger coordinates for mapping contacts to image pixels.
            auto_pad_half_extent: If True, derive per-side extents from MuJoCo
                geom half-sizes along the selected tactile axes and use the
                larger of `(configured, derived)` for robust coverage.
            gaussian_sigma: Standard deviation in pixels for per-contact Gaussian blobs.
            force_scale: Multiplicative scale applied to contact strength.
            clip_value: Upper bound used during clipping (and normalization when enabled).
            normalize: If True, outputs are normalized to [0, 1].
            use_contact_force: If True, use `mujoco.mj_contactForce` for strength.
                If False, fallback to penetration-depth proxy.
            include_depth_falloff: If True, attenuate contact amplitude based on
                distance from pad plane in local normal direction.
            tactile_plane_axes: Pair of local axes used as tactile width/height.
                Values must be distinct and in `{0, 1, 2}`.
            tactile_normal_axis: Local axis index used as pad normal.
            inside_normal_sign_by_side: Sign convention for local normal-axis
                coordinate classification `(left_sign, right_sign)`.
                A contact is considered "inside" when
                `local_normal * side_sign >= 0`, otherwise "outside".
                Defaults match the common MetaWorld gripper XML convention.
            post_smoothing_sigma: Optional final Gaussian smoothing in pixel units.
        """
        self.image_size = image_size
        self.left_geom_names = tuple(left_geom_names)
        self.right_geom_names = tuple(right_geom_names)
        self.pad_half_extent = pad_half_extent
        self.auto_pad_half_extent = auto_pad_half_extent
        self.gaussian_sigma = gaussian_sigma
        self.force_scale = force_scale
        self.clip_value = clip_value
        self.normalize = normalize
        self.use_contact_force = use_contact_force
        self.include_depth_falloff = include_depth_falloff
        self.tactile_plane_axes = tactile_plane_axes
        self.tactile_normal_axis = tactile_normal_axis
        self.inside_normal_sign_by_side = inside_normal_sign_by_side
        self.post_smoothing_sigma = post_smoothing_sigma

        self._left_geom_ids: set[int] = set()
        self._right_geom_ids: set[int] = set()
        self._left_image: npt.NDArray[np.float32] | None = None
        self._right_image: npt.NDArray[np.float32] | None = None
        self._left_inside_image: npt.NDArray[np.float32] | None = None
        self._left_outside_image: npt.NDArray[np.float32] | None = None
        self._right_inside_image: npt.NDArray[np.float32] | None = None
        self._right_outside_image: npt.NDArray[np.float32] | None = None
        self._reading: npt.NDArray[np.float32] | None = None
        self._pixel_grid_x: npt.NDArray[np.float32] | None = None
        self._pixel_grid_y: npt.NDArray[np.float32] | None = None
        self._contact_wrench_tmp = np.zeros(6, dtype=np.float64)
        self._left_pad_half_extent_effective: tuple[float, float] = self.pad_half_extent
        self._right_pad_half_extent_effective: tuple[float, float] = (
            self.pad_half_extent
        )

    @property
    def name(self) -> str:
        """Return unique identifier for this sensor."""
        height, width = self.image_size
        return f"gripper_image_tactile_{height}x{width}"

    def validate(self, env: SawyerXYZEnv) -> bool:
        """Validate sensor configuration and geom availability.

        Args:
            env: The environment to validate against.

        Returns:
            True when validation succeeds.

        Raises:
            ValueError: If configuration values are invalid.
            RuntimeError: If required geoms are missing in the MuJoCo model.
        """
        self._validate_configuration()
        model = self._unwrap_env(env).model
        missing_report = self._check_missing_geoms(model)
        if missing_report.has_missing:
            raise RuntimeError(missing_report.to_error_message())
        return True

    def reset(self, env: SawyerXYZEnv) -> None:
        """Reset sensor buffers and resolve finger geom ids.

        Args:
            env: MetaWorld environment.

        Raises:
            ValueError: If configuration values are invalid.
            RuntimeError: If configured geom names are not found.
        """
        self.validate(env)
        model = self._unwrap_env(env).model
        self._left_geom_ids = self._resolve_geom_ids(model, self.left_geom_names)
        self._right_geom_ids = self._resolve_geom_ids(model, self.right_geom_names)
        (
            self._left_pad_half_extent_effective,
            self._right_pad_half_extent_effective,
        ) = self._resolve_effective_pad_half_extents(model)

        height, width = self.image_size
        self._left_image = np.zeros((height, width), dtype=np.float32)
        self._right_image = np.zeros((height, width), dtype=np.float32)
        self._left_inside_image = np.zeros((height, width), dtype=np.float32)
        self._left_outside_image = np.zeros((height, width), dtype=np.float32)
        self._right_inside_image = np.zeros((height, width), dtype=np.float32)
        self._right_outside_image = np.zeros((height, width), dtype=np.float32)
        self._reading = np.zeros(2 * height * width, dtype=np.float32)
        yy, xx = np.mgrid[0:height, 0:width]
        self._pixel_grid_x = xx.astype(np.float32)
        self._pixel_grid_y = yy.astype(np.float32)

    def update(self, env: SawyerXYZEnv) -> None:
        """Update tactile images from current MuJoCo contacts.

        Args:
            env: MetaWorld environment after physics update.

        Raises:
            RuntimeError: If called before reset.
        """
        if (
            self._left_image is None
            or self._right_image is None
            or self._left_inside_image is None
            or self._left_outside_image is None
            or self._right_inside_image is None
            or self._right_outside_image is None
            or self._reading is None
            or self._pixel_grid_x is None
            or self._pixel_grid_y is None
        ):
            raise RuntimeError(
                f"Sensor '{self.name}' update() called before reset(). "
                "Call env.reset() and sensor.reset(env) first."
            )

        unwrapped = self._unwrap_env(env)
        model = unwrapped.model
        data = unwrapped.data

        self._left_image.fill(0.0)
        self._right_image.fill(0.0)
        self._left_inside_image.fill(0.0)
        self._left_outside_image.fill(0.0)
        self._right_inside_image.fill(0.0)
        self._right_outside_image.fill(0.0)

        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            side, finger_geom_id = self._classify_contact(contact)
            if side is None or finger_geom_id is None:
                continue

            strength = self._compute_contact_strength(
                model, data, contact_index, contact
            )
            if strength <= 0.0:
                continue

            local_point = self._contact_point_world_to_local(
                data=data,
                geom_id=finger_geom_id,
                point_world=np.asarray(contact.pos, dtype=np.float64),
            )
            uv_local = self._local_to_pad_uv(local_point, side=side)
            if uv_local is None:
                continue

            amplitude = strength * self.force_scale
            if self.include_depth_falloff:
                amplitude *= self._depth_falloff(local_point, side=side)
            if amplitude <= 0.0:
                continue

            contact_region = self._classify_inside_outside(local_point, side=side)
            if side == "left":
                self._accumulate_gaussian_contact(
                    self._left_image,
                    uv_local,
                    amplitude,
                    side="left",
                )
                target_split = (
                    self._left_inside_image
                    if contact_region == "inside"
                    else self._left_outside_image
                )
                self._accumulate_gaussian_contact(
                    target_split,
                    uv_local,
                    amplitude,
                    side="left",
                )
            else:
                self._accumulate_gaussian_contact(
                    self._right_image,
                    uv_local,
                    amplitude,
                    side="right",
                )
                target_split = (
                    self._right_inside_image
                    if contact_region == "inside"
                    else self._right_outside_image
                )
                self._accumulate_gaussian_contact(
                    target_split,
                    uv_local,
                    amplitude,
                    side="right",
                )

        if self.post_smoothing_sigma > 0.0:
            self._left_image[:] = gaussian_filter(
                self._left_image, sigma=self.post_smoothing_sigma
            )
            self._right_image[:] = gaussian_filter(
                self._right_image, sigma=self.post_smoothing_sigma
            )
            self._left_inside_image[:] = gaussian_filter(
                self._left_inside_image, sigma=self.post_smoothing_sigma
            )
            self._left_outside_image[:] = gaussian_filter(
                self._left_outside_image, sigma=self.post_smoothing_sigma
            )
            self._right_inside_image[:] = gaussian_filter(
                self._right_inside_image, sigma=self.post_smoothing_sigma
            )
            self._right_outside_image[:] = gaussian_filter(
                self._right_outside_image, sigma=self.post_smoothing_sigma
            )

        self._postprocess_images()
        self._reading[: self._left_image.size] = self._left_image.reshape(-1)
        self._reading[self._left_image.size :] = self._right_image.reshape(-1)

    def read(self) -> npt.NDArray[np.float64]:
        """Return flattened tactile observation.

        Returns:
            Flattened vector containing `[left_map, right_map]`.
        """
        if self._reading is None:
            raise RuntimeError(
                f"Sensor '{self.name}' read() called before reset(). "
                "Call env.reset() first."
            )
        return self._reading.astype(np.float64, copy=True)

    def get_observation_space(self) -> spaces.Space:
        """Return tactile observation space."""
        height, width = self.image_size
        high = 1.0 if self.normalize else max(self.clip_value, 0.0)
        return spaces.Box(
            low=0.0,
            high=high,
            shape=(2 * height * width,),
            dtype=np.float32,
        )

    def get_metadata(self) -> dict[str, object]:
        """Return metadata about this tactile sensor."""
        return {
            "type": "tactile",
            "modality": "image_based_tactile",
            "image_size": self.image_size,
            "channels": 2,
            "description": (
                "Two synthetic tactile maps from gripper pad contacts "
                "(left/right), flattened as a single vector."
            ),
            "left_geom_names": self.left_geom_names,
            "right_geom_names": self.right_geom_names,
            "pad_half_extent": self.pad_half_extent,
            "auto_pad_half_extent": self.auto_pad_half_extent,
            "left_pad_half_extent_effective": self._left_pad_half_extent_effective,
            "right_pad_half_extent_effective": self._right_pad_half_extent_effective,
            "inside_normal_sign_by_side": self.inside_normal_sign_by_side,
            "left_surface_dimensions_m": self._get_surface_dimensions_for_side("left"),
            "right_surface_dimensions_m": self._get_surface_dimensions_for_side(
                "right"
            ),
        }

    @property
    def left_image(self) -> npt.NDArray[np.float32]:
        """Return latest left tactile image (2D)."""
        if self._left_image is None:
            raise RuntimeError("Sensor has not been reset yet.")
        return self._left_image.copy()

    @property
    def right_image(self) -> npt.NDArray[np.float32]:
        """Return latest right tactile image (2D)."""
        if self._right_image is None:
            raise RuntimeError("Sensor has not been reset yet.")
        return self._right_image.copy()

    @property
    def left_inside_image(self) -> npt.NDArray[np.float32]:
        """Return latest left inside-contact map."""
        if self._left_inside_image is None:
            raise RuntimeError("Sensor has not been reset yet.")
        return self._left_inside_image.copy()

    @property
    def left_outside_image(self) -> npt.NDArray[np.float32]:
        """Return latest left outside-contact map."""
        if self._left_outside_image is None:
            raise RuntimeError("Sensor has not been reset yet.")
        return self._left_outside_image.copy()

    @property
    def right_inside_image(self) -> npt.NDArray[np.float32]:
        """Return latest right inside-contact map."""
        if self._right_inside_image is None:
            raise RuntimeError("Sensor has not been reset yet.")
        return self._right_inside_image.copy()

    @property
    def right_outside_image(self) -> npt.NDArray[np.float32]:
        """Return latest right outside-contact map."""
        if self._right_outside_image is None:
            raise RuntimeError("Sensor has not been reset yet.")
        return self._right_outside_image.copy()

    def _validate_configuration(self) -> None:
        height, width = self.image_size
        if height <= 0 or width <= 0:
            raise ValueError(
                f"image_size must be positive (height, width), got {self.image_size}."
            )
        if len(self.left_geom_names) == 0 or len(self.right_geom_names) == 0:
            raise ValueError(
                "left_geom_names and right_geom_names must both be non-empty."
            )
        if self.pad_half_extent[0] <= 0.0 or self.pad_half_extent[1] <= 0.0:
            raise ValueError(
                f"pad_half_extent must contain positive values, got {self.pad_half_extent}."
            )
        if self.gaussian_sigma <= 0.0:
            raise ValueError(f"gaussian_sigma must be > 0, got {self.gaussian_sigma}.")
        if self.force_scale < 0.0:
            raise ValueError(f"force_scale must be >= 0, got {self.force_scale}.")
        if self.clip_value <= 0.0:
            raise ValueError(f"clip_value must be > 0, got {self.clip_value}.")
        if self.post_smoothing_sigma < 0.0:
            raise ValueError(
                f"post_smoothing_sigma must be >= 0, got {self.post_smoothing_sigma}."
            )

        axis_u, axis_v = self.tactile_plane_axes
        if axis_u == axis_v:
            raise ValueError("tactile_plane_axes must contain two distinct axes.")
        if axis_u not in (0, 1, 2) or axis_v not in (0, 1, 2):
            raise ValueError(
                f"tactile_plane_axes must be in {{0,1,2}}, got {self.tactile_plane_axes}."
            )
        if self.tactile_normal_axis not in (0, 1, 2):
            raise ValueError(
                f"tactile_normal_axis must be in {{0,1,2}}, got {self.tactile_normal_axis}."
            )
        if self.tactile_normal_axis in self.tactile_plane_axes:
            raise ValueError(
                "tactile_normal_axis must be distinct from tactile_plane_axes."
            )
        if len(self.inside_normal_sign_by_side) != 2:
            raise ValueError(
                "inside_normal_sign_by_side must contain exactly 2 values."
            )
        if abs(float(self.inside_normal_sign_by_side[0])) <= 0.0:
            raise ValueError("inside_normal_sign_by_side[0] must be non-zero.")
        if abs(float(self.inside_normal_sign_by_side[1])) <= 0.0:
            raise ValueError("inside_normal_sign_by_side[1] must be non-zero.")

    @staticmethod
    def _unwrap_env(env: SawyerXYZEnv) -> SawyerXYZEnv:
        """Return unwrapped environment for direct `model` and `data` access."""
        unwrapped = getattr(env, "unwrapped", None)
        return unwrapped if unwrapped is not None else env

    @staticmethod
    def _resolve_single_geom_id(model: mujoco.MjModel, name: str) -> int:
        """Resolve a geom name to geom id with robust API fallback."""
        try:
            return int(model.geom(name).id)
        except KeyError as exc:
            try:
                geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
            except (TypeError, AttributeError):
                raise KeyError(name) from exc
            if geom_id < 0:
                raise KeyError(name) from exc
            return geom_id

    def _resolve_geom_ids(
        self, model: mujoco.MjModel, geom_names: tuple[str, ...]
    ) -> set[int]:
        """Resolve all geom names for one side into a geom-id set."""
        ids: set[int] = set()
        for name in geom_names:
            ids.add(self._resolve_single_geom_id(model, name))
        return ids

    def _check_missing_geoms(self, model: mujoco.MjModel) -> _GeomMissingReport:
        """Return missing geom names by side."""
        left_missing: list[str] = []
        right_missing: list[str] = []

        for name in self.left_geom_names:
            try:
                self._resolve_single_geom_id(model, name)
            except KeyError:
                left_missing.append(name)

        for name in self.right_geom_names:
            try:
                self._resolve_single_geom_id(model, name)
            except KeyError:
                right_missing.append(name)

        return _GeomMissingReport(
            left_missing=tuple(left_missing),
            right_missing=tuple(right_missing),
        )

    def _classify_contact(
        self, contact: mujoco.MjContact
    ) -> tuple[_Side | None, int | None]:
        """Classify contact as left/right finger contact.

        Returns:
            Tuple `(side, finger_geom_id)`. If not a valid tactile contact,
            returns `(None, None)`.
        """
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        geom1_left = geom1 in self._left_geom_ids
        geom2_left = geom2 in self._left_geom_ids
        geom1_right = geom1 in self._right_geom_ids
        geom2_right = geom2 in self._right_geom_ids

        # Ignore contacts where both sides are present (typically finger-finger contact).
        if (geom1_left or geom2_left) and (geom1_right or geom2_right):
            return None, None

        if geom1_left:
            return "left", geom1
        if geom2_left:
            return "left", geom2
        if geom1_right:
            return "right", geom1
        if geom2_right:
            return "right", geom2
        return None, None

    def _compute_contact_strength(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        contact_index: int,
        contact: mujoco.MjContact,
    ) -> float:
        """Compute scalar strength for one contact."""
        if self.use_contact_force:
            self._contact_wrench_tmp.fill(0.0)
            mujoco.mj_contactForce(model, data, contact_index, self._contact_wrench_tmp)
            # First three terms are force in contact frame.
            return float(np.linalg.norm(self._contact_wrench_tmp[:3]))

        # Contact distance is negative under penetration. Convert to positive scale.
        return max(0.0, float(-contact.dist))

    def _contact_point_world_to_local(
        self,
        data: mujoco.MjData,
        geom_id: int,
        point_world: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Transform world-space point into local frame of reference geom."""
        geom_pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        geom_rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        # Local = R^T (p_world - t).
        return geom_rot.T @ (point_world - geom_pos)

    def _local_to_pad_uv(
        self, local_point: npt.NDArray[np.float64], side: _Side
    ) -> tuple[float, float] | None:
        """Project local 3D point onto tactile pad plane and bounds-check it."""
        axis_u, axis_v = self.tactile_plane_axes
        u = float(local_point[axis_u])
        v = float(local_point[axis_v])
        half_u, half_v = self._get_half_extent_for_side(side)
        if abs(u) > half_u or abs(v) > half_v:
            return None
        return u, v

    def _depth_falloff(
        self, local_point: npt.NDArray[np.float64], side: _Side
    ) -> float:
        """Optional attenuation based on normal-axis offset from tactile plane."""
        normal_distance = abs(float(local_point[self.tactile_normal_axis]))
        depth_sigma = max(min(self._get_half_extent_for_side(side)) * 0.5, 1e-6)
        return float(np.exp(-0.5 * (normal_distance / depth_sigma) ** 2))

    def _accumulate_gaussian_contact(
        self,
        tactile_image: npt.NDArray[np.float32],
        uv_local: tuple[float, float],
        amplitude: float,
        side: _Side,
    ) -> None:
        """Rasterize one contact as a Gaussian blob on a tactile image."""
        if self._pixel_grid_x is None or self._pixel_grid_y is None:
            raise RuntimeError("Pixel grids are unavailable. Call reset() first.")

        u, v = uv_local
        half_u, half_v = self._get_half_extent_for_side(side)
        height, width = self.image_size
        # Map local pad coordinates to pixel coordinates.
        x = (u + half_u) / (2.0 * half_u) * (width - 1)
        y = (v + half_v) / (2.0 * half_v) * (height - 1)
        sq_dist = (self._pixel_grid_x - x) ** 2 + (self._pixel_grid_y - y) ** 2
        gaussian_blob = np.exp(-0.5 * sq_dist / (self.gaussian_sigma**2))
        tactile_image += np.asarray(amplitude * gaussian_blob, dtype=np.float32)

    def _postprocess_images(self) -> None:
        """Clip and normalize tactile images according to current settings."""
        if (
            self._left_image is None
            or self._right_image is None
            or self._left_inside_image is None
            or self._left_outside_image is None
            or self._right_inside_image is None
            or self._right_outside_image is None
        ):
            raise RuntimeError("Sensor has not been reset yet.")

        np.clip(self._left_image, 0.0, self.clip_value, out=self._left_image)
        np.clip(self._right_image, 0.0, self.clip_value, out=self._right_image)
        np.clip(
            self._left_inside_image, 0.0, self.clip_value, out=self._left_inside_image
        )
        np.clip(
            self._left_outside_image, 0.0, self.clip_value, out=self._left_outside_image
        )
        np.clip(
            self._right_inside_image, 0.0, self.clip_value, out=self._right_inside_image
        )
        np.clip(
            self._right_outside_image,
            0.0,
            self.clip_value,
            out=self._right_outside_image,
        )

        if self.normalize:
            scale = self.clip_value if self.clip_value > 0 else 1.0
            self._left_image /= scale
            self._right_image /= scale
            self._left_inside_image /= scale
            self._left_outside_image /= scale
            self._right_inside_image /= scale
            self._right_outside_image /= scale
            np.clip(self._left_image, 0.0, 1.0, out=self._left_image)
            np.clip(self._right_image, 0.0, 1.0, out=self._right_image)
            np.clip(self._left_inside_image, 0.0, 1.0, out=self._left_inside_image)
            np.clip(self._left_outside_image, 0.0, 1.0, out=self._left_outside_image)
            np.clip(self._right_inside_image, 0.0, 1.0, out=self._right_inside_image)
            np.clip(self._right_outside_image, 0.0, 1.0, out=self._right_outside_image)

    def _get_half_extent_for_side(self, side: _Side) -> tuple[float, float]:
        """Return active pad half-extent for one finger side."""
        if side == "left":
            return self._left_pad_half_extent_effective
        return self._right_pad_half_extent_effective

    def _classify_inside_outside(
        self, local_point: npt.NDArray[np.float64], side: _Side
    ) -> Literal["inside", "outside"]:
        """Classify one local contact as inside or outside finger-side contact."""
        local_normal_value = float(local_point[self.tactile_normal_axis])
        side_sign = (
            float(self.inside_normal_sign_by_side[0])
            if side == "left"
            else float(self.inside_normal_sign_by_side[1])
        )
        return "inside" if (local_normal_value * side_sign) >= 0.0 else "outside"

    def _get_surface_dimensions_for_side(self, side: _Side) -> dict[str, float]:
        """Return surface dimensions in meters for one tactile pad."""
        half_u, half_v = self._get_half_extent_for_side(side)
        width_m = 2.0 * float(half_u)
        height_m = 2.0 * float(half_v)
        return {
            "width_m": width_m,
            "height_m": height_m,
            "area_m2": width_m * height_m,
        }

    def _resolve_effective_pad_half_extents(
        self, model: mujoco.MjModel
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Resolve per-side tactile extents, optionally from geom sizes."""
        configured = self.pad_half_extent
        if not self.auto_pad_half_extent or not hasattr(model, "geom_size"):
            return configured, configured

        left_derived = self._derive_half_extent_from_geoms(model, self._left_geom_ids)
        right_derived = self._derive_half_extent_from_geoms(model, self._right_geom_ids)

        left_effective = (
            max(configured[0], left_derived[0]),
            max(configured[1], left_derived[1]),
        )
        right_effective = (
            max(configured[0], right_derived[0]),
            max(configured[1], right_derived[1]),
        )
        return left_effective, right_effective

    def _derive_half_extent_from_geoms(
        self, model: mujoco.MjModel, geom_ids: set[int]
    ) -> tuple[float, float]:
        """Derive tactile plane half-extent from MuJoCo geom half-sizes."""
        if not geom_ids:
            return self.pad_half_extent

        axis_u, axis_v = self.tactile_plane_axes
        max_u = 0.0
        max_v = 0.0
        for geom_id in geom_ids:
            geom_size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
            max_u = max(max_u, float(abs(geom_size[axis_u])))
            max_v = max(max_v, float(abs(geom_size[axis_v])))

        if max_u <= 0.0 or max_v <= 0.0:
            return self.pad_half_extent
        # Small margin helps avoid dropping border contacts due to numeric noise.
        return max_u * 1.05, max_v * 1.05
