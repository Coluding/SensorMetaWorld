"""DIGIT/GelSight-style vision-based tactile sensor for MetaWorld."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gymnasium import spaces
import numpy as np
import numpy.typing as npt

from metaworld.sensors.base import SensorBase

try:
    import mujoco
except ImportError:  # pragma: no cover - optional dependency at runtime.
    mujoco = None

if TYPE_CHECKING:
    from metaworld.sawyer_xyz_env import SawyerXYZEnv


@dataclass(frozen=True)
class _FingerAttachment:
    """Resolved MuJoCo attachment information for one tactile finger."""

    side: str
    geom_ids: tuple[int, ...]
    geom_names: tuple[str, ...]
    anchor_geom_id: int
    anchor_site_id: int | None
    anchor_name: str
    half_extent_u: float
    half_extent_v: float


class TactileDigitSensor(SensorBase):
    """Approximate DIGIT/GelSight tactile sensor using MuJoCo contact data."""

    _DEFAULT_HALF_EXTENT_U = 0.012
    _DEFAULT_HALF_EXTENT_V = 0.018
    _MIN_HALF_EXTENT = 0.006
    _TACTILE_U_AXIS = 0
    _TACTILE_V_AXIS = 2
    _TACTILE_NORMAL_AXIS = 1

    def __init__(
        self,
        resolution: int = 64,
        sigma_px: float = 2.0,
        noise_std: float = 0.0,
        base_texture: bool = True,
        seed: int | None = None,
        normalize: bool = True,
    ) -> None:
        self.resolution = int(resolution)
        self.sigma_px = float(sigma_px)
        self.noise_std = float(noise_std)
        self.base_texture = bool(base_texture)
        self.seed = seed
        self.normalize = bool(normalize)

        if self.resolution <= 0:
            raise ValueError("resolution must be positive.")
        if self.sigma_px <= 0.0:
            raise ValueError("sigma_px must be positive.")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")

        self._rng = np.random.default_rng(seed)
        self._left_finger: _FingerAttachment | None = None
        self._right_finger: _FingerAttachment | None = None
        self._reading: npt.NDArray[np.float32] | None = None
        self._left_image: npt.NDArray[np.float32] | None = None
        self._right_image: npt.NDArray[np.float32] | None = None
        self._contact_force_tmp = np.zeros(6, dtype=np.float64)

        self._base_gel_color = np.array([0.33, 0.38, 0.44], dtype=np.float32)
        self._texture_template = self._build_texture_template()
        self._base_image = self._build_base_image()

    @property
    def name(self) -> str:
        """Return unique identifier for this sensor."""
        return "tactile_digit"

    def reset(self, env: SawyerXYZEnv) -> None:
        """Resolve finger attachments and initialize image buffers."""
        self.validate(env)
        self._left_finger = self._resolve_finger_attachment(env, "left")
        self._right_finger = self._resolve_finger_attachment(env, "right")
        self._left_image = self._base_image.copy()
        self._right_image = self._base_image.copy()
        self._reading = self._flatten_images(self._left_image, self._right_image)

    def update(self, env: SawyerXYZEnv) -> None:
        """Update tactile images from active MuJoCo contacts."""
        if self._left_finger is None or self._right_finger is None:
            raise RuntimeError(
                f"Sensor '{self.name}' update() called before reset(). "
                "Call env.reset() and sensor.reset(env) first."
            )

        model = env.unwrapped.model
        data = env.unwrapped.data

        left_frame = self._get_attachment_frame(data, self._left_finger)
        right_frame = self._get_attachment_frame(data, self._right_finger)

        left_pressure = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        right_pressure = np.zeros((self.resolution, self.resolution), dtype=np.float32)

        for contact_idx in range(int(data.ncon)):
            contact = data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            if geom1 in self._left_finger.geom_ids or geom2 in self._left_finger.geom_ids:
                self._accumulate_contact(
                    model=model,
                    data=data,
                    contact_index=contact_idx,
                    contact=contact,
                    attachment=self._left_finger,
                    frame=left_frame,
                    opposite_frame=right_frame,
                    opposite_geom_ids=self._right_finger.geom_ids,
                    pressure_map=left_pressure,
                )

            if geom1 in self._right_finger.geom_ids or geom2 in self._right_finger.geom_ids:
                self._accumulate_contact(
                    model=model,
                    data=data,
                    contact_index=contact_idx,
                    contact=contact,
                    attachment=self._right_finger,
                    frame=right_frame,
                    opposite_frame=left_frame,
                    opposite_geom_ids=self._left_finger.geom_ids,
                    pressure_map=right_pressure,
                )

        self._left_image = self._render_tactile_image(left_pressure)
        self._right_image = self._render_tactile_image(right_pressure)
        self._reading = self._flatten_images(self._left_image, self._right_image)

    def read(self) -> npt.NDArray[np.float32]:
        """Return the current concatenated tactile reading."""
        if self._reading is None:
            raise RuntimeError(
                f"Sensor '{self.name}' read() called before reset(). "
                "Call env.reset() first."
            )
        return self._reading

    def get_observation_space(self) -> spaces.Space:
        """Return observation space for two flattened RGB tactile images."""
        high = 1.0 if self.normalize else 255.0
        return spaces.Box(
            low=0.0,
            high=high,
            shape=(2 * self.resolution * self.resolution * 3,),
            dtype=np.float32,
        )

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata describing this tactile sensor."""
        return {
            "type": "tactile",
            "modality": "vision_tactile",
            "resolution": self.resolution,
            "channels": 3,
            "fingers": 2,
            "normalize": self.normalize,
            "sigma_px": self.sigma_px,
        }

    def validate(self, env: SawyerXYZEnv) -> bool:
        """Validate that the environment exposes the data needed for this sensor."""
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "model") or not hasattr(unwrapped, "data"):
            raise RuntimeError(
                "TactileDigitSensor requires env.unwrapped.model and env.unwrapped.data."
            )

        model = unwrapped.model
        geom_names = self._get_geom_names(model)

        missing: list[str] = []
        for side in ("left", "right"):
            names = self._match_geom_names(geom_names, side)
            if not names:
                missing.append(side)

        if missing:
            raise RuntimeError(
                "Could not identify tactile finger geometries for "
                f"{', '.join(missing)} finger(s). "
                "Searched for combinations of "
                f"{self._search_description()}. "
                f"Available geoms: {geom_names}"
            )

        return True

    def _build_texture_template(self) -> npt.NDArray[np.float32]:
        """Create a deterministic pseudo-texture shared by both finger pads."""
        if not self.base_texture:
            return np.zeros((self.resolution, self.resolution, 3), dtype=np.float32)

        texture = self._rng.normal(
            loc=0.0,
            scale=0.015,
            size=(self.resolution, self.resolution, 3),
        ).astype(np.float32)

        texture = (
            texture
            + np.roll(texture, 1, axis=0)
            + np.roll(texture, -1, axis=0)
            + np.roll(texture, 1, axis=1)
            + np.roll(texture, -1, axis=1)
        ) / np.float32(5.0)
        return texture

    def _build_base_image(self) -> npt.NDArray[np.float32]:
        """Create the nominal gel image before pressure is applied."""
        base = np.broadcast_to(
            self._base_gel_color.reshape(1, 1, 3),
            (self.resolution, self.resolution, 3),
        ).astype(np.float32)

        if self.base_texture:
            base = base + self._texture_template

        yy = np.linspace(-1.0, 1.0, self.resolution, dtype=np.float32).reshape(-1, 1)
        xx = np.linspace(-1.0, 1.0, self.resolution, dtype=np.float32).reshape(1, -1)
        vignette = 0.05 * (xx**2 + yy**2)
        base = base - vignette[..., None]
        return np.clip(base, 0.0, 1.0).astype(np.float32)

    def _resolve_finger_attachment(
        self, env: SawyerXYZEnv, side: str
    ) -> _FingerAttachment:
        """Resolve pad geom IDs and anchor the tactile frame on the pad geom itself."""
        model = env.unwrapped.model
        geom_names = self._get_geom_names(model)
        matched_geom_names = self._match_geom_names(geom_names, side)
        if not matched_geom_names:
            raise RuntimeError(
                f"Could not identify {side} finger tactile geometry. "
                f"Available geoms: {geom_names}"
            )

        geom_ids = tuple(int(model.geom(name).id) for name in matched_geom_names)
        anchor_geom_name = matched_geom_names[0]
        anchor_geom_id = int(model.geom(anchor_geom_name).id)

        half_extent_u, half_extent_v = self._estimate_pad_extents(model, anchor_geom_id)

        return _FingerAttachment(
            side=side,
            geom_ids=geom_ids,
            geom_names=tuple(matched_geom_names),
            anchor_geom_id=anchor_geom_id,
            anchor_site_id=None,
            anchor_name=anchor_geom_name,
            half_extent_u=half_extent_u,
            half_extent_v=half_extent_v,
        )

    def _accumulate_contact(
        self,
        *,
        model: Any,
        data: Any,
        contact_index: int,
        contact: Any,
        attachment: _FingerAttachment,
        frame: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        opposite_frame: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        opposite_geom_ids: tuple[int, ...],
        pressure_map: npt.NDArray[np.float32],
    ) -> None:
        """Project one contact into the local tactile plane and splat pressure."""
        if not self._is_internal_pad_contact(
            contact=contact,
            attachment=attachment,
            frame=frame,
            opposite_frame=opposite_frame,
            opposite_geom_ids=opposite_geom_ids,
        ):
            return

        pos_world, rot_world = frame
        contact_world = np.asarray(contact.pos, dtype=np.float64)
        local = rot_world.T @ (contact_world - pos_world)

        u = float(local[self._TACTILE_U_AXIS])
        v = float(local[self._TACTILE_V_AXIS])

        # Contacts far outside the nominal pad do not contribute to the tactile image.
        if (
            abs(u) > 1.5 * attachment.half_extent_u
            or abs(v) > 1.5 * attachment.half_extent_v
        ):
            return

        contact_weight = self._compute_contact_weight(model, data, contact_index, contact)
        if contact_weight <= 0.0:
            return

        px = self._coord_to_pixel(u, attachment.half_extent_u)
        py = self._coord_to_pixel(v, attachment.half_extent_v)
        self._add_gaussian_splat(pressure_map, px, py, contact_weight)

    def _is_internal_pad_contact(
        self,
        *,
        contact: Any,
        attachment: _FingerAttachment,
        frame: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        opposite_frame: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        opposite_geom_ids: tuple[int, ...],
    ) -> bool:
        """Keep only contacts that occur on the inward-facing finger pad side."""
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        other_geom = geom2 if geom1 in attachment.geom_ids else geom1

        # Ignore finger-finger or same-side self contacts. The tactile image should
        # represent object interaction on the internal pad, not gripper self-collision.
        if other_geom in attachment.geom_ids or other_geom in opposite_geom_ids:
            return False

        pos_world, rot_world = frame
        opposite_pos_world, _ = opposite_frame
        contact_world = np.asarray(contact.pos, dtype=np.float64)
        local = rot_world.T @ (contact_world - pos_world)
        opposite_local = rot_world.T @ (opposite_pos_world - pos_world)

        inward_sign = 1.0 if opposite_local[self._TACTILE_NORMAL_AXIS] >= 0.0 else -1.0
        inward_depth = float(local[self._TACTILE_NORMAL_AXIS] * inward_sign)

        # Only keep contacts that lie on the half-space facing the other finger.
        return inward_depth >= -0.0015

    def _compute_contact_weight(
        self, model: Any, data: Any, contact_index: int, contact: Any
    ) -> float:
        """Compute a bounded pressure proxy from penetration and contact force."""
        dist = float(getattr(contact, "dist", 0.0))
        penetration = max(0.0, -dist)
        penetration_term = 1.0 - np.exp(-penetration / 0.0008)

        force_term = 0.0
        if mujoco is not None:
            self._contact_force_tmp.fill(0.0)
            try:
                mujoco.mj_contactForce(model, data, contact_index, self._contact_force_tmp)
            except Exception:  # pragma: no cover - depends on runtime mujoco bindings.
                force_term = 0.0
            else:
                normal_force = abs(float(self._contact_force_tmp[0]))
                force_term = 1.0 - np.exp(-normal_force / 2.5)

        return float(
            np.clip(0.75 * penetration_term + 0.55 * force_term, 0.0, 1.35)
        )

    def _render_tactile_image(
        self, pressure_map: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Convert a pressure map into a lightweight RGB gel image."""
        pressure = np.clip(pressure_map, 0.0, 1.5).astype(np.float32)
        pressure_visual = np.clip(1.5 * pressure, 0.0, 2.0).astype(np.float32)

        grad_y, grad_x = np.gradient(pressure_visual)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)

        image = self._base_image.copy()
        image[..., 0] += 0.48 * pressure_visual + 0.14 * grad_mag
        image[..., 1] += 0.34 * pressure_visual + 0.16 * grad_mag
        image[..., 2] += 0.28 * pressure_visual + 0.22 * grad_mag

        # Directional highlights create a simple bump-like appearance.
        image[..., 0] += 0.18 * np.clip(-grad_x, 0.0, None)
        image[..., 1] += 0.05 * np.clip(-grad_y, 0.0, None)
        image[..., 2] += 0.14 * np.clip(grad_y, 0.0, None)

        if self.noise_std > 0.0:
            noise = self._rng.normal(
                loc=0.0,
                scale=self.noise_std,
                size=image.shape,
            ).astype(np.float32)
            image = image + noise

        image = np.clip(image, 0.0, 1.0).astype(np.float32)
        if not self.normalize:
            image = image * np.float32(255.0)
        return image.astype(np.float32)

    def _get_attachment_frame(
        self,
        data: Any,
        attachment: _FingerAttachment,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the world position and rotation for one finger tactile plane."""
        pos = np.asarray(data.geom_xpos[attachment.anchor_geom_id], dtype=np.float64)
        rot = np.asarray(
            data.geom_xmat[attachment.anchor_geom_id].reshape(3, 3),
            dtype=np.float64,
        )
        return pos, rot

    def _flatten_images(
        self,
        left_image: npt.NDArray[np.float32],
        right_image: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Flatten both RGB images in C-order and concatenate them."""
        return np.concatenate(
            (left_image.reshape(-1, order="C"), right_image.reshape(-1, order="C"))
        ).astype(np.float32)

    def _coord_to_pixel(self, value: float, half_extent: float) -> float:
        """Map local finger-plane coordinates to image pixel coordinates."""
        normalized = 0.5 + 0.5 * (value / max(half_extent, self._MIN_HALF_EXTENT))
        normalized = float(np.clip(normalized, 0.0, 1.0))
        return normalized * float(self.resolution - 1)

    def _add_gaussian_splat(
        self,
        pressure_map: npt.NDArray[np.float32],
        px: float,
        py: float,
        weight: float,
    ) -> None:
        """Add a truncated Gaussian splat to the pressure map."""
        radius = max(1, int(np.ceil(3.0 * self.sigma_px)))
        cx = int(round(px))
        cy = int(round(py))

        x0 = max(0, cx - radius)
        x1 = min(self.resolution, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(self.resolution, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        dx2 = (xs - np.float32(px)) ** 2
        dy2 = (ys - np.float32(py)) ** 2
        patch = np.exp(
            -(dy2[:, None] + dx2[None, :]) / np.float32(2.0 * self.sigma_px**2)
        ).astype(np.float32)
        pressure_map[y0:y1, x0:x1] += np.float32(weight) * patch

    def _estimate_pad_extents(self, model: Any, geom_id: int) -> tuple[float, float]:
        """Estimate x-z tactile-plane half-extents from MuJoCo geom size."""
        try:
            size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        except Exception:
            return self._DEFAULT_HALF_EXTENT_U, self._DEFAULT_HALF_EXTENT_V

        u_extent = (
            float(size[self._TACTILE_U_AXIS])
            if size.size > self._TACTILE_U_AXIS
            else self._DEFAULT_HALF_EXTENT_U
        )
        v_extent = (
            float(size[self._TACTILE_V_AXIS])
            if size.size > self._TACTILE_V_AXIS
            else 0.0
        )

        if u_extent <= 0.0:
            u_extent = self._DEFAULT_HALF_EXTENT_U
        if v_extent <= 0.0:
            v_extent = max(u_extent, self._DEFAULT_HALF_EXTENT_V)

        return (
            max(u_extent, self._MIN_HALF_EXTENT),
            max(v_extent, self._MIN_HALF_EXTENT),
        )

    def _match_geom_names(self, geom_names: list[str], side: str) -> list[str]:
        """Return finger geom names ordered from best to fallback match."""
        primary: list[tuple[int, str]] = []
        secondary: list[tuple[int, str]] = []
        tertiary: list[tuple[int, str]] = []

        side_tokens = self._side_tokens(side)
        finger_tokens = ("finger", "gripper")
        contact_tokens = ("pad", "tip")
        gripper_part_tokens = finger_tokens + contact_tokens + ("claw",)

        for name in geom_names:
            tokens = self._tokenize_name(name)
            if not self._matches_any(tokens, side_tokens):
                continue

            score = self._name_score(tokens, side)
            has_contact = self._matches_any(tokens, contact_tokens)
            has_finger = self._matches_any(tokens, finger_tokens)
            has_gripper_part = self._matches_any(tokens, gripper_part_tokens)

            # Best case: explicit tactile-contact geometry such as leftpad_geom.
            if has_contact:
                primary.append((score, name))
            # Fallback: finger or gripper link names still indicate the correct side.
            elif has_finger:
                secondary.append((score, name))
            # Last-resort fallback for names like rightclaw_it / leftclaw_it.
            elif has_gripper_part:
                tertiary.append((score, name))

        ordered = [name for _, name in sorted(primary, reverse=True)]
        if ordered:
            return ordered
        ordered = [name for _, name in sorted(secondary, reverse=True)]
        if ordered:
            return ordered
        return [name for _, name in sorted(tertiary, reverse=True)]

    def _match_site_names(self, site_names: list[str], side: str) -> list[str]:
        """Return plausible fingertip sites for the requested finger side."""
        matches: list[tuple[int, str]] = []
        side_tokens = self._side_tokens(side)
        finger_tokens = ("finger", "gripper")
        contact_tokens = ("pad", "tip", "end", "distal")

        for name in site_names:
            tokens = self._tokenize_name(name)
            if not self._matches_any(tokens, side_tokens):
                continue
            if not self._matches_any(tokens, finger_tokens + contact_tokens):
                continue
            score = self._name_score(tokens, side) + 2 * int(
                self._matches_any(tokens, contact_tokens)
            )
            matches.append((score, name))

        return [name for _, name in sorted(matches, reverse=True)]

    def _name_score(self, tokens: set[str], side: str) -> int:
        """Score candidate names so fingertip pads rank above generic links."""
        score = 0
        if self._matches_any(tokens, self._side_tokens(side)):
            score += 4
        if self._matches_any(tokens, ("finger", "gripper")):
            score += 3
        if self._matches_any(tokens, ("pad", "tip")):
            score += 6
        if "claw" in tokens:
            score += 2
        if "geom" in tokens or "site" in tokens:
            score += 1
        return score

    def _search_description(self) -> str:
        """Describe the geom matching patterns used during validation."""
        return (
            "('left'|'l') + ('finger'|'gripper') + ('pad'|'tip'), "
            "('right'|'r') + ('finger'|'gripper') + ('pad'|'tip'), "
            "with fallback to side + finger/gripper matches"
        )

    def _get_geom_names(self, model: Any) -> list[str]:
        """Return all MuJoCo geom names that are present in the model."""
        return [
            str(model.geom(i).name)
            for i in range(int(model.ngeom))
            if getattr(model.geom(i), "name", None)
        ]

    def _get_site_names(self, model: Any) -> list[str]:
        """Return all MuJoCo site names that are present in the model."""
        return [
            str(model.site(i).name)
            for i in range(int(model.nsite))
            if getattr(model.site(i), "name", None)
        ]

    def _side_tokens(self, side: str) -> tuple[str, ...]:
        """Return side-specific tokens used in fuzzy matching."""
        return ("left", "l") if side == "left" else ("right", "r")

    def _tokenize_name(self, name: str) -> set[str]:
        """Split MuJoCo names into lowercase alphanumeric tokens."""
        lowered = name.lower()
        cleaned = []
        for char in lowered:
            cleaned.append(char if char.isalnum() else " ")
        tokens = set("".join(cleaned).split())
        for keyword in (
            "left",
            "right",
            "finger",
            "gripper",
            "pad",
            "tip",
            "claw",
            "site",
            "geom",
            "end",
            "distal",
        ):
            if keyword in lowered:
                tokens.add(keyword)
        if "left" in tokens:
            tokens.add("l")
        if "right" in tokens:
            tokens.add("r")
        tokens.add(lowered)
        return tokens

    def _matches_any(self, tokens: set[str], candidates: tuple[str, ...]) -> bool:
        """Return True when any candidate token is present or embedded."""
        for candidate in candidates:
            if candidate in tokens:
                return True
            if len(candidate) > 1 and any(candidate in token for token in tokens):
                return True
        return False
