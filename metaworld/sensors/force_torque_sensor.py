"""Force/torque sensor for MetaWorld environments.

This module implements a virtual 6-axis wrench sensor from MuJoCo contact data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import mujoco
import numpy as np
import numpy.typing as npt

from metaworld.sensors.base import SensorBase

if TYPE_CHECKING:
    from metaworld.sawyer_xyz_env import SawyerXYZEnv


class ForceTorqueSensor(SensorBase):
    """Virtual force/torque sensor from contact wrench aggregation.

    The sensor sums contact forces on a set of monitored geometries and reports
    a 6D wrench [Fx, Fy, Fz, Tx, Ty, Tz]. Torques are computed about the chosen
    `origin_site`.
    """

    def __init__(
        self,
        geom_names: tuple[str, ...] = ("leftpad_geom", "rightpad_geom"),
        origin_site: str = "endEffector",
        # If "sensor", the wrench is expressed in the local frame of the `origin_site`, which means
        # that if the site rotates, the reported force/torque axes rotate with it. If "world", the wrench is expressed in the world frame.
        output_frame: Literal["world", "sensor"] = "world",
        invert_sign: bool = False,
    ) -> None:
        """Initialize the force/torque sensor.

        Args:
            geom_names: Geometries to monitor for contacts.
            origin_site: Site used as the torque reference point.
            output_frame: Output frame. "world" or local "sensor" frame.
            invert_sign: If True, flip reported wrench sign.
        """
        self.geom_names = geom_names
        self.origin_site = origin_site
        self.output_frame = output_frame
        self.invert_sign = invert_sign

        self._geom_ids: set[int] = set()
        self._site_id: int | None = None
        self._wrench: npt.NDArray[np.float64] | None = None
        self._contact_wrench_tmp = np.zeros(6, dtype=np.float64)

    @property
    def name(self) -> str:
        """Return unique identifier for this sensor."""
        geom_key = "_".join(self.geom_names)
        return f"force_torque_{geom_key}_at_{self.origin_site}"

    def reset(self, env: MujocoEnv) -> None:
        """Reset sensor state and resolve MuJoCo IDs.

        Args:
            env: MetaWorld environment.

        Raises:
            RuntimeError: If monitored geoms or origin site are missing.
        """
        try:
            self._geom_ids = {
                env.unwrapped.model.geom(name).id for name in self.geom_names
            }
        except KeyError as exc:
            raise RuntimeError(
                f"One or more geometries not found: {self.geom_names}"
            ) from exc

        try:
            self._site_id = env.unwrapped.model.site(self.origin_site).id
        except KeyError as exc:
            raise RuntimeError(f"Site '{self.origin_site}' not found.") from exc

        self._wrench = np.zeros(6, dtype=np.float64)

    def update(self, env: MujocoEnv) -> None:
        """Update wrench from active contacts."""
        if self._wrench is None or self._site_id is None:
            raise RuntimeError(
                f"Sensor '{self.name}' update() called before reset(). "
                "Call env.reset() and sensor.reset(env) first."
            )

        model = env.unwrapped.model
        data = env.unwrapped.data

        self._wrench.fill(0.0)
        origin_ft_sensor = data.site_xpos[self._site_id].copy()

        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            geom1_in = geom1 in self._geom_ids
            geom2_in = geom2 in self._geom_ids
            if not geom1_in and not geom2_in:
                continue
            if geom1_in and geom2_in:
                # Internal contact inside the monitored set: skip.
                continue

            self._contact_wrench_tmp.fill(0.0)
            # Extracts contact force/torque in the contact frame. Force is first 3 elements, torque is last 3 elements.
            mujoco.mj_contactForce(model, data, i, self._contact_wrench_tmp)

            # Contact wrench is expressed in the contact frame (x axis is contact normal and y-z axis are tangent plane).
            rot_contact_to_world = np.asarray(contact.frame, dtype=np.float64).reshape(
                3, 3
            )
            # We rotate the contact wrench to world frame before accumulating, so that forces from different contacts can be summed correctly.
            # We will rotate back to sensor frame at the end if needed.
            force_world = rot_contact_to_world @ self._contact_wrench_tmp[:3]
            # Torque that is happening at contact point: torsional and rolling friction
            torque_world = rot_contact_to_world @ self._contact_wrench_tmp[3:]

            # `mj_contactForce` follows (geom1 -> geom2) orientation conventions.
            # We report force acting ON monitored geometry.
            sign = -1.0 if geom1_in else 1.0

            force_world *= sign
            torque_world *= sign

            sensor_to_contact_vector = (
                np.asarray(contact.pos, dtype=np.float64) - origin_ft_sensor
            )
            # np.cross(sensor_to_contact_vector, force_world) is literally r*F. This is the "real torque" part that comes from the contact force applied at a distance to the origin.
            torque_about_origin_world = torque_world + np.cross(
                sensor_to_contact_vector, force_world
            )

            self._wrench[:3] += force_world
            self._wrench[3:] += torque_about_origin_world

        if self.invert_sign:
            self._wrench *= -1.0

        # Simply convert to sensor frame if needed. This means the sensor axes rotate with the `origin_site`.
        if self.output_frame == "sensor":
            rot_sensor_to_world = data.site_xmat[self._site_id].reshape(3, 3)
            rot_world_to_sensor = rot_sensor_to_world.T
            self._wrench[:3] = rot_world_to_sensor @ self._wrench[:3]
            self._wrench[3:] = rot_world_to_sensor @ self._wrench[3:]

    def read(self) -> npt.NDArray[np.float64]:
        """Return current wrench [Fx, Fy, Fz, Tx, Ty, Tz]."""
        if self._wrench is None:
            raise RuntimeError(
                f"Sensor '{self.name}' read() called before reset(). "
                "Call env.reset() first."
            )
        return self._wrench.copy()

    def get_observation_space(self) -> spaces.Space:
        """Return observation space for 6D wrench output."""
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float64,
        )

    def get_metadata(self) -> dict[str, str | tuple[str, ...] | bool]:
        """Return metadata about this force/torque sensor."""
        return {
            "type": "force",
            "subtype": "contact_wrench",
            "geom_names": self.geom_names,
            "origin_site": self.origin_site,
            "output_frame": self.output_frame,
            "invert_sign": self.invert_sign,
            "units_force": "N",
            "units_torque": "N*m",
        }

    def validate(self, env: SawyerXYZEnv) -> bool:
        """Validate monitored geoms and origin site exist."""
        try:
            for name in self.geom_names:
                env.unwrapped.model.geom(name).id
            env.unwrapped.model.site(self.origin_site).id
            return True
        except KeyError:
            return False
