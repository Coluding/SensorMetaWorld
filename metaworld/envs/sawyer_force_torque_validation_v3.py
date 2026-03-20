from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict


class SawyerForceTorqueValidationEnvV3(SawyerXYZEnv):
    """Minimal env for force/torque sensor validation with a welded payload."""

    PAYLOAD_MASS = 10.0
    PAYLOAD_OFFSET = np.array([0.10, 0.0, 0.0], dtype=np.float64)

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        hand_low = (-0.1, 0.55, 0.15)
        hand_high = (0.1, 0.75, 0.30)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )
        self.reward_function_version = reward_function_version
        self.init_config: InitConfigDict = {
            "obj_init_angle": 0.0,
            "obj_init_pos": np.array([0.14, 0.65, 0.20], dtype=np.float64),
            "hand_init_pos": np.array([0.0, 0.65, 0.20], dtype=np.float64),
        }
        self.goal = self.init_config["obj_init_pos"].copy()
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"].copy()
        self.hand_init_pos = self.init_config["hand_init_pos"].copy()

        reset_state = np.hstack((self.obj_init_pos, self.goal))
        self._random_reset_space = Box(reset_state, reset_state, dtype=np.float64)
        self.goal_space = Box(self.goal.copy(), self.goal.copy(), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_force_torque_validation_v3.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        del action
        return 0.0, {
            "success": 0.0,
            "near_object": 0.0,
            "grasp_success": 1.0,
            "grasp_reward": 0.0,
            "in_place_reward": 0.0,
            "obj_to_target": 0.0,
            "unscaled_reward": 0.0,
        }

    def _get_id_main_object(self) -> int:
        return self.model.geom("payload_geom").id

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("payload")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return Rotation.from_matrix(
            self.data.body("payload").xmat.reshape(3, 3)
        ).as_quat()

    def _set_payload_pose(self, pos: npt.NDArray[np.float64]) -> None:
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qpos[12:16] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        qvel[9:15] = 0.0
        self.set_state(qpos, qvel)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.init_tcp = self.tcp_center.copy()
        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")
        self.obj_init_pos = self.get_body_com("payload").copy()
        self.model.site("goal").pos = self._target_pos
        return self._get_obs()
