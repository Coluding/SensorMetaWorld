"""Kinematic future-state prediction sensor.

This module provides a single sensor that predicts future collision-free robot
states from a sequence of end-effector displacement actions using Jacobian IK.

Pipeline:
    action deltas -> target EE positions -> IK -> predicted future states

No dynamics stepping is used. The solver only calls MuJoCo forward kinematics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from metaworld.sensors.base import SensorBase

if TYPE_CHECKING:
    from metaworld.sawyer_xyz_env import SawyerXYZEnv


def _quat_mul(q1: npt.NDArray[np.float64], q2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Hamilton quaternion product for [w, x, y, z] quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_conj(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_error(
    target_quat: npt.NDArray[np.float64],
    current_quat: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return small-angle 3D orientation error between quaternions."""
    q_err = _quat_mul(target_quat, _quat_conj(current_quat))
    if q_err[0] < 0.0:
        q_err = -q_err
    return 2.0 * q_err[1:]


class FutureStateIKSensor(SensorBase):
    """Predict future states from end-effector displacement actions via IK.

    The sensor accepts a sequence of future end-effector displacement actions
    (shape ``(horizon, 3)`` or ``(horizon, 4)`` where only xyz are used), then
    predicts a trajectory by repeatedly:

    1. Converting delta action -> target EE position
    2. Solving IK in joint space (no physics stepping)
    3. Using solved state as warm start for the next step

    Returned sensor vector (flattened):
        ``[arm_joint_angles, achieved_ee_positions, converged_flags]``

    where each component is stacked over the prediction horizon.
    """

    def __init__(
        self,
        prediction_horizon: int = 10,
        ee_action_scale: float = 1.0,
        clip_to_workspace: bool = True,
        max_ik_iterations: int = 120,
        ik_tolerance: float = 1e-4,
        ik_damping: float = 1e-3,
        ik_max_delta: float = 0.05,
        ee_body_name: str = "hand",
    ):
        """Initialize future-state IK sensor.

        Args:
            prediction_horizon: Number of future actions/states to predict.
            ee_action_scale: Multiplier applied to incoming xyz displacement actions.
            clip_to_workspace: If True, clip target EE positions to env hand bounds.
            max_ik_iterations: Maximum iterations per IK solve.
            ik_tolerance: Position+orientation error norm threshold.
            ik_damping: Damped least-squares regularization.
            ik_max_delta: Per-iteration joint delta clip (rad).
            ee_body_name: MuJoCo body name for end effector.
        """
        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be > 0")

        self.prediction_horizon = prediction_horizon
        self.ee_action_scale = ee_action_scale
        self.clip_to_workspace = clip_to_workspace
        self.max_ik_iterations = max_ik_iterations
        self.ik_tolerance = ik_tolerance
        self.ik_damping = ik_damping
        self.ik_max_delta = ik_max_delta
        self.ee_body_name = ee_body_name

        self._arm_joint_ids: list[int] = []
        self._arm_qpos_ids: list[int] = []
        self._arm_dof_ids: list[int] = []
        self._ee_body_id: int | None = None

        self._hand_low: npt.NDArray[np.float64] | None = None
        self._hand_high: npt.NDArray[np.float64] | None = None

        self._action_sequence = np.zeros((prediction_horizon, 3), dtype=np.float64)
        self._prediction = np.zeros((prediction_horizon * (7 + 3 + 1),), dtype=np.float64)

        self._last_prediction: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.bool_]] = {
            "target_ee_positions": np.zeros((prediction_horizon, 3), dtype=np.float64),
            "achieved_ee_positions": np.zeros((prediction_horizon, 3), dtype=np.float64),
            "arm_joint_angles": np.zeros((prediction_horizon, 7), dtype=np.float64),
            "qpos_trajectory": np.zeros((prediction_horizon, 0), dtype=np.float64),
            "converged": np.zeros((prediction_horizon,), dtype=np.bool_),
        }

    @property
    def name(self) -> str:
        return f"future_state_ik_h{self.prediction_horizon}"

    def reset(self, env: SawyerXYZEnv) -> None:
        """Reset sensor state and resolve model IDs."""
        mw_env = env.unwrapped
        model = mw_env.model

        arm_joint_ids: list[int] = []
        for jid in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if joint_name and joint_name.startswith("right_j"):
                arm_joint_ids.append(jid)

        arm_joint_ids.sort(key=lambda j: str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)))
        if len(arm_joint_ids) != 7:
            raise RuntimeError(f"Expected 7 Sawyer arm joints, found {len(arm_joint_ids)}")

        ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)
        if ee_body_id < 0:
            raise RuntimeError(f"End-effector body '{self.ee_body_name}' not found")

        self._arm_joint_ids = arm_joint_ids
        self._arm_qpos_ids = [int(model.jnt_qposadr[j]) for j in arm_joint_ids]
        self._arm_dof_ids = [int(model.jnt_dofadr[j]) for j in arm_joint_ids]
        self._ee_body_id = ee_body_id

        hand_low = getattr(mw_env, "hand_low", None)
        hand_high = getattr(mw_env, "hand_high", None)
        if hand_low is not None and hand_high is not None:
            self._hand_low = np.asarray(hand_low, dtype=np.float64).copy()
            self._hand_high = np.asarray(hand_high, dtype=np.float64).copy()

        self._action_sequence.fill(0.0)

        nq = int(model.nq)
        self._last_prediction = {
            "target_ee_positions": np.zeros((self.prediction_horizon, 3), dtype=np.float64),
            "achieved_ee_positions": np.zeros((self.prediction_horizon, 3), dtype=np.float64),
            "arm_joint_angles": np.zeros((self.prediction_horizon, 7), dtype=np.float64),
            "qpos_trajectory": np.zeros((self.prediction_horizon, nq), dtype=np.float64),
            "converged": np.zeros((self.prediction_horizon,), dtype=np.bool_),
        }

    def set_action_sequence(self, actions: npt.NDArray[np.float32] | npt.NDArray[np.float64]) -> None:
        """Set future EE displacement actions used by ``update()``.

        Accepted shapes:
            - ``(horizon, 3)``: xyz displacement actions
            - ``(horizon, 4)``: MetaWorld-style action, xyz used and gripper ignored
        """
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape == (self.prediction_horizon, 3):
            self._action_sequence = actions.copy()
            return

        if actions.shape == (self.prediction_horizon, 4):
            self._action_sequence = actions[:, :3].copy()
            return

        raise ValueError(
            f"Expected actions shape ({self.prediction_horizon}, 3) or "
            f"({self.prediction_horizon}, 4), got {actions.shape}"
        )

    def update(self, env: SawyerXYZEnv) -> None:
        """Predict trajectory from currently stored action sequence."""
        if self._ee_body_id is None:
            raise RuntimeError("Sensor not initialized. Call reset() first.")

        mw_env = env.unwrapped
        model, data = mw_env.model, mw_env.data

        current_qpos = data.qpos.copy()
        current_qvel = data.qvel.copy()

        mujoco.mj_forward(model, data)
        current_ee_pos = data.xpos[self._ee_body_id].copy()
        target_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(target_quat, data.xmat[self._ee_body_id])

        target_ee_positions = np.zeros((self.prediction_horizon, 3), dtype=np.float64)
        achieved_ee_positions = np.zeros((self.prediction_horizon, 3), dtype=np.float64)
        arm_joint_angles = np.zeros((self.prediction_horizon, 7), dtype=np.float64)
        qpos_trajectory = np.zeros((self.prediction_horizon, model.nq), dtype=np.float64)
        converged = np.zeros((self.prediction_horizon,), dtype=np.bool_)

        for t in range(self.prediction_horizon):
            target_pos = current_ee_pos + self._action_sequence[t] * self.ee_action_scale
            if self.clip_to_workspace and self._hand_low is not None and self._hand_high is not None:
                target_pos = np.clip(target_pos, self._hand_low, self._hand_high)

            solved_qpos, ok = self._solve_pose_ik(
                model=model,
                seed_qpos=current_qpos,
                seed_qvel=current_qvel,
                target_pos=target_pos,
                target_quat=target_quat,
            )

            verify = mujoco.MjData(model)
            verify.qpos[:] = solved_qpos
            verify.qvel[:] = current_qvel
            mujoco.mj_forward(model, verify)
            reached_pos = verify.xpos[self._ee_body_id].copy()

            target_ee_positions[t] = target_pos
            achieved_ee_positions[t] = reached_pos
            qpos_trajectory[t] = solved_qpos
            arm_joint_angles[t] = solved_qpos[self._arm_qpos_ids]
            converged[t] = ok

            current_qpos = solved_qpos
            current_ee_pos = reached_pos

        self._last_prediction = {
            "target_ee_positions": target_ee_positions,
            "achieved_ee_positions": achieved_ee_positions,
            "arm_joint_angles": arm_joint_angles,
            "qpos_trajectory": qpos_trajectory,
            "converged": converged,
        }

        self._prediction = np.concatenate(
            [
                arm_joint_angles.reshape(-1),
                achieved_ee_positions.reshape(-1),
                converged.astype(np.float64),
            ]
        )

    def read(self) -> npt.NDArray[np.float64]:
        return self._prediction.copy()

    def get_observation_space(self) -> spaces.Space:
        total = self.prediction_horizon * 7 + self.prediction_horizon * 3 + self.prediction_horizon
        return spaces.Box(low=-np.inf, high=np.inf, shape=(total,), dtype=np.float64)

    def get_last_prediction(
        self,
    ) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.bool_]]:
        """Return structured prediction arrays from the most recent update."""
        return {
            "target_ee_positions": self._last_prediction["target_ee_positions"].copy(),
            "achieved_ee_positions": self._last_prediction["achieved_ee_positions"].copy(),
            "arm_joint_angles": self._last_prediction["arm_joint_angles"].copy(),
            "qpos_trajectory": self._last_prediction["qpos_trajectory"].copy(),
            "converged": self._last_prediction["converged"].copy(),
        }

    def get_metadata(self) -> dict[str, str | int | float | bool]:
        return {
            "type": "kinematic",
            "subtype": "future_state_ik",
            "prediction_horizon": self.prediction_horizon,
            "ee_action_scale": self.ee_action_scale,
            "clip_to_workspace": self.clip_to_workspace,
            "max_ik_iterations": self.max_ik_iterations,
            "ik_tolerance": self.ik_tolerance,
            "ik_damping": self.ik_damping,
            "ik_max_delta": self.ik_max_delta,
            "output_layout": "[arm_joint_angles, achieved_ee_positions, converged_flags]",
            "description": "Future-state prediction via EE displacement actions and collision-free IK",
        }

    def _solve_pose_ik(
        self,
        model: mujoco.MjModel,
        seed_qpos: npt.NDArray[np.float64],
        seed_qvel: npt.NDArray[np.float64],
        target_pos: npt.NDArray[np.float64],
        target_quat: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], bool]:
        """Damped least-squares pose IK using only forward kinematics."""
        if self._ee_body_id is None:
            raise RuntimeError("Sensor IK called before reset().")

        work = mujoco.MjData(model)
        work.qpos[:] = seed_qpos
        work.qvel[:] = seed_qvel

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)

        converged = False
        for _ in range(self.max_ik_iterations):
            mujoco.mj_forward(model, work)

            current_pos = work.xpos[self._ee_body_id].copy()
            current_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(current_quat, work.xmat[self._ee_body_id])

            err = np.concatenate([target_pos - current_pos, _quat_error(target_quat, current_quat)])
            if np.linalg.norm(err) < self.ik_tolerance:
                converged = True
                break

            mujoco.mj_jacBodyCom(model, work, jacp, jacr, self._ee_body_id)
            J = np.vstack([jacp, jacr])[:, self._arm_dof_ids]

            A = J @ J.T + (self.ik_damping**2) * np.eye(6)
            dq = J.T @ np.linalg.solve(A, err)
            dq = np.clip(dq, -self.ik_max_delta, self.ik_max_delta)

            work.qpos[self._arm_qpos_ids] += dq

            for jid, qid in zip(self._arm_joint_ids, self._arm_qpos_ids):
                if model.jnt_limited[jid]:
                    lo, hi = model.jnt_range[jid]
                    work.qpos[qid] = np.clip(work.qpos[qid], lo, hi)

        return work.qpos.copy(), converged

    def validate(self, env: SawyerXYZEnv) -> bool:
        """Return True when the environment has Sawyer arm joints and EE body."""
        try:
            mw_env = env.unwrapped
            model = mw_env.model

            arm_count = 0
            for jid in range(model.njnt):
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                if joint_name and joint_name.startswith("right_j"):
                    arm_count += 1
            if arm_count != 7:
                return False

            ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)
            return ee_body_id >= 0
        except Exception:
            return False


__all__ = ["FutureStateIKSensor"]
