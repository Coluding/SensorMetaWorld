import numpy as np
import scipy.stats
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple


class DirectionPolicy(ABC):
    """Samples movement direction in 3D Spherical Coordinates."""

    @abstractmethod
    def sample_angles(
        self, rng: np.random.Generator, position: np.ndarray
    ) -> Tuple[float, float]:
        """Returns (azimuth, polar) in radians.
        azimuth (phi): [0, 2pi] (Rotation around Z axis)
        polar (theta): [0, pi]  (Angle from +Z axis)
        """
        raise NotImplementedError


class StepLengthPolicy(ABC):
    """Samples movement distance for each step."""

    @abstractmethod
    def sample_length(self, rng: np.random.Generator) -> float:
        raise NotImplementedError


class UniformDirection(DirectionPolicy):
    def sample_angles(
        self,
        rng: np.random.Generator,
        position: np.ndarray,
    ) -> Tuple[float, float]:
        del position
        # Azimuth is uniform [0, 2pi]
        azimuth = rng.uniform(0, 2 * np.pi)

        # Polar is NOT uniform in angle (results in bunching at poles).
        # To get uniform distribution over a sphere, we sample cos(theta) uniformly.
        # However, for simple random walks, uniform angle is often acceptable.
        # We will stick to uniform angle for simplicity unless strict sphere uniformity is needed.
        polar = rng.uniform(0, np.pi)

        return azimuth, polar


class GaussianMeanDirection(DirectionPolicy):
    """3D Random walk with Gaussian perturbations on both Azimuth and Polar angles."""

    def __init__(
        self,
        stddev: float = 0.1,
    ) -> None:
        if stddev <= 0:
            raise ValueError("stddev must be > 0")

        self.stddev = stddev
        self.prev_azimuth: Optional[float] = None
        self.prev_polar: Optional[float] = None
        self.step_idx = 0

    def sample_angles(
        self,
        rng: np.random.Generator,
        position: np.ndarray,
    ) -> Tuple[float, float]:

        if self.step_idx == 0 or self.prev_azimuth is None:
            self.prev_azimuth = rng.uniform(0, 2 * np.pi)
            self.prev_polar = rng.uniform(0, np.pi)
            return self.prev_azimuth, self.prev_polar

        # Perturb Azimuth (XY plane rotation)
        d_azi = rng.normal(loc=0.0, scale=self.stddev)
        self.prev_azimuth = (self.prev_azimuth + d_azi) % (2.0 * np.pi)

        # Perturb Polar (Up/Down)
        # We clip Polar to [0, pi] to prevent flipping upside down through the pole
        d_pol = rng.normal(loc=0.0, scale=self.stddev)
        self.prev_polar = np.clip(self.prev_polar + d_pol, 0.01, np.pi - 0.01)

        self.step_idx += 1
        return self.prev_azimuth, self.prev_polar


class GaussianMeanDirectionGravityPull(DirectionPolicy):
    """3D Random walk with a soft pull toward a reference position."""

    def __init__(self, stddev: float = 0.5, gravity_strength: float = 0.05) -> None:
        if stddev <= 0:
            raise ValueError("stddev must be > 0")
        if gravity_strength < 0:
            raise ValueError("gravity_strength must be >= 0")
        self.stddev = stddev
        self.gravity_strength = gravity_strength

        self.curr_azimuth: Optional[float] = None
        self.curr_polar: Optional[float] = None
        self.reference_position: Optional[np.ndarray] = None

    def set_reference_position(self, position: np.ndarray) -> None:
        self.reference_position = np.array(position, dtype=np.float32, copy=True)

    def sample_angles(
        self,
        rng: np.random.Generator,
        position: np.ndarray,
    ) -> Tuple[float, float]:
        if self.reference_position is None:
            self.set_reference_position(position)

        # Initialize
        if self.curr_azimuth is None:
            self.curr_azimuth = rng.uniform(0, 2 * np.pi)
            self.curr_polar = rng.uniform(0, np.pi)
            return self.curr_azimuth, self.curr_polar

        # Apply Gaussian Noise (Momentum)
        cand_azimuth = self.curr_azimuth + rng.normal(0, self.stddev)
        cand_polar = self.curr_polar + rng.normal(0, self.stddev)

        # Calculate angles toward reference position from current position.
        vec_to_reference = self.reference_position - position
        x, y, z = vec_to_reference[0], vec_to_reference[1], vec_to_reference[2]
        r = np.linalg.norm(vec_to_reference)

        if r < 1e-6:
            # At reference position, no gravity needed
            target_azimuth = cand_azimuth
            target_polar = cand_polar
        else:
            # Azimuth / polar toward reference.
            target_azimuth = np.arctan2(y, x)
            target_polar = np.arccos(z / r)

        # Apply Gravity Pull (Azimuth)
        diff_azi = target_azimuth - cand_azimuth
        diff_azi = (diff_azi + np.pi) % (2 * np.pi) - np.pi
        final_azimuth = cand_azimuth + (diff_azi * self.gravity_strength)

        # Apply Gravity Pull (Polar)
        diff_pol = target_polar - cand_polar
        # No modular arithmetic needed for polar, just simple difference
        final_polar = cand_polar + (diff_pol * self.gravity_strength)

        # Update State (Clip Polar, Wrap Azimuth)
        self.curr_azimuth = final_azimuth % (2.0 * np.pi)
        self.curr_polar = np.clip(final_polar, 0.01, np.pi - 0.01)

        return self.curr_azimuth, self.curr_polar


class ConstantStepLength(StepLengthPolicy):
    def __init__(self, step_size: float = 1.0) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        self.step_size = step_size

    def sample_length(
        self,
        rng: np.random.Generator,
    ) -> float:
        del rng
        return self.step_size


class LevyStepLength(StepLengthPolicy):
    """Pareto-based heavy-tailed step lengths.

    length = min_step * (1 + Pareto(alpha))
    """

    def __init__(
        self, alpha: float = 1.5, min_step: float = 0.3, max_step: float = 20.0
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if min_step <= 0:
            raise ValueError("min_step must be > 0")
        if max_step <= min_step:
            raise ValueError("max_step must be > min_step")
        self.alpha = alpha
        self.min_step = min_step
        self.max_step = max_step

    def sample_length(
        self,
        rng: np.random.Generator,
    ) -> float:
        length = self.min_step * (1.0 + rng.pareto(self.alpha))
        return float(min(length, self.max_step))


class MetaWorldPolicy(ABC):
    @abstractmethod
    def get_action(self, obs):
        pass


class RandomWalk(MetaWorldPolicy):
    """
    A composite policy that combines a DirectionPolicy and a StepLengthPolicy
    to generate 3D actions for Meta-World.

    It converts the spherical coordinates (phi, theta, r) from the sub-policies
    into Cartesian (dx, dy, dz) actions.
    """

    # Registry to map string names to classes
    DIRECTION_POLICIES = {
        "uniform": UniformDirection,
        "gaussian": GaussianMeanDirection,
        "gravity": GaussianMeanDirectionGravityPull,
    }

    STEP_LENGTH_POLICIES = {
        "constant": ConstantStepLength,
        "levy": LevyStepLength,
    }

    def __init__(
        self,
        direction_policy: str = "gravity",
        step_length_policy: str = "levy",
        direction_kwargs: dict = None,
        step_length_kwargs: dict = None,
        seed: int = None,
    ):
        self.rng = np.random.default_rng(seed)

        # Instantiate Direction Policy
        if direction_policy not in self.DIRECTION_POLICIES:
            raise ValueError(
                f"Unknown direction policy: {direction_policy}. Available: {list(self.DIRECTION_POLICIES.keys())}"
            )

        dir_cls = self.DIRECTION_POLICIES[direction_policy]
        self.direction_policy = dir_cls(**(direction_kwargs or {}))

        # Instantiate Step Length Policy
        if step_length_policy not in self.STEP_LENGTH_POLICIES:
            raise ValueError(
                f"Unknown step length policy: {step_length_policy}. Available: {list(self.STEP_LENGTH_POLICIES.keys())}"
            )

        len_cls = self.STEP_LENGTH_POLICIES[step_length_policy]
        self.step_length_policy = len_cls(**(step_length_kwargs or {}))

    def set_reference_position(self, position: np.ndarray) -> None:
        if hasattr(self.direction_policy, "set_reference_position"):
            self.direction_policy.set_reference_position(position)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Generates a 4D action [dx, dy, dz, gripper] based on current observation.
        """
        # Extract Position from Observation
        # In Meta-World, obs[0:3] is the end-effector position (x, y, z)
        current_pos = obs[0:3]

        # Get Spherical Coordinates
        # Phi (Azimuth) and Theta (Polar)
        phi, theta = self.direction_policy.sample_angles(
            rng=self.rng, position=current_pos
        )

        # Radius (Step Length)
        r = self.step_length_policy.sample_length(rng=self.rng)

        # Convert Spherical to Cartesian (Physics Convention)
        # x = r * sin(theta) * cos(phi)
        # y = r * sin(theta) * sin(phi)
        # z = r * cos(theta)
        sin_theta = np.sin(theta)

        dx = r * sin_theta * np.cos(phi)
        dy = r * sin_theta * np.sin(phi)
        dz = r * np.cos(theta)

        # Construct 4D Action
        # [dx, dy, dz, gripper_control]
        gripper = np.random.uniform(-1.0, 1.0)
        return np.array([dx, dy, dz, gripper], dtype=np.float32)


class NoisyExpertPolicy(MetaWorldPolicy):
    def __init__(self, expert_policy, noise_scale=0.1):
        self.expert_policy = expert_policy
        # Standard deviation of the noise
        self.noise_scale = noise_scale

    def get_action(self, obs):
        # Get the perfect expert action
        expert_action = self.expert_policy.get_action(obs)

        # Generate Gaussian noise
        # We are also adding noise to the gripper (last dim)
        noise = np.random.normal(0, self.noise_scale, size=expert_action.shape)

        return expert_action + noise
