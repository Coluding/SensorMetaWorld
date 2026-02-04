"""Test script for DepthCameraSensor with reach-v3 environment.

This script tests the depth camera sensor mounted on the gripper.
Run this after implementing DepthCameraSensor.update().

Usage:
    python -m pytest tests/sensors/test_depth_camera.py -v
    # Or run directly:
    python tests/sensors/test_depth_camera.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pytest

from metaworld.sensors.visual import DepthCameraSensor


class TestDepthCameraSensor:
    """Test suite for DepthCameraSensor."""

    def test_sensor_creation(self):
        """Test that we can create a DepthCameraSensor."""
        sensor = DepthCameraSensor(
            camera_name="gripper_depth_cam",
            height=64,
            width=64,
            normalize=False,
        )

        assert sensor.name == "depth_camera_gripper_depth_cam"
        assert sensor.camera_name == "gripper_depth_cam"
        assert sensor.height == 64
        assert sensor.width == 64
        assert not sensor.normalize

    def test_observation_space(self):
        """Test that observation space is correctly defined."""
        sensor = DepthCameraSensor(
            camera_name="gripper_depth_cam", height=64, width=64, normalize=False
        )

        obs_space = sensor.get_observation_space()
        assert obs_space.shape == (64 * 64,)
        assert obs_space.dtype == np.float32

    def test_observation_space_normalized(self):
        """Test observation space with normalization enabled."""
        sensor = DepthCameraSensor(
            camera_name="gripper_depth_cam", height=64, width=64, normalize=True
        )

        obs_space = sensor.get_observation_space()
        assert obs_space.low[0] == 0.0
        assert obs_space.high[0] == 1.0

    def test_metadata(self):
        """Test that sensor metadata is populated."""
        sensor = DepthCameraSensor(camera_name="gripper_depth_cam", height=64, width=64)

        metadata = sensor.get_metadata()
        assert metadata["type"] == "visual"
        assert metadata["subtype"] == "depth"
        assert metadata["camera_name"] == "gripper_depth_cam"
        assert "64x64" in metadata["resolution"]

    def test_sensor_with_environment(self):
        """Test sensor with actual reach-v3 environment.

        This test is skipped until sensor integration is complete.
        """
        # Create environment
        env = gym.make("Meta-World/MT1", env_name="reach-v3")

        # Create sensor
        sensor = DepthCameraSensor(
            camera_name="gripper_depth_cam", height=64, width=64
        )

        # Reset environment and sensor
        env.reset()
        sensor.reset(env)

        # Validate camera exists
        assert sensor.validate(env)

        # Update sensor (this will call the rendering)
        sensor.update(env)

        # Read sensor data
        depth_data = sensor.read()
        assert depth_data.shape == (64 * 64,)
        assert depth_data.dtype == np.float64

        env.close()

    def test_read_before_reset_raises_error(self):
        """Test that reading before reset raises an error."""
        sensor = DepthCameraSensor(camera_name="gripper_depth_cam", height=64, width=64)

        with pytest.raises(RuntimeError, match="read\\(\\) called before reset\\(\\)"):
            sensor.read()


def visualize_depth_camera():
    """Interactive test: visualize depth camera output.

    Run this function directly to see the depth camera in action.
    Requires implementing DepthCameraSensor.update().
    """
    print("=" * 60)
    print("Depth Camera Visualization Test")
    print("=" * 60)

    # Create environment
    print("\n1. Creating reach-v3 environment...")
    env = gym.make("Meta-World/MT1", env_name="reach-v3")

    # Create depth sensor
    print("2. Creating depth camera sensor...")
    sensor = DepthCameraSensor(
        camera_name="gripper_depth_cam", height=128, width=128, normalize=False
    )

    print(f"   - Sensor name: {sensor.name}")
    print(f"   - Resolution: {sensor.width}x{sensor.height}")
    print(f"   - Observation space: {sensor.get_observation_space()}")

    # Reset environment
    print("\n3. Resetting environment...")
    obs, info = env.reset(seed=42)

    # Reset sensor
    print("4. Resetting sensor...")
    try:
        sensor.reset(env)
        print("   ✓ Sensor reset successful")
    except RuntimeError as e:
        print(f"   ✗ Sensor reset failed: {e}")
        print(
            "\n   Make sure the camera 'gripper_depth_cam' exists in sawyer_reach_v3.xml"
        )
        return

    # Run a few steps and visualize
    print("\n5. Running environment and capturing depth images...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(6):
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Update sensor
        try:
            sensor.update(env)
            depth_data = sensor.read()

            # Get colored depth image for visualization
            depth_rgb = sensor.depth_to_color_image()

            # Plot
            axes[i].imshow(depth_rgb)
            axes[i].set_title(f"Step {i + 1}\nReward: {reward:.3f}")
            axes[i].axis("off")

            # Print some stats
            depth_image = sensor.get_depth_as_image()
            print(
                f"   Step {i + 1}: depth range [{depth_image.min():.3f}, {depth_image.max():.3f}]"
            )

        except NotImplementedError:
            print(
                "\n   ✗ DepthCameraSensor.update() not implemented yet!"
            )
            print(
                "     Implement the update() method in metaworld/sensors/visual.py"
            )
            return
        except Exception as e:
            print(f"\n   ✗ Error during sensor update: {e}")
            import traceback

            traceback.print_exc()
            return

    plt.tight_layout()
    plt.savefig("depth_camera_test.png", dpi=150)
    print("\n6. Visualization saved to 'depth_camera_test.png'")
    plt.show()

    # Cleanup
    env.close()
    print("\n✓ Test complete!")


def test_observation_space_integration():
    """Test that sensor observation space can be integrated into env obs space.

    This demonstrates how the sensor's observation space will be combined
    with the environment's proprioceptive observation.
    """
    print("\n" + "=" * 60)
    print("Observation Space Integration Test")
    print("=" * 60)

    # Create environment
    env = gym.make("Meta-World/MT1", env_name="reach-v3")
    print(f"\n1. Original observation space: {env.observation_space}")
    print(f"   Shape: {env.observation_space.shape}")

    # Create sensor
    sensor = DepthCameraSensor(camera_name="gripper_depth_cam", height=64, width=64)
    sensor_space = sensor.get_observation_space()
    print(f"\n2. Sensor observation space: {sensor_space}")
    print(f"   Shape: {sensor_space.shape}")

    # Show what combined observation would look like
    from gymnasium import spaces

    combined_space = spaces.Dict(
        {
            "proprioceptive": env.observation_space,
            "sensors": spaces.Dict({"depth_camera_gripper_depth_cam": sensor_space}),
        }
    )

    print(f"\n3. Combined observation space (future):")
    print(f"   {combined_space}")

    env.close()


if __name__ == "__main__":
    print("MetaWorld Depth Camera Sensor Test\n")

    # Run basic unit tests
    print("Running unit tests...")
    test = TestDepthCameraSensor()

    try:
        test.test_sensor_creation()
        print("✓ test_sensor_creation passed")
    except Exception as e:
        print(f"✗ test_sensor_creation failed: {e}")

    try:
        test.test_observation_space()
        print("✓ test_observation_space passed")
    except Exception as e:
        print(f"✗ test_observation_space failed: {e}")

    try:
        test.test_metadata()
        print("✓ test_metadata passed")
    except Exception as e:
        print(f"✗ test_metadata failed: {e}")

    # Test observation space integration
    test_observation_space_integration()

    # Run visualization test (requires implementation)
    print("\n" + "=" * 60)
    response = input(
        "\nRun depth camera visualization? (requires update() implementation) [y/N]: "
    )
    if response.lower() == "y":
        visualize_depth_camera()
    else:
        print("\nSkipping visualization test.")
        print(
            "Run 'python tests/sensors/test_depth_camera.py' and type 'y' when ready."
        )