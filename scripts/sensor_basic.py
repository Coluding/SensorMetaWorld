#!/usr/bin/env python3
"""Quick standalone test for the depth camera sensor.

This script can be run directly without pytest to quickly test the sensor.

Usage:
    python sensor_basic.py
"""

import sys

import gymnasium as gym
import numpy as np
from metaworld.sensors.visual import DepthCameraSensor


def test_basic_sensor_creation():
    """Test 1: Can we create a sensor?"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Sensor Creation")
    print("=" * 60)

    try:
        sensor = DepthCameraSensor(
            camera_name="gripper_depth_cam", height=64, width=64, normalize=False
        )
        print(f"✓ Sensor created: {sensor}")
        print(f"  - Name: {sensor.name}")
        print(f"  - Camera: {sensor.camera_name}")
        print(f"  - Resolution: {sensor.width}x{sensor.height}")
        print(f"  - Normalize: {sensor.normalize}")
        return True
    except Exception as e:
        print(f"✗ Failed to create sensor: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_observation_space():
    """Test 2: Is observation space correct?"""
    print("\n" + "=" * 60)
    print("TEST 2: Observation Space")
    print("=" * 60)

    try:
        sensor = DepthCameraSensor(
            camera_name="gripper_depth_cam", height=64, width=64
        )
        obs_space = sensor.get_observation_space()
        print(f"✓ Observation space: {obs_space}")
        print(f"  - Shape: {obs_space.shape}")
        print(f"  - Expected: (4096,)")  # 64*64
        print(f"  - Dtype: {obs_space.dtype}")
        print(f"  - Low: {obs_space.low[0]}")
        print(f"  - High: {obs_space.high[0]}")

        assert obs_space.shape == (
            64 * 64,
        ), f"Shape mismatch: {obs_space.shape} != (4096,)"
        print("✓ Shape matches!")
        return True
    except Exception as e:
        print(f"✗ Observation space test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_metadata():
    """Test 3: Metadata populated?"""
    print("\n" + "=" * 60)
    print("TEST 3: Sensor Metadata")
    print("=" * 60)

    try:
        sensor = DepthCameraSensor(
            camera_name="gripper_depth_cam", height=128, width=128, normalize=True
        )
        metadata = sensor.get_metadata()
        print("✓ Metadata:")
        for key, value in metadata.items():
            print(f"  - {key}: {value}")
        return True
    except Exception as e:
        print(f"✗ Metadata test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_with_environment():
    """Test 4: Can sensor interact with environment?"""
    print("\n" + "=" * 60)
    print("TEST 4: Environment Integration")
    print("=" * 60)

    try:
        # Create environment
        print("Creating reach-v3 environment...")
        env = gym.make("Meta-World/MT1", env_name="reach-v3")
        print(f"✓ Environment created")
        print(f"  - Observation space: {env.observation_space.shape}")

        # Create sensor
        print("\nCreating depth camera sensor...")
        sensor = DepthCameraSensor(
            camera_name="gripper_depth_cam", height=64, width=64
        )
        print(f"✓ Sensor created: {sensor.name}")

        # Reset environment
        print("\nResetting environment...")
        obs, info = env.reset(seed=42)
        print(f"✓ Environment reset")
        print(f"  - Observation shape: {obs.shape}")

        # Reset sensor
        print("\nResetting sensor...")
        try:
            sensor.reset(env)
            print(f"✓ Sensor reset successful!")

            # Check camera exists
            camera_id = sensor._camera_id
            print(f"  - Camera ID: {camera_id}")
            print(f"  - Camera found in MuJoCo model!")
        except RuntimeError as e:
            print(f"✗ Sensor reset failed: {e}")
            print(
                "\n  This is expected if camera not properly defined in XML."
            )
            print(
                "  Make sure gripper_depth_cam exists in sawyer_reach_v3.xml"
            )
            env.close()
            return False

        # Try to validate
        print("\nValidating sensor...")
        if sensor.validate(env):
            print("✓ Sensor validated successfully!")
        else:
            print("✗ Sensor validation failed")

        # Try to update (will fail if not implemented)
        print("\nUpdating sensor (rendering depth)...")
        try:
            sensor.update(env)
            print("✓ Sensor update successful!")

            # Try to read
            depth_data = sensor.read()
            print(f"✓ Sensor read successful!")
            print(f"  - Depth data shape: {depth_data.shape}")
            print(f"  - Depth data dtype: {depth_data.dtype}")
            print(f"  - Depth range: [{depth_data.min():.3f}, {depth_data.max():.3f}]")

            # Try to get as image
            depth_image = sensor.get_depth_as_image()
            print(f"  - Depth image shape: {depth_image.shape}")

        except NotImplementedError:
            print("⚠ Sensor.update() not implemented yet!")
            print(
                "  This is expected. You need to implement DepthCameraSensor.update()"
            )
            print("  See metaworld/sensors/visual.py line ~90")
            env.close()
            return False
        except Exception as e:
            print(f"✗ Sensor update/read failed: {e}")
            import traceback

            traceback.print_exc()
            env.close()
            return False

        env.close()
        return True

    except Exception as e:
        print(f"✗ Environment integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MetaWorld Depth Camera Sensor - Basic Tests")
    print("=" * 60)

    results = []

    # Test 1: Basic creation
    results.append(("Sensor Creation", test_basic_sensor_creation()))

    # Test 2: Observation space
    results.append(("Observation Space", test_observation_space()))

    # Test 3: Metadata
    results.append(("Metadata", test_metadata()))

    # Test 4: Environment integration
    results.append(("Environment Integration", test_with_environment()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} - {name}")

    total_passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\nTotal: {total_passed}/{total} tests passed")

    if total_passed == total:
        print("\n🎉 All tests passed! Sensor is working correctly.")
    else:
        print(
            f"\n⚠ {total - total_passed} test(s) failed. See output above for details."
        )


if __name__ == "__main__":
    main()