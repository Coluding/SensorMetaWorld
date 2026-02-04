#!/usr/bin/env python3
"""Policy execution with debug functions for MetaWorld.

This script runs expert policies while allowing inspection of contacts,
gripper state, and object positions during execution.

Usage:
    python scripts/policy_debug.py reach-v3
    python scripts/policy_debug.py pick-place-v3
"""

import sys
import time
import numpy as np
import gymnasium as gym
import metaworld
from importlib import import_module


class PolicyDebugger:
    """Policy executor with debug inspection functions."""

    def __init__(self, env_name: str = "reach-v3"):
        """Initialize policy debugger.

        Args:
            env_name: MetaWorld environment name
        """
        self.env_name = env_name

        # Create environment
        print(f"\nCreating environment: {env_name}...")
        self.env = gym.make('Meta-World/MT1', env_name=env_name, render_mode='human')
        self.env.reset()
        print("✓ Environment ready!")

        # Load expert policy
        print(f"Loading expert policy for {env_name}...")
        self.policy = self._load_policy(env_name)
        print("✓ Policy loaded!")

        # Cache geometry IDs
        try:
            self.left_finger_id = self.env.unwrapped.model.geom("leftpad_geom").id
            self.right_finger_id = self.env.unwrapped.model.geom("rightpad_geom").id
        except:
            self.left_finger_id = None
            self.right_finger_id = None

        # Debug flags
        self.auto_print_contacts = True
        self.auto_print_gripper = False
        self.auto_print_objects = False
        self.continuous_contact_monitor = True
        self.pause_on_contact = False
        self.step_delay = 1.0  # seconds

        self._print_commands()

    def _load_policy(self, env_name: str):
        """Load the expert policy for the given environment."""
        # Convert env name to policy class name
        # e.g., "reach-v3" -> "SawyerReachV3Policy"
        parts = env_name.replace("-", "_").split("_")
        class_name = "Sawyer" + "".join(p.capitalize() for p in parts[:-1]) + parts[-1].upper() + "Policy"
        module_name = f"metaworld.policies.sawyer_{env_name.replace('-', '_')}_policy"

        try:
            module = import_module(module_name)
            policy_class = getattr(module, class_name)
            return policy_class()
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not load policy: {e}")
            print("Continuing with random actions...")
            return None

    def _print_commands(self):
        """Print available commands."""
        print("\n" + "=" * 70)
        print("COMMANDS")
        print("=" * 70)
        print("During execution, press Ctrl+C to pause and enter commands:")
        print("\nDebug Commands:")
        print("  1 - Print contact information")
        print("  2 - Print gripper state")
        print("  3 - Print object positions")
        print("  4 - Print sensor readings")
        print("\nAuto-Print Toggles:")
        print("  ac - Toggle auto-print contacts each step")
        print("  ag - Toggle auto-print gripper state each step")
        print("  ao - Toggle auto-print object positions each step")
        print("  c  - Toggle continuous contact monitoring")
        print("\nExecution Control:")
        print("  pc - Toggle pause on finger contact")
        print("  d <seconds> - Set delay between steps (e.g., 'd 0.1')")
        print("  r  - Reset environment")
        print("  s  - Continue (resume execution)")
        print("  q  - Quit")
        print("=" * 70 + "\n")

    def print_contacts(self):
        """Debug function: Print detailed contact information."""
        print("\n" + "=" * 70)
        print("CONTACT INFORMATION")
        print("=" * 70)

        ncon = self.env.unwrapped.data.ncon
        print(f"Total active contacts: {ncon}\n")

        if ncon == 0:
            print("  No contacts detected.\n")
            return

        for i in range(ncon):
            contact = self.env.unwrapped.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Get geometry names
            try:
                name1 = self.env.unwrapped.model.geom(geom1).name
                if not name1:
                    name1 = f"geom_{geom1}"
            except:
                name1 = f"geom_{geom1}"

            try:
                name2 = self.env.unwrapped.model.geom(geom2).name
                if not name2:
                    name2 = f"geom_{geom2}"
            except:
                name2 = f"geom_{geom2}"

            # Check if involves fingers
            involves_left = (geom1 == self.left_finger_id or geom2 == self.left_finger_id)
            involves_right = (geom1 == self.right_finger_id or geom2 == self.right_finger_id)

            marker = ""
            if involves_left:
                marker = "👈 LEFT FINGER  "
            elif involves_right:
                marker = "👉 RIGHT FINGER "
            else:
                marker = "   "

            print(f"  {marker}Contact {i}: {name1} (ID {geom1}) <-> {name2} (ID {geom2})")
            print(f"              Position: [{contact.pos[0]:.3f}, {contact.pos[1]:.3f}, {contact.pos[2]:.3f}]")

        print("=" * 70 + "\n")

    def print_gripper_state(self):
        """Debug function: Print gripper state."""
        print("\n" + "=" * 70)
        print("GRIPPER STATE")
        print("=" * 70)

        try:
            # Gripper position
            gripper_pos = self.env.unwrapped.data.body("hand").xpos
            print(f"Gripper position: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")

            # Finger joint positions
            left_joint = self.env.unwrapped.data.joint("l_close").qpos[0]
            right_joint = self.env.unwrapped.data.joint("r_close").qpos[0]
            print(f"Left finger joint:  {left_joint:.4f}")
            print(f"Right finger joint: {right_joint:.4f}")

            # Gripper width
            gripper_width = abs(left_joint) + abs(right_joint)
            print(f"Gripper width (approx): {gripper_width:.4f}")

        except Exception as e:
            print(f"Error reading gripper state: {e}")

        print("=" * 70 + "\n")

    def print_object_positions(self):
        """Debug function: Print positions of task objects."""
        print("\n" + "=" * 70)
        print("OBJECT POSITIONS")
        print("=" * 70)

        # Robot body names to skip
        robot_bodies = {'base', 'pedestal', 'torso', 'controller_box', 'pedestal_feet',
                       'right_arm_base_link', 'right_l0', 'head', 'screen', 'head_camera',
                       'right_torso_itb', 'right_l1', 'right_l2', 'right_l3', 'right_l4',
                       'right_arm_itb', 'right_l5', 'right_hand_camera', 'right_wrist',
                       'right_l6', 'right_hand', 'rightclaw', 'rightpad', 'leftclaw',
                       'leftpad', 'hand', 'right_l4_2', 'right_l2_2', 'right_l1_2',
                       'mocap', 'gripper_depth_cam_mount'}

        print("\nTask objects:")
        found_objects = False
        for i in range(self.env.unwrapped.model.nbody):
            body_name = self.env.unwrapped.model.body(i).name
            if body_name and body_name not in robot_bodies:
                pos = self.env.unwrapped.data.body(i).xpos
                print(f"  {body_name:20s}: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")
                found_objects = True

        if not found_objects:
            print("  (No task objects found)")

        print("\nGripper:")
        gripper_pos = self.env.unwrapped.data.body("hand").xpos
        print(f"  hand:                 [{gripper_pos[0]:6.3f}, {gripper_pos[1]:6.3f}, {gripper_pos[2]:6.3f}]")

        print("=" * 70 + "\n")

    def print_sensor_readings(self):
        """Debug function: Print sensor readings."""
        print("\n" + "=" * 70)
        print("SENSOR READINGS")
        print("=" * 70)
        print("  (Not implemented yet - integrate sensors here)")
        print("=" * 70 + "\n")

    def check_finger_contacts(self) -> tuple[bool, bool]:
        """Check if fingers are in contact.

        Returns:
            (left_touching, right_touching)
        """
        left_touching = False
        right_touching = False

        ncon = self.env.unwrapped.data.ncon
        for i in range(ncon):
            contact = self.env.unwrapped.data.contact[i]
            if contact.geom1 == self.left_finger_id or contact.geom2 == self.left_finger_id:
                left_touching = True
            if contact.geom1 == self.right_finger_id or contact.geom2 == self.right_finger_id:
                right_touching = True

        return left_touching, right_touching

    def handle_command(self):
        """Handle user commands during pause."""
        print("\n[PAUSED - Enter command, or 's' to continue]")
        cmd = input("> ").strip().lower()

        if cmd == '1':
            self.print_contacts()
        elif cmd == '2':
            self.print_gripper_state()
        elif cmd == '3':
            self.print_object_positions()
        elif cmd == '4':
            self.print_sensor_readings()
        elif cmd == 'ac':
            self.auto_print_contacts = not self.auto_print_contacts
            print(f"Auto-print contacts: {'ON' if self.auto_print_contacts else 'OFF'}")
        elif cmd == 'ag':
            self.auto_print_gripper = not self.auto_print_gripper
            print(f"Auto-print gripper: {'ON' if self.auto_print_gripper else 'OFF'}")
        elif cmd == 'ao':
            self.auto_print_objects = not self.auto_print_objects
            print(f"Auto-print objects: {'ON' if self.auto_print_objects else 'OFF'}")
        elif cmd == 'c':
            self.continuous_contact_monitor = not self.continuous_contact_monitor
            print(f"Continuous contact monitor: {'ON' if self.continuous_contact_monitor else 'OFF'}")
        elif cmd == 'pc':
            self.pause_on_contact = not self.pause_on_contact
            print(f"Pause on finger contact: {'ON' if self.pause_on_contact else 'OFF'}")
        elif cmd.startswith('d '):
            try:
                delay = float(cmd.split()[1])
                self.step_delay = delay
                print(f"Step delay set to {delay}s")
            except:
                print("Invalid delay format. Use: d 0.1")
        elif cmd == 'r':
            print("Resetting environment...")
            self.env.reset()
            print("✓ Reset complete")
        elif cmd == 's':
            print("Continuing...\n")
            return True
        elif cmd == 'q':
            print("Quitting...")
            return False
        else:
            print(f"Unknown command: {cmd}")

        # Continue prompting unless user said continue or quit
        return None if cmd not in ['s', 'q'] else (cmd == 's')

    def run(self, max_episodes: int = 1, max_steps_per_episode: int = 500):
        """Run policy with debug monitoring.

        Args:
            max_episodes: Number of episodes to run
            max_steps_per_episode: Maximum steps per episode
        """
        print(f"\nStarting policy execution...")
        print(f"Episodes: {max_episodes}, Max steps: {max_steps_per_episode}\n")

        try:
            for episode in range(max_episodes):
                print(f"\n{'='*70}")
                print(f"EPISODE {episode + 1}/{max_episodes}")
                print('='*70)

                obs, info = self.env.reset()
                done = False
                step = 0

                while not done and step < max_steps_per_episode:
                    # Get action from policy
                    if self.policy:
                        action = self.policy.get_action(obs)
                    else:
                        action = self.env.action_space.sample()

                    # Step environment
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    step += 1

                    # Check for success
                    done = int(info.get('success', 0)) == 1

                    # Auto-print enabled features
                    if self.auto_print_contacts:
                        self.print_contacts()
                    if self.auto_print_gripper:
                        self.print_gripper_state()
                    if self.auto_print_objects:
                        self.print_object_positions()

                    # Continuous contact monitoring
                    if self.continuous_contact_monitor:
                        left, right = self.check_finger_contacts()
                        ncon = self.env.unwrapped.data.ncon
                        if left or right:
                            status = f"[Contacts: {ncon} total | "
                            if left:
                                status += "👈 LEFT "
                            if right:
                                status += "👉 RIGHT "
                            status += "]"
                            print(status, end='\r')

                    # Pause on finger contact
                    if self.pause_on_contact:
                        left, right = self.check_finger_contacts()
                        if left or right:
                            print(f"\n[FINGER CONTACT DETECTED at step {step}]")
                            self.print_contacts()
                            while True:
                                result = self.handle_command()
                                if result is not None:
                                    if not result:  # quit
                                        return
                                    break  # continue

                    # Render
                    self.env.render()


                    # Step delay
                    if self.step_delay > 0:
                        time.sleep(self.step_delay)

                # Episode summary
                print(f"\n{'='*70}")
                print(f"Episode {episode + 1} finished:")
                print(f"  Steps: {step}")
                print(f"  Success: {info.get('success', False)}")
                print('='*70)

        except KeyboardInterrupt:
            print("\n\n[Interrupted - Entering command mode]")
            while True:
                result = self.handle_command()
                if result is False:  # quit
                    break
                elif result is True:  # continue
                    self.run(max_episodes=max_episodes - episode, max_steps_per_episode=max_steps_per_episode)
                    break

        finally:
            self.env.close()
            print("\n✓ Goodbye!\n")


def main():
    """Main entry point."""
    env_name = sys.argv[1] if len(sys.argv) > 1 else "basketball-v3"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    debugger = PolicyDebugger(env_name=env_name)
    debugger.run(max_episodes=episodes)


if __name__ == "__main__":
    main()