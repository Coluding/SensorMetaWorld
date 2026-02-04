#!/usr/bin/env python3
"""Enhanced keyboard control for MetaWorld with debug functions.

This extends the original keyboard_control.py with debug inspection functions.

Controls:
    Movement (PyGame window must be in focus):
        W/S - Move Y-axis
        A/D - Move X-axis
        K/J - Move Z up/down
        H/L - Close/open gripper
        X - Toggle lock action
        R - Reset environment

    Debug Functions (press in console, not PyGame window):
        1 - Print contact information
        2 - Print gripper state
        3 - Print object positions
        4 - Print sensor readings
        C - Toggle continuous contact monitoring

Usage:
    python scripts/keyboard_control_debug.py
"""

import sys
import numpy as np
import pygame
from pygame.locals import KEYDOWN, QUIT
import threading
import metaworld

import gymnasium as gym


class DebugKeyboardController:
    """Keyboard controller with debug functions."""

    def __init__(self, env_name="basketball-v3"):
        """Initialize controller.

        Args:
            env_name: MetaWorld environment name
        """
        self.env_name = env_name

        # Setup pygame
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption(f"MetaWorld Control - {env_name}")

        # Create environment
        print(f"\nCreating environment: {env_name}...")
        self.env = gym.make('Meta-World/MT1', env_name=env_name, render_mode='human')
        self.env.reset()
        print("✓ Environment ready!")

        # Control state
        self.action = np.zeros(4, dtype=np.float32)
        self.lock_action = False

        # Debug state
        self.continuous_contact_monitor = False
        self.running = True

        # Cache geometry IDs
        try:
            self.left_finger_id = self.env.unwrapped.model.geom("leftpad_geom").id
            self.right_finger_id = self.env.unwrapped.model.geom("rightpad_geom").id
        except:
            self.left_finger_id = None
            self.right_finger_id = None

        self._print_controls()

    def _print_controls(self):
        """Print control instructions."""
        print("\n" + "=" * 70)
        print("CONTROLS")
        print("=" * 70)
        print("PyGame Window Controls (window must be in focus):")
        print("  W/S - Move Y-axis")
        print("  A/D - Move X-axis")
        print("  K/J - Move Z up/down")
        print("  H/L - Close/open gripper")
        print("  X - Toggle lock action")
        print("  R - Reset environment")
        print("\nConsole Debug Commands:")
        print("  1 - Print contact information")
        print("  2 - Print gripper state")
        print("  3 - Print object positions")
        print("  4 - Print sensor readings")
        print("  c - Toggle continuous contact monitoring")
        print("  q - Quit")
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

    def toggle_contact_monitor(self):
        """Toggle continuous contact monitoring."""
        self.continuous_contact_monitor = not self.continuous_contact_monitor
        status = "ENABLED" if self.continuous_contact_monitor else "DISABLED"
        print(f"\n[Continuous contact monitoring: {status}]\n")

    def console_input_thread(self):
        """Thread for handling console input."""
        print("\n[Console ready for debug commands. Type '1', '2', '3', '4', 'c', or 'q']\n")

        while self.running:
            try:
                cmd = input().strip().lower()

                if cmd == '1':
                    self.print_contacts()
                elif cmd == '2':
                    self.print_gripper_state()
                elif cmd == '3':
                    self.print_object_positions()
                elif cmd == '4':
                    self.print_sensor_readings()
                elif cmd == 'c':
                    self.toggle_contact_monitor()
                elif cmd == 'q':
                    print("\n[Quitting...]")
                    self.running = False
                    pygame.quit()
                    break

            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")

    def run(self):
        """Main control loop."""
        # Start console input thread
        console_thread = threading.Thread(target=self.console_input_thread, daemon=True)
        console_thread.start()

        # Key mapping
        char_to_action = {
            ord('w'): np.array([0, -1, 0, 0]),
            ord('a'): np.array([1, 0, 0, 0]),
            ord('s'): np.array([0, 1, 0, 0]),
            ord('d'): np.array([-1, 0, 0, 0]),
            ord('k'): np.array([0, 0, 1, 0]),
            ord('j'): np.array([0, 0, -1, 0]),
        }

        step_count = 0

        try:
            while self.running:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == QUIT:
                        self.running = False
                        break

                    if event.type == KEYDOWN:
                        char = event.dict["key"]

                        # Gripper control
                        if char == ord('h'):  # Close gripper
                            self.action[3] = -1
                        elif char == ord('l'):  # Open gripper
                            self.action[3] = 1

                        # Lock action
                        elif char == ord('x'):
                            self.lock_action = not self.lock_action
                            status = "LOCKED" if self.lock_action else "UNLOCKED"
                            print(f"[Action {status}]")

                        # Reset
                        elif char == ord('r'):
                            print("\n[Resetting environment...]")
                            self.env.reset()
                            self.action = np.zeros(4, dtype=np.float32)
                            step_count = 0
                            print("✓ Environment reset!\n")

                # Check currently pressed keys for movement (continuous control)
                keys = pygame.key.get_pressed()
                obs, reward, terminated, truncated, info = self.env.step(self.action)
                step_count += 1

                # Continuous contact monitoring
                if self.continuous_contact_monitor:
                    ncon = self.env.unwrapped.data.ncon
                    finger_contacts = 0
                    for i in range(ncon):
                        contact = self.env.unwrapped.data.contact[i]
                        if (contact.geom1 in [self.left_finger_id, self.right_finger_id] or
                            contact.geom2 in [self.left_finger_id, self.right_finger_id]):
                            finger_contacts += 1

                    if finger_contacts > 0:
                        print(f"[Contacts: {ncon} total, {finger_contacts} finger]", end='\r')

                # Handle episode end
                if terminated or truncated:
                    print(f"\n[Episode ended after {step_count} steps]")
                    if 'success' in info:
                        print(f"  Success: {info['success']}")
                    print("[Auto-resetting...]\n")
                    self.env.reset()
                    step_count = 0

                # Render
                self.env.render()

        except KeyboardInterrupt:
            print("\n[Interrupted]")
        finally:
            self.running = False
            self.env.close()
            pygame.quit()
            print("\n✓ Goodbye!\n")


def main():
    """Main entry point."""
    env_name = sys.argv[1] if len(sys.argv) > 1 else "assembly-v3"

    controller = DebugKeyboardController(env_name=env_name)
    controller.run()


if __name__ == "__main__":
    main()