# LiDAR Site Placement in MuJoCo XML - Visual Guide

## Understanding the XML Structure

The `xyz_base.xml` file defines the Sawyer robot and its environment. Here's the hierarchy:

```
<mujocoinclude>
  ├── <camera> tags (lines 16-22) - Fixed cameras in world
  │
  ├── <body name="base"> (line 30) - ROOT of the robot
  │   ├── <site name="basesite"> - Marker at robot base
  │   ├── <body name="controller_box"> - Robot controller
  │   ├── <body name="pedestal"> - Robot stand
  │   └── <body name="right_arm_base_link"> - ARM STARTS HERE (line 53)
  │       └── <body name="right_l0"> (line 57) - Shoulder joint
  │           ├── <body name="head"> - Robot head (not used)
  │           └── <body name="right_l1"> (line 79) - Upper arm
  │               └── <body name="right_l2"> (line 88) - Forearm
  │                   └── <body name="right_l3"> (line 93) - Wrist
  │                       └── <body name="right_l4"> (line 99)
  │                           └── <body name="right_l5"> (line 107)
  │                               └── <body name="right_l6"> (line 118)
  │                                   └── <body name="right_hand"> (line 123) - GRIPPER
  │                                       └── <body name="hand"> (line 153) - Actual gripper
  │                                           ├── gripperPOV camera
  │                                           ├── gripper_depth_cam
  │                                           └── gripper fingers
  │
  └── <body name="mocap"> (line 225) - Motion capture marker
```

---

## Where to Add the Fixed LiDAR Site

For a **fixed position** LiDAR (doesn't move with robot), add it **outside the robot body**, at the **same level as the cameras**.

### Option 1: Fixed Position in World (Recommended for Testing)

Add **after line 22** (after the existing cameras, before the robot base):

```xml
<!-- EXISTING CAMERAS (lines 16-22) -->
<camera pos="0 0.5 1.5" name="topview" />
<camera name="corner" mode="fixed" pos="-1.1 -0.4 0.6" xyaxes="-1 1 0 -0.2 -0.2 -1"/>
<camera name="corner2" fovy="60" mode="fixed" pos="1.3 -0.2 1.1" euler="3.9 2.3 0.6"/>
<camera name="corner3" fovy="45" mode="fixed" pos="0.9 0 1.5" euler="3.5 2.7 1"/>
<camera name="corner4" fovy="60" mode="fixed" pos="0.75 0.075 0.7" euler="3.9 2.3 0.6"/>

<!-- ADD THIS: Fixed LiDAR sensor -->
<body name="lidar_fixed_mount" pos="0.5 0 0.3">
    <inertial pos="0 0 0" mass="0.001" diaginertia="1e-08 1e-08 1e-08" />

    <!-- Site for LiDAR sensor origin (used in Python code) -->
    <site name="lidar_origin"
          pos="0 0 0"
          size="0.015"
          rgba="0 1 0 1"
          type="sphere"/>

    <!-- Visual marker (small box to see LiDAR in viewer) -->
    <geom name="lidar_mount_visual"
          type="box"
          size="0.03 0.03 0.06"
          rgba="0.1 0.8 0.1 0.8"
          contype="0"
          conaffinity="0"
          group="1"/>
</body>

<!-- THEN THE ROBOT BASE STARTS (line 30) -->
<body name="base" childclass="xyz_base" pos="0 0 0">
```

---

## Understanding the Tags

### `<body>` Tag
- Creates a physical body in the simulation
- `name`: Unique identifier
- `pos`: Position in parent frame (X, Y, Z) in meters
  - `pos="0.5 0 0.3"` = 50cm forward (X), 0cm left/right (Y), 30cm up (Z)

### `<inertial>` Tag
- Defines mass and inertia
- For sensors, use tiny mass (doesn't affect physics)
- `mass="0.001"` = 1 gram (negligible)
- `diaginertia="1e-08 1e-08 1e-08"` = tiny rotational inertia

### `<site>` Tag (THIS IS WHAT YOU NEED!)
- Invisible attachment point for sensors/tools
- `name="lidar_origin"` - **THIS is what your Python code will reference**
- `pos="0 0 0"` - Position relative to parent body (centered)
- `size="0.015"` - Visual size (only for rendering)
- `rgba="0 1 0 1"` - Color (Red, Green, Blue, Alpha)
  - `0 1 0 1` = Green, fully opaque
- `type="sphere"` - Shape for visualization

### `<geom>` Tag (Visual Marker)
- Actual geometry for visualization/collision
- `name="lidar_mount_visual"` - Unique name
- `type="box"` - Box shape
- `size="0.03 0.03 0.06"` - Half-sizes (X, Y, Z)
  - Total size = 6cm × 6cm × 12cm box
- `rgba="0.1 0.8 0.1 0.8"` - Light green, semi-transparent
- `contype="0"` and `conaffinity="0"` - **NO COLLISION** (important!)
- `group="1"` - Visualization group (won't interfere with physics)

---

## Coordinate System

MuJoCo uses **right-handed coordinate system**:

```
      Z (up)
      |
      |
      |_____ Y (left from robot's view)
     /
    /
   X (forward from robot's view)
```

### Example Positions:

| Position | X | Y | Z | Description |
|----------|---|---|---|-------------|
| In front of robot | 0.5 | 0 | 0.3 | 50cm forward, table height |
| Left side | 0 | 0.5 | 0.3 | 50cm to left |
| Above robot | 0 | 0 | 1.0 | Directly above, 1m up |
| Behind robot | -0.5 | 0 | 0.3 | 50cm behind |

---

## Alternative Placement Locations

### Option 2: On the Gripper (Moving with Robot)

Add **inside** the gripper body at line ~161 (after the `endEffector` site):

```xml
<body name="hand" pos="0 0 0.12" quat="-1 0 1 0">
    <camera name="behindGripper" mode="track" pos="0 0 -0.5" quat="0 1 0 0" fovy="60" />
    <camera name="gripperPOV" mode="track" pos="0.04 -0.06 0" quat="-1 -1.3 0 0" fovy="90" />
    <camera name="gripper_depth_cam" mode="track" pos="0.05 0 0" quat="0.707 0 0.707 0" fovy="75"/>

    <site name="endEffector" pos="0.04 0 0" size="0.01" rgba='1 1 1 0' />

    <!-- ADD THIS: LiDAR on gripper -->
    <site name="lidar_origin"
          pos="0.06 0 0"
          size="0.01"
          rgba="0 1 0 1"
          type="sphere"/>

    <geom name="rail" type="box" pos="-0.05 0 0" density="7850" size="0.005 0.055 0.005" ...
```

**Note:** For **testing**, start with **Option 1 (fixed position)** - it's simpler!

### Option 3: On the Wrist (More Stable than Gripper)

Add inside `right_l5` body at line ~107:

```xml
<body name="right_l5" pos="0 0.031 0.275" quat="0.707107 -0.707107 0 0">
    <inertial pos="0.0061133 -0.023697 0.076416" ... />
    <joint name="right_j5" pos="0 0 0" axis="0 0 1" ... />
    <geom type="mesh" ... mesh="l5" />

    <!-- ADD THIS: LiDAR on wrist -->
    <site name="lidar_origin"
          pos="0 0 0.05"
          size="0.01"
          rgba="0 1 0 1"
          type="sphere"/>
```

---

## Recommended Starting Configuration

**Use this for your first test:**

```xml
<!-- Add after line 22, before the robot base -->

<!-- Fixed LiDAR Sensor Mount -->
<body name="lidar_fixed_mount" pos="0.5 0 0.3">
    <inertial pos="0 0 0" mass="0.001" diaginertia="1e-08 1e-08 1e-08" />

    <site name="lidar_origin"
          pos="0 0 0"
          size="0.015"
          rgba="0 1 0 1"
          type="sphere"/>

    <geom name="lidar_mount_visual"
          type="box"
          size="0.03 0.03 0.06"
          rgba="0.1 0.8 0.1 0.8"
          contype="0"
          conaffinity="0"
          group="1"/>
</body>
```

**Why this position?**
- `pos="0.5 0 0.3"` places it:
  - **50cm in front** of robot base (X=0.5)
  - **Centered** left-right (Y=0)
  - **30cm above ground** (Z=0.3) - roughly table/workspace height
- Good view of the workspace
- Won't collide with robot
- Easy to visualize

---

## Testing Your Addition

### 1. Check if site exists (Python):

```python
import gymnasium as gym
import mujoco

env = gym.make("Meta-World/MT1", env_name="reach-v3")
env.reset()

# Try to find the site
try:
    site_id = env.unwrapped.model.site("lidar_origin").id
    print(f"✓ Found lidar_origin site with ID: {site_id}")

    # Get position
    pos = env.unwrapped.data.site_xpos[site_id]
    print(f"  Position: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
except KeyError:
    print("✗ Site 'lidar_origin' not found!")
    print("Available sites:", [env.unwrapped.model.site(i).name
                               for i in range(env.unwrapped.model.nsite)])

env.close()
```

### 2. Visualize in MuJoCo viewer:

```python
import gymnasium as gym

env = gym.make("Meta-World/MT1", env_name="reach-v3", render_mode="human")
env.reset()

# Step through to see the LiDAR mount (green sphere/box)
for _ in range(100):
    action = env.action_space.sample()
    env.step(action)

env.close()
```

The green sphere (site) and light green box (visual marker) should be visible in the scene!

---

## Common Issues

### ❌ "Site 'lidar_origin' not found"
- Check XML syntax (closing tags, proper nesting)
- Make sure you edited the correct XML file
- Verify the file is being loaded (check environment's `model_name`)

### ❌ LiDAR rays all return max_range
- Site might be inside a collision geometry
- Try different position (move up/sideways)
- Check `contype="0"` on visual marker (no collision)

### ❌ Can't see the LiDAR in viewer
- Increase `size` of site: `size="0.05"` (bigger sphere)
- Make sure `rgba` has alpha=1: `rgba="0 1 0 1"`
- Check that visualization is enabled in viewer (press `G` to toggle geom groups)

---

## Summary

**What you need to do:**

1. Open `/home/lukas/Projects/Metaworld/metaworld/assets/objects/assets/xyz_base.xml`
2. Find line 22 (after the last camera)
3. Add the LiDAR mount body (10 lines of XML)
4. Save the file
5. Test that the site exists in Python
6. Use `origin_site="lidar_origin"` when creating your sensor

That's it! The site is now available for your LiDAR sensor to use.