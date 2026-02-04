# LiDAR Sensor Implementation Guide

## Overview
This guide walks through implementing a 2D LiDAR sensor for Meta-World using three approaches:
1. **Python + NumPy** (baseline, ~1-5ms for 64-256 rays)
2. **C++ + Pybind11** (optimized, ~0.3-1ms for 256-512 rays)
3. **Embree Intel Raytracer** (production, ~0.1ms for 512+ rays)

We'll start with Python+NumPy to understand the algorithm, then optimize progressively.

---

## Part 1: Understanding the Python+NumPy Implementation

### Core Concepts

#### 1. What is 2D LiDAR?
A 2D LiDAR sensor:
- Emits rays in a **single plane** (e.g., XY plane at fixed Z height)
- Measures **distance to nearest obstacle** along each ray
- Common pattern: 360° sweep with N evenly-spaced rays
- Output: Array of N distance measurements

```
      Ray 0 (0°)
         |
   Ray 7 |  Ray 1 (45°)
      \  |  /
       \ | /
        \|/
    ----[S]---- Ray 4 (180°)
        /|\
       / | \
      /  |  \
   Ray 5 |  Ray 3
         |
      Ray 6 (270°)

S = Sensor origin
```

#### 2. MuJoCo Raycasting API

MuJoCo provides two key functions:

**`mj_ray()` - Single ray:**
```python
distance = mujoco.mj_ray(
    m=model,              # MjModel
    d=data,               # MjData
    pnt=origin,           # Ray origin [x, y, z] - shape (3,)
    vec=direction,        # Ray direction [dx, dy, dz] - shape (3,)
    geomgroup=None,       # Optional: which geom groups to check
    flg_static=1,         # Include static geometries
    bodyexclude=-1,       # Exclude specific body (-1 = none)
    geomid=geom_id_out    # Output: which geom was hit
)
# Returns: distance to hit, or -1 if no hit
```

**`mj_multiRay()` - Batch rays (MUCH FASTER!):**
```python
mujoco.mj_multiRay(
    m=model,
    d=data,
    pnt=origin,                    # Single origin [x, y, z] - shape (3,)
    vec=directions,                # Multiple directions [N, 3] - shape (N, 3)
    geomgroup=None,
    flg_static=1,
    bodyexclude=-1,
    geomid=geom_ids,               # Output array [N,] - which geoms hit
    dist=distances,                # Output array [N,] - distances
    nray=num_rays,                 # Number of rays
    cutoff=max_range               # Max distance (rays beyond return max_range)
)
# Modifies geom_ids and distances arrays in-place
```

**Key insight:** `mj_multiRay` is vectorized inside MuJoCo's C code, so it's **much faster** than calling `mj_ray()` N times in Python!

---

### Implementation Steps for Python+NumPy 2D LiDAR

#### Step 1: Sensor Configuration

Define sensor parameters:
```python
class Lidar2DSensor(SensorBase):
    def __init__(
        self,
        origin_site: str = "lidar_origin",  # MuJoCo site name
        num_rays: int = 64,                  # Number of rays
        max_range: float = 2.0,              # Max distance (meters)
        fov_degrees: float = 360.0,          # Field of view
        z_height: float = 0.0,               # Height in local frame
    ):
        self.origin_site = origin_site
        self.num_rays = num_rays
        self.max_range = max_range
        self.fov_degrees = fov_degrees
        self.z_height = z_height
```

#### Step 2: Pre-compute Ray Directions (in `reset()`)

Generate ray directions **once** during reset, then reuse:

```python
def reset(self, env: SawyerXYZEnv) -> None:
    # Validate site exists
    try:
        self._site_id = env.unwrapped.model.site(self.origin_site).id
    except KeyError:
        raise RuntimeError(f"Site '{self.origin_site}' not found")

    # Pre-compute ray directions in LOCAL frame
    # For 2D LiDAR in XY plane:
    angles = np.linspace(
        0,
        np.deg2rad(self.fov_degrees),
        self.num_rays,
        endpoint=False  # Don't duplicate 0° and 360°
    )

    # Ray directions: (num_rays, 3)
    self._ray_dirs_local = np.zeros((self.num_rays, 3), dtype=np.float64)
    self._ray_dirs_local[:, 0] = np.cos(angles)  # X
    self._ray_dirs_local[:, 1] = np.sin(angles)  # Y
    self._ray_dirs_local[:, 2] = 0.0             # Z (2D = stays in plane)

    # Allocate output buffers (reused each step)
    self._distances = np.full(self.num_rays, self.max_range, dtype=np.float64)
    self._geom_ids = np.zeros(self.num_rays, dtype=np.int32)
```

**Why pre-compute?**
- Ray directions are constant in the sensor's local frame
- Only need to rotate them to world frame each step (cheap operation)
- Saves computation time

#### Step 3: Get Sensor Pose (in `update()`)

Each step, get the sensor's **world-space position and orientation**:

```python
def update(self, env: SawyerXYZEnv) -> None:
    model = env.unwrapped.model
    data = env.unwrapped.data

    # Get sensor origin in world coordinates
    origin = data.site_xpos[self._site_id].copy()  # Shape: (3,)

    # Get sensor orientation (rotation matrix)
    rotation_mat = data.site_xmat[self._site_id].reshape(3, 3)  # Shape: (3, 3)
```

**MuJoCo conventions:**
- `data.site_xpos[id]`: Position in world frame [x, y, z]
- `data.site_xmat[id]`: Rotation matrix (3x3) from local → world frame
  - Stored as flat array of 9 floats, need to reshape to (3, 3)

#### Step 4: Transform Rays to World Frame

Rotate pre-computed local directions to world frame:

```python
    # Transform ray directions from local to world frame
    # ray_dirs_world = rotation_mat @ ray_dirs_local^T
    ray_dirs_world = (rotation_mat @ self._ray_dirs_local.T).T  # (N, 3)
```

**Why this works:**
- `rotation_mat`: (3, 3) matrix transforms local → world
- `ray_dirs_local.T`: (3, N) transposed for matrix multiplication
- `@ operator`: Matrix multiplication
- `.T` at end: Transpose back to (N, 3)

Alternatively (more explicit):
```python
ray_dirs_world = np.zeros((self.num_rays, 3), dtype=np.float64)
for i in range(self.num_rays):
    ray_dirs_world[i] = rotation_mat @ self._ray_dirs_local[i]
```

#### Step 5: Call MuJoCo Raycaster

```python
    # Cast all rays at once
    mujoco.mj_multiRay(
        m=model,
        d=data,
        pnt=origin,                      # Ray origin (3,)
        vec=ray_dirs_world.ravel(),      # Flatten to 1D: (N*3,)
        geomgroup=None,                  # Check all geom groups
        flg_static=1,                    # Include static geoms
        bodyexclude=-1,                  # Don't exclude any body
        geomid=self._geom_ids,           # Output: geom IDs (N,)
        dist=self._distances,            # Output: distances (N,)
        nray=self.num_rays,              # Number of rays
        cutoff=self.max_range            # Max range
    )
```

**Important notes:**
- `vec` must be **flattened** to 1D array of shape (N*3,)
- `geomid` and `dist` are **modified in-place**
- `dist[i] == -1` means ray i didn't hit anything (can replace with max_range)
- `cutoff` stops rays early if they exceed max_range (performance optimization)

#### Step 6: Post-process Results

```python
    # Replace -1 (no hit) with max_range
    self._distances[self._distances < 0] = self.max_range

    # Optional: Clip to [0, max_range]
    np.clip(self._distances, 0.0, self.max_range, out=self._distances)
```

#### Step 7: Return Data

```python
def read(self) -> npt.NDArray[np.float64]:
    """Return distance measurements."""
    return self._distances.copy()  # Return copy to prevent external modification

def get_observation_space(self) -> spaces.Space:
    """Define observation space."""
    return spaces.Box(
        low=0.0,
        high=self.max_range,
        shape=(self.num_rays,),
        dtype=np.float32
    )
```

---

### Complete Python Structure

```python
# metaworld/sensors/lidar.py

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
import mujoco

from metaworld.sensors.base import SensorBase

if TYPE_CHECKING:
    from metaworld.sawyer_xyz_env import SawyerXYZEnv


class Lidar2DSensor(SensorBase):
    """2D LiDAR sensor using MuJoCo raycasting.

    Emits rays in a horizontal plane around a fixed point in the environment.
    Returns distance to nearest obstacle for each ray.
    """

    def __init__(
        self,
        origin_site: str,           # Where is the LiDAR mounted?
        num_rays: int = 64,          # Resolution
        max_range: float = 2.0,      # Max distance
        fov_degrees: float = 360.0,  # Field of view
    ):
        self.origin_site = origin_site
        self.num_rays = num_rays
        self.max_range = max_range
        self.fov_degrees = fov_degrees

        # Internal buffers (allocated in reset())
        self._site_id: int | None = None
        self._ray_dirs_local: npt.NDArray[np.float64] | None = None
        self._distances: npt.NDArray[np.float64] | None = None
        self._geom_ids: npt.NDArray[np.int32] | None = None

    @property
    def name(self) -> str:
        return f"lidar2d_{self.origin_site}_{self.num_rays}rays"

    def reset(self, env: SawyerXYZEnv) -> None:
        """Initialize sensor and pre-compute ray directions."""
        # TODO: Implement (Step 2 above)
        pass

    def update(self, env: SawyerXYZEnv) -> None:
        """Cast rays and measure distances."""
        # TODO: Implement (Steps 3-6 above)
        pass

    def read(self) -> npt.NDArray[np.float64]:
        """Return distance measurements."""
        # TODO: Implement (Step 7 above)
        pass

    def get_observation_space(self) -> spaces.Space:
        """Define observation space."""
        # TODO: Implement (Step 7 above)
        pass

    def get_metadata(self) -> dict[str, str | int | float]:
        """Return sensor metadata."""
        return {
            "type": "lidar",
            "subtype": "2d",
            "num_rays": self.num_rays,
            "max_range": self.max_range,
            "fov_degrees": self.fov_degrees,
            "origin_site": self.origin_site,
        }

    def validate(self, env: SawyerXYZEnv) -> bool:
        """Check if origin site exists."""
        try:
            env.unwrapped.model.site(self.origin_site).id
            return True
        except KeyError:
            return False
```

---

## Key NumPy Operations You'll Use

### 1. **Generating angles:**
```python
angles = np.linspace(start, stop, num, endpoint=False)
```

### 2. **Trigonometry:**
```python
x = np.cos(angles)  # Vectorized cosine
y = np.sin(angles)  # Vectorized sine
```

### 3. **Matrix multiplication:**
```python
result = A @ B  # Python 3.5+ matrix multiply operator
# or
result = np.matmul(A, B)
# or
result = np.dot(A, B)  # For 2D arrays
```

### 4. **Reshaping:**
```python
flat = array.ravel()           # Flatten to 1D
mat = array.reshape(3, 3)      # Reshape to (3, 3)
transposed = array.T           # Transpose
```

### 5. **Array slicing:**
```python
array[:, 0]  # All rows, first column
array[i, :]  # i-th row, all columns
array.copy() # Deep copy
```

### 6. **Conditional operations:**
```python
array[array < 0] = max_range   # Replace negative values
np.clip(array, min, max)       # Clamp values
```

---

## MuJoCo XML: Adding Fixed LiDAR Site

You need to add a "site" (attachment point) to the MuJoCo scene:

```xml
<!-- In metaworld/assets/objects/assets/xyz_base.xml -->
<!-- Add inside <worldbody> tag, outside robot body for fixed position -->

<body name="lidar_fixed" pos="0.5 0 0.3">
    <site name="lidar_origin"
          pos="0 0 0"
          size="0.01"
          rgba="0 1 0 1"  <!-- Green marker -->
          type="sphere"/>
    <geom name="lidar_mount"
          type="box"
          size="0.02 0.02 0.05"
          rgba="0.3 0.3 0.3 1"
          contype="0"      <!-- No collision -->
          conaffinity="0"  <!-- No collision -->
          group="1"/>       <!-- Visualization only -->
</body>
```

**Explanation:**
- `pos="0.5 0 0.3"`: Fixed position in world (50cm in X, 0 in Y, 30cm in Z)
- `site`: Invisible attachment point (what we reference in code)
- `geom`: Visual marker (small box to see where LiDAR is)
- `contype="0"`: Don't collide with anything
- `rgba`: RGBA color (R, G, B, Alpha)

---

## Testing Strategy

### 1. **Unit test: Ray direction generation**
```python
def test_ray_directions():
    sensor = Lidar2DSensor("lidar_origin", num_rays=4, fov_degrees=360)
    # After reset, check ray directions
    # Expected: [1,0,0], [0,1,0], [-1,0,0], [0,-1,0] for 4 rays at 0°,90°,180°,270°
```

### 2. **Unit test: Distance measurement**
```python
def test_distance_measurement():
    env = gym.make("Meta-World/MT1", env_name="reach-v3")
    sensor = Lidar2DSensor("lidar_origin", num_rays=64)
    env.reset()
    sensor.reset(env)
    sensor.update(env)
    distances = sensor.read()

    assert distances.shape == (64,)
    assert np.all(distances >= 0)
    assert np.all(distances <= sensor.max_range)
```

### 3. **Visualization test:**
```python
def visualize_lidar():
    # Create environment
    # Add LiDAR sensor
    # Run for N steps
    # Plot polar plot of ray distances

    import matplotlib.pyplot as plt

    angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(angles, distances, 'r-')
    ax.set_ylim(0, max_range)
    plt.show()
```

---

## Performance Considerations

### Python+NumPy Performance Tips:

1. **Pre-allocate arrays** (in `reset()`):
   - ✅ `self._distances = np.zeros(N)`
   - ❌ `distances = []` then `append()` in loop

2. **Reuse buffers** (don't create new arrays each step):
   - ✅ `mj_multiRay(..., dist=self._distances, ...)`
   - ❌ `distances = np.zeros(N)` in `update()`

3. **Use vectorized operations**:
   - ✅ `rotation_mat @ ray_dirs.T`
   - ❌ `for i in range(N): ...`

4. **Use `mj_multiRay` not `mj_ray` in loop**:
   - ✅ One call to `mj_multiRay` with N rays
   - ❌ N calls to `mj_ray`

5. **Avoid Python loops** for numerical operations:
   - ✅ `np.cos(angles)` (vectorized)
   - ❌ `[np.cos(a) for a in angles]` (list comprehension)

### Expected Performance:
- **64 rays**: ~1-2ms per update
- **128 rays**: ~2-3ms per update
- **256 rays**: ~4-6ms per update

(Will improve significantly with C++ implementation!)

---

## Next Steps

1. **Implement the basic Python class** in `metaworld/sensors/lidar.py`
2. **Add the XML site** to `xyz_base.xml`
3. **Write unit tests** in `tests/sensors/test_lidar.py`
4. **Create visualization script** in `scripts/demo_lidar.py`
5. **Profile performance** to establish baseline
6. **Move to C++ implementation** (next guide)

---

## Questions to Think About

1. **What coordinate frame do you want for ray directions?**
   - Local (relative to sensor orientation)
   - World (fixed directions regardless of sensor rotation)

2. **How to handle "no hit"?**
   - Return `max_range` (sensor's limit)
   - Return `-1` or `np.inf` (explicit no-hit marker)
   - Return `None` (but arrays don't support this)

3. **What additional data is useful?**
   - Just distances (most common)
   - Geom IDs (what was hit)
   - Surface normals (requires extra computation)
   - Intensity/reflectivity (simulated)

4. **Fixed vs. moving sensor?**
   - Fixed: Simpler, origin is constant
   - Moving: More realistic, origin changes each step

---

## Debugging Tips

### Common Issues:

1. **All rays return -1 (no hits):**
   - Check if `cutoff` is too small
   - Check if sensor is inside a geometry (collision)
   - Check if `flg_static` is correct

2. **All rays hit immediately (distance ≈ 0):**
   - Sensor might be inside a collision geometry
   - Check `bodyexclude` parameter

3. **Ray directions not rotating with sensor:**
   - Make sure you're using `site_xmat` to rotate directions
   - Check matrix multiplication order

4. **Performance is slow:**
   - Make sure using `mj_multiRay` not `mj_ray` in loop
   - Check you're not creating new arrays each step
   - Profile with `python -m cProfile script.py`

### Useful Debugging Code:

```python
# Print sensor pose
print(f"Origin: {origin}")
print(f"Rotation matrix:\n{rotation_mat}")

# Print first few rays
print(f"First 3 ray directions (world):\n{ray_dirs_world[:3]}")

# Check for invalid distances
invalid = np.sum(distances < 0)
print(f"Invalid rays (< 0): {invalid}/{len(distances)}")
```

---

## Summary: What You Need to Implement

### Files to create:
1. `metaworld/sensors/lidar.py` - Main sensor class
2. `tests/sensors/test_lidar.py` - Unit tests
3. `scripts/demo_lidar.py` - Interactive demo

### Files to modify:
1. `metaworld/sensors/__init__.py` - Export new sensor
2. `metaworld/assets/objects/assets/xyz_base.xml` - Add LiDAR mount point

### Core implementation tasks:
1. Pre-compute ray directions in local frame
2. Get sensor pose from MuJoCo each step
3. Transform rays to world frame
4. Call `mj_multiRay` to cast rays
5. Post-process results (handle -1, clip values)
6. Return data in correct format

Ready to start coding? Let me know which part you want to tackle first!