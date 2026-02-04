# MetaWorld Sensors Module

This module provides the infrastructure for adding configurable sensors to MetaWorld environments.

## Architecture Overview

The sensor system is built around a simple but extensible abstraction:

```
SensorBase (abstract)
    ↓
Concrete Sensors (DepthCameraSensor, TactileSensor, etc.)
    ↓
SensorManager (orchestrates multiple sensors)
    ↓
Modified SawyerXYZEnv (integrates sensor observations)
```

## Quick Start

### 1. Creating a Sensor

All sensors inherit from `SensorBase` and implement 4 core methods:

```python
from metaworld.sensors.base import SensorBase
from gymnasium import spaces
import numpy as np

class MyCustomSensor(SensorBase):
    @property
    def name(self) -> str:
        return "my_sensor"

    def reset(self, env) -> None:
        # Initialize sensor state
        self._reading = np.zeros(10)

    def update(self, env) -> None:
        # Update sensor based on env.data (MuJoCo state)
        self._reading = np.random.randn(10)  # placeholder

    def read(self) -> np.ndarray:
        # Return current reading
        return self._reading

    def get_observation_space(self) -> spaces.Space:
        # Define the observation space
        return spaces.Box(low=-np.inf, high=np.inf, shape=(10,))
```

### 2. Using the Depth Camera Sensor

The `DepthCameraSensor` is the first concrete sensor implementation:

```python
from metaworld.sensors.visual import DepthCameraSensor

# Create sensor (camera must exist in XML)
depth_sensor = DepthCameraSensor(
    camera_name="gripper_depth",
    height=64,
    width=64,
    normalize=False  # raw depth by default
)
```

**Required**: Add camera to your environment's XML:

```xml
<worldbody>
  <camera name="gripper_depth"
          pos="0 0 0.05"      <!-- 5cm forward of gripper -->
          quat="1 0 0 0"      <!-- looking forward -->
          fovy="60"/>          <!-- 60 degree field of view -->
</worldbody>
```

## Implementation Status

### ✅ Completed
- `SensorBase` abstract class with full documentation
- `DepthCameraSensor` interface and structure
- Sensor validation and metadata system

### 🚧 In Progress (Your Tasks)
- [ ] Implement `DepthCameraSensor.update()` - render depth from MuJoCo
- [ ] Create `SensorManager` for orchestrating multiple sensors
- [ ] Integrate sensors into `SawyerXYZEnv` observation pipeline
- [ ] Add gripper camera to `reach-v3` XML for testing

### 📋 TODO (Future)
- [ ] Force/torque sensors (MuJoCo native)
- [ ] Tactile sensors (contact-based)
- [ ] Distance sensors (geometric)
- [ ] RGB camera sensor
- [ ] C++ implementation of depth sensor for performance comparison

## Testing Your Implementation

### Test 1: Sensor Creation
```python
from metaworld.sensors.visual import DepthCameraSensor

sensor = DepthCameraSensor("gripper_depth", height=64, width=64)
print(f"Sensor name: {sensor.name}")
print(f"Observation space: {sensor.get_observation_space()}")
print(f"Metadata: {sensor.get_metadata()}")
```

### Test 2: Observation Space
```python
import gymnasium as gym

# After integrating sensor into environment
env = gym.make('Meta-World/MT1', env_name='reach-v3')
print(f"Observation space: {env.observation_space}")
# Should show depth camera dimensions in observation
```

### Test 3: Depth Visualization
```python
# After implementing update()
sensor = DepthCameraSensor("gripper_depth", height=64, width=64)
env.reset()
sensor.reset(env)
sensor.update(env)

# Visualize depth as colored image
depth_rgb = sensor.depth_to_color_image()
import matplotlib.pyplot as plt
plt.imshow(depth_rgb)
plt.title("Depth Camera View (Gripper)")
plt.show()
```

## Design Principles

1. **Modularity**: Sensors are independent, composable components
2. **Minimal Invasiveness**: Extend existing MetaWorld, don't rewrite it
3. **Performance Awareness**: Expensive operations (rendering) only in update()
4. **Backward Compatibility**: Environments work without sensors
5. **Type Safety**: Full type hints for IDE support

## File Structure

```
metaworld/sensors/
├── __init__.py          # Public API exports
├── README.md            # This file
├── base.py              # SensorBase abstract class
├── visual.py            # Camera sensors (RGB, depth)
├── manager.py           # SensorManager (to be implemented)
├── tactile.py           # Tactile sensors (future)
├── force.py             # Force/torque sensors (future)
└── geometric.py         # Distance/ray-casting (future)
```

## Implementation Notes

### For DepthCameraSensor.update()

You'll need to:

1. Access the MuJoCo renderer from the environment
2. Render a depth image using the specified camera
3. Process the depth buffer (normalize, invert if configured)
4. Store in `self._depth_buffer`

**Key MuJoCo/Gymnasium APIs**:
- `env.mujoco_renderer`: The renderer object
- `env.mujoco_renderer.render(render_mode='depth_array', camera_name=...)`: Renders depth
- Returns a numpy array of shape `(height, width)` with depth values

**Gotchas**:
- MuJoCo depth is sometimes in a non-intuitive format (check if needs conversion)
- Rendered size might not match requested size (may need resize)
- Depth values might need clipping to avoid infinities

### For SensorManager (next step)

The manager should:
- Store a list of sensors
- Call `reset()` on all sensors when env resets
- Call `update()` on all sensors after each step
- Collect readings from all sensors via `read()`
- Build a dictionary observation: `{'sensor_name': reading}`
- Handle sensor failures gracefully (catch exceptions, log warnings)

## Questions?

When implementing, if you encounter issues:

1. **Camera not found**: Check the XML has `<camera name="gripper_depth" .../>`
2. **Observation space errors**: Ensure `read()` returns shape matching `get_observation_space()`
3. **Rendering issues**: Verify `env.mujoco_renderer` exists and is initialized
4. **Performance problems**: Profile `update()` - rendering is the bottleneck

Feel free to ask for help debugging any of these!