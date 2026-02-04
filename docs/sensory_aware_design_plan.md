# Sensory-Aware MetaWorld Fork — Design Plan

## 1. Purpose and Motivation

This project aims to create a **sensory-aware fork of MetaWorld** that extends the existing benchmark suite with rich, configurable sensing capabilities, while preserving the original task semantics, rewards, and benchmarking interfaces.

The fork is intended to serve as a **research-grade simulation foundation** for studying:
- contact-rich manipulation
- partial observability
- sensor-driven control
- world model learning
- sim-to-real transfer via realistic sensing

The system must support both **rapid prototyping** and **long-term extensibility**, including integration of custom low-level sensors implemented outside Python.

---

## 2. Core Design Principles

### 2.1 Capability-Based Forking
The fork extends MetaWorld by **adding capabilities**, not by modifying task logic. All original tasks remain conceptually unchanged.

### 2.2 Task–Sensor Separation
Tasks define:
- goals
- rewards
- success criteria
- resets
- action semantics

Sensors define:
- what the agent can perceive
- how perception is structured
- what information is available or withheld

No task should depend on a specific sensor configuration.

### 2.3 Minimal Upstream Divergence
The fork must be structured to:
- minimize changes to upstream MetaWorld files
- keep merge conflicts localized
- allow rebasing onto newer MetaWorld versions with minimal effort

---

## 3. Scope of Extension

The fork introduces **sensory awareness** at the environment level, without altering:
- action spaces
- reward functions
- episode termination logic
- benchmark APIs

The following areas are explicitly in scope:
- sensor definition
- sensor aggregation
- observation construction
- sensor metadata and semantics
- sensor-driven partial observability

---

## 4. Current MetaWorld Architecture Analysis

### 4.1 Observation Construction Pipeline

Currently, observations are constructed in `SawyerXYZEnv` via:

1. **`_get_curr_obs_combined_no_goal()`** (line 475-511 in `sawyer_xyz_env.py`)
   - Extracts hand position (3D)
   - Computes gripper distance (1D)
   - Collects object positions and quaternions (padded to 14D)
   - Returns: `[hand_pos(3), gripper(1), objects(14)]` = 18D vector

2. **`_get_obs()`** (line 513-527)
   - Frame-stacks current and previous observations
   - Conditionally appends goal position (3D) if `_partially_observable=False`
   - Returns: `[curr_obs(18), prev_obs(18), goal(3)]` = 39D vector

3. **`sawyer_observation_space`** (line 537-577)
   - Cached property defining Box space bounds
   - Dynamically adjusts based on `_partially_observable` flag

### 4.2 Key Integration Points

**Primary insertion points for sensor system:**

| Location | Method | Purpose | Line Reference |
|----------|--------|---------|----------------|
| Observation construction | `_get_obs()` | Append sensor data to observations | 513-527 |
| Observation space | `sawyer_observation_space` | Include sensor dimensions in space definition | 537-577 |
| Step function | `step()` | Update sensors per timestep | 580-642 |
| Reset function | `reset()` | Initialize/reset sensor state | (throughout env) |
| Task setting | `set_task()` | Configure sensors per task | 298-318 |

**Leverage points:**
- `self.data` (MuJoCo data structure) — provides access to all simulation state
- `mujoco.mj_forward()` — called before observation extraction
- `_partially_observable` flag — existing mechanism for hiding goal information

---

## 5. Sensor Abstraction Model

### 5.1 Sensor Concept

A *sensor* is any mechanism that produces perceptual data derived from the simulator state, including but not limited to:
- physical interaction signals (forces, torques, contacts)
- geometric perception (distances, depths, normals)
- contact events (onset, release, slip)
- derived or abstract percepts (grasp stability, object engagement)

Sensors may operate at different conceptual levels:
- **L0 (Raw Physical)**: Direct MuJoCo sensor outputs (force, touch, accelerometer)
- **L1 (Processed Perception)**: Computed from simulation state (tactile arrays, distance fields)
- **L2 (Event/Abstract)**: High-level percepts (contact onset detection, slip events)

### 5.2 Sensor Lifecycle

Each sensor must define:

```python
class SensorBase(abc.ABC):
    """Abstract base class for all sensors."""

    @abc.abstractmethod
    def reset(self, env: SawyerXYZEnv) -> None:
        """Reset sensor state when environment resets."""
        pass

    @abc.abstractmethod
    def update(self, env: SawyerXYZEnv) -> None:
        """Update sensor state after physics step."""
        pass

    @abc.abstractmethod
    def read(self) -> npt.NDArray[np.float64]:
        """Return current sensor reading."""
        pass

    @abc.abstractmethod
    def get_observation_space(self) -> gym.Space:
        """Return the observation space for this sensor."""
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier for this sensor."""
        pass
```

Sensors must be:
- **Composable**: Multiple sensors can be active simultaneously
- **Optional**: Can be enabled/disabled without breaking environments
- **Independently testable**: Each sensor can be validated in isolation

---

## 6. Categories of Sensors

The system must support multiple sensor categories:

### 6.1 Physical Interaction Sensors (MuJoCo Native)

These leverage MuJoCo's built-in sensor system defined in XML:

```xml
<sensor>
  <!-- Joint sensors -->
  <jointpos name="robot_joint_pos" joint="r_close"/>
  <jointvel name="robot_joint_vel" joint="r_close"/>

  <!-- Force/torque sensors -->
  <force name="gripper_force" site="gripper_site"/>
  <torque name="gripper_torque" site="gripper_site"/>

  <!-- Touch sensors -->
  <touch name="finger_touch_left" site="leftEndEffector"/>
  <touch name="finger_touch_right" site="rightEndEffector"/>

  <!-- Accelerometer/gyro -->
  <accelerometer name="hand_accel" site="hand"/>
  <gyro name="hand_gyro" site="hand"/>
</sensor>
```

**Access pattern**: `env.data.sensor('sensor_name').data`

**Advantages**:
- Fast (native C implementation)
- Physically accurate
- No additional overhead

**Limitations**:
- Must be defined in XML before simulation starts
- Cannot be dynamically enabled/disabled
- Limited to MuJoCo's sensor types

### 6.2 Tactile Perception (Python-Computed)

Simulated tactile arrays on gripper surfaces:

**Approach**:
1. Define tactile sites in MuJoCo XML (e.g., grid of sites on finger pads)
2. Query contact forces at each site via `env.data.site_xpos` and `env.data.contact`
3. Aggregate into tactile image/array

**Example structure**:
```python
class TactileSensor(SensorBase):
    def __init__(self, site_names: list[str], grid_shape: tuple[int, int]):
        self.site_names = site_names
        self.grid_shape = grid_shape
        self.tactile_data = np.zeros(grid_shape)

    def update(self, env: SawyerXYZEnv) -> None:
        # Query contact forces at each site
        for i, site_name in enumerate(self.site_names):
            contact_force = self._get_site_contact_force(env, site_name)
            grid_idx = np.unravel_index(i, self.grid_shape)
            self.tactile_data[grid_idx] = np.linalg.norm(contact_force)

    def read(self) -> npt.NDArray[np.float64]:
        return self.tactile_data.flatten()
```

### 6.3 Geometric Perception (Python-Computed)

Distance-to-object, depth sensing, ray casting:

**Approach**:
- Use `mujoco.mj_ray()` for ray-casting queries
- Compute signed distance fields using `env.data.geom_xpos`
- Derive geometric features from simulation state

**Example: Distance Sensor**:
```python
class DistanceSensor(SensorBase):
    """Measures distance from end-effector to nearest object."""

    def update(self, env: SawyerXYZEnv) -> None:
        tcp_pos = env.tcp_center
        obj_positions = [env.data.body(name).xpos for name in env._target_obj_names]
        self.distances = np.array([np.linalg.norm(tcp_pos - obj_pos)
                                   for obj_pos in obj_positions])

    def read(self) -> npt.NDArray[np.float64]:
        return self.distances
```

### 6.4 Visual Perception (Rendering-Based)

Image-based sensing using MuJoCo's offscreen rendering:

**Current support**: MetaWorld already supports `render_mode='rgb_array'` and `'depth_array'`

**Extension strategy**:
```python
class CameraSensor(SensorBase):
    def __init__(self, camera_name: str, height: int, width: int,
                 modality: Literal['rgb', 'depth', 'segmentation']):
        self.camera_name = camera_name
        self.height = height
        self.width = width
        self.modality = modality

    def read(self) -> npt.NDArray[np.float64]:
        if self.modality == 'rgb':
            return self.env.mujoco_renderer.render(
                render_mode='rgb_array',
                camera_name=self.camera_name
            )
        elif self.modality == 'depth':
            return self.env.mujoco_renderer.render(
                render_mode='depth_array',
                camera_name=self.camera_name
            )
```

**Considerations**:
- High computational cost
- May require downsampling/compression for RL
- Multiple camera viewpoints possible

### 6.5 Derived and Event-Based Sensors (Python Logic)

High-level percepts derived from multiple signals:

**Examples**:

| Sensor Type | Inputs | Output | Use Case |
|------------|--------|--------|----------|
| Contact Onset | Touch sensor history | Binary event signal | Detecting contact initiation |
| Slip Detection | Tactile + velocity | Continuous slip magnitude | Grasp stability |
| Grasp Stability | Force balance + contact area | Scalar confidence | Success prediction |
| Object Engagement | Distance + velocity | Binary flag | Task phase detection |

**Implementation pattern**:
```python
class ContactOnsetSensor(SensorBase):
    """Detects contact onset via touch sensor derivative."""

    def __init__(self, touch_sensor: TouchSensor, threshold: float = 0.1):
        self.touch_sensor = touch_sensor
        self.threshold = threshold
        self.prev_reading = 0.0
        self.onset_detected = False

    def update(self, env: SawyerXYZEnv) -> None:
        current = self.touch_sensor.read().sum()
        derivative = current - self.prev_reading
        self.onset_detected = derivative > self.threshold
        self.prev_reading = current

    def read(self) -> npt.NDArray[np.float64]:
        return np.array([float(self.onset_detected)])
```

---

## 7. Observation Composition Architecture

### 7.1 Proposed Observation Structure

```python
class SensoryObservation(TypedDict):
    """Structured observation with sensor data."""
    # Original MetaWorld observation (preserved)
    proprioceptive: npt.NDArray[np.float64]  # hand pos, gripper, objects
    goal: npt.NDArray[np.float64]  # goal position (if observable)

    # Sensor-derived components
    sensors: dict[str, npt.NDArray[np.float64]]  # keyed by sensor name

    # Metadata
    sensor_validity: dict[str, bool]  # which sensors produced valid readings
```

**Backward compatibility**:
- When no sensors enabled: return flat array (existing behavior)
- When sensors enabled: return dictionary (structured observation)

### 7.2 Observation Space Construction

The observation space must dynamically account for active sensors:

```python
def _build_observation_space(self) -> gym.Space:
    """Construct observation space including active sensors."""

    if not self.sensor_manager.has_sensors():
        # Fall back to original behavior
        return self._original_observation_space()

    # Build dictionary space
    space_dict = {
        'proprioceptive': self._proprioceptive_space(),
        'goal': self._goal_space(),
        'sensors': gym.spaces.Dict({
            sensor.name: sensor.get_observation_space()
            for sensor in self.sensor_manager.active_sensors()
        })
    }

    return gym.spaces.Dict(space_dict)
```

### 7.3 Observation Construction Pipeline (Modified)

**Proposed modifications to `SawyerXYZEnv._get_obs()`**:

```python
def _get_obs(self) -> npt.NDArray[np.float64] | dict:
    """Construct observation including sensor data."""

    # Original MetaWorld observation
    proprioceptive = self._get_curr_obs_combined_no_goal()
    goal = self._get_pos_goal() if not self._partially_observable else np.zeros(3)

    # Update and read sensors
    if hasattr(self, 'sensor_manager') and self.sensor_manager.has_sensors():
        self.sensor_manager.update(self)
        sensor_readings = self.sensor_manager.read_all()

        # Return structured observation
        return {
            'proprioceptive': np.hstack((proprioceptive, self._prev_obs, goal)),
            'goal': goal,
            'sensors': sensor_readings,
            'sensor_validity': self.sensor_manager.get_validity_flags()
        }
    else:
        # Backward compatibility: return flat array
        return np.hstack((proprioceptive, self._prev_obs, goal))
```

### 7.4 Partial Observability and Sensor-Only Observations

New configuration option: `observation_mode`

| Mode | Description | Observation Content |
|------|-------------|-------------------|
| `'full'` | Original MetaWorld | Proprioceptive + goal |
| `'goal_hidden'` | Existing partial obs | Proprioceptive only |
| `'sensor_augmented'` | Add sensors to full obs | Proprioceptive + goal + sensors |
| `'sensor_only'` | Sensors replace proprioception | Sensors only (no privileged info) |
| `'sensor_partial'` | Sensors + limited proprioception | User-configurable subset |

**Implementation**:
```python
class ObservationMode(Enum):
    FULL = 'full'
    GOAL_HIDDEN = 'goal_hidden'
    SENSOR_AUGMENTED = 'sensor_augmented'
    SENSOR_ONLY = 'sensor_only'
    SENSOR_PARTIAL = 'sensor_partial'

def _get_obs(self) -> npt.NDArray[np.float64] | dict:
    if self.observation_mode == ObservationMode.SENSOR_ONLY:
        # Return ONLY sensor readings (no privileged state)
        return self.sensor_manager.read_all()
    elif self.observation_mode == ObservationMode.SENSOR_PARTIAL:
        # Return user-configured subset
        return self._build_partial_observation()
    # ... other modes
```

---

## 8. Sensor Configuration and Experimentation

### 8.1 Declarative Sensor Configuration

Sensors should be configurable via Python API or YAML/JSON config files:

**Python API**:
```python
from metaworld.sensors import SensorConfig, TactileSensor, ForceSensor

sensor_config = SensorConfig([
    TactileSensor(site_names=['left_finger_sites'], grid_shape=(4, 4)),
    ForceSensor(site_name='gripper_force'),
    DistanceSensor(target_objects=['obj'])
])

env = gym.make('Meta-World/MT1',
               env_name='pick-place-v3',
               sensor_config=sensor_config)
```

**YAML Config**:
```yaml
sensors:
  - type: tactile
    name: left_finger_tactile
    sites: [left_finger_0, left_finger_1, ..., left_finger_15]
    grid_shape: [4, 4]

  - type: force
    name: gripper_force
    site: gripper_site

  - type: distance
    name: obj_distance
    target_objects: [obj]
```

### 8.2 Sensor Manager

Central coordinator for sensor lifecycle:

```python
class SensorManager:
    """Manages sensor lifecycle and observation aggregation."""

    def __init__(self, sensors: list[SensorBase]):
        self.sensors = {sensor.name: sensor for sensor in sensors}
        self._validity_flags = {name: True for name in self.sensors}

    def reset(self, env: SawyerXYZEnv) -> None:
        """Reset all sensors."""
        for sensor in self.sensors.values():
            try:
                sensor.reset(env)
                self._validity_flags[sensor.name] = True
            except Exception as e:
                logger.warning(f"Sensor {sensor.name} reset failed: {e}")
                self._validity_flags[sensor.name] = False

    def update(self, env: SawyerXYZEnv) -> None:
        """Update all sensors after physics step."""
        for sensor in self.sensors.values():
            if self._validity_flags[sensor.name]:
                try:
                    sensor.update(env)
                except Exception as e:
                    logger.warning(f"Sensor {sensor.name} update failed: {e}")
                    self._validity_flags[sensor.name] = False

    def read_all(self) -> dict[str, npt.NDArray[np.float64]]:
        """Read all valid sensors."""
        return {
            name: sensor.read()
            for name, sensor in self.sensors.items()
            if self._validity_flags[name]
        }

    def get_validity_flags(self) -> dict[str, bool]:
        """Return sensor validity status."""
        return self._validity_flags.copy()
```

### 8.3 Sensor Ablation and Experimentation

Support for systematic ablation studies:

```python
# Example: Compare performance with different sensor configurations
configs = [
    SensorConfig([]),  # No sensors (baseline)
    SensorConfig([TactileSensor(...)]),  # Tactile only
    SensorConfig([ForceSensor(...)]),  # Force only
    SensorConfig([TactileSensor(...), ForceSensor(...)]),  # Both
]

for i, config in enumerate(configs):
    env = gym.make('Meta-World/MT10', sensor_config=config, seed=42)
    results = train_and_evaluate(env)
    save_results(f'ablation_{i}', results)
```

---

## 9. Support for Low-Level Custom Sensors (C++ Integration)

### 9.1 Motivation

Some sensors require:
- High-frequency updates (beyond Python's performance)
- Custom physics algorithms (e.g., soft-body simulation)
- Integration with external libraries (e.g., contact mechanics solvers)
- Low-latency access to MuJoCo internals

### 9.2 Integration Strategy

**Option A: MuJoCo Custom Sensor Plugin**

MuJoCo 3.0+ supports custom sensors via plugin system:

1. Implement sensor in C++ following MuJoCo plugin API
2. Compile as shared library (`.so`/`.dll`)
3. Register in XML:
```xml
<mujoco>
  <extension>
    <plugin plugin="path/to/libcustom_sensor.so"/>
  </extension>

  <sensor>
    <plugin name="my_custom_sensor" plugin="custom_sensor" objtype="site" objname="gripper"/>
  </sensor>
</mujoco>
```
4. Access in Python via `env.data.sensor('my_custom_sensor').data`

**Advantages**:
- Native performance
- Seamless MuJoCo integration
- Can access internal MuJoCo state

**Disadvantages**:
- Requires C++ development
- More complex build/deployment
- Debugging overhead

**Option B: Python Extension Module (Cython/pybind11)**

1. Implement sensor logic in C++
2. Expose Python-callable interface via pybind11:
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

class CustomTactileSensor {
public:
    py::array_t<double> compute_tactile_response(
        py::array_t<double> contact_forces,
        py::array_t<double> contact_positions) {
        // High-performance computation here
        return result;
    }
};

PYBIND11_MODULE(custom_sensors, m) {
    py::class_<CustomTactileSensor>(m, "CustomTactileSensor")
        .def(py::init<>())
        .def("compute", &CustomTactileSensor::compute_tactile_response);
}
```

3. Wrap in Python sensor class:
```python
from custom_sensors import CustomTactileSensor as _CustomTactile

class CustomTactileSensor(SensorBase):
    def __init__(self):
        self._cpp_backend = _CustomTactile()

    def update(self, env: SawyerXYZEnv) -> None:
        forces = self._extract_contact_forces(env)
        positions = self._extract_contact_positions(env)
        self._reading = self._cpp_backend.compute(forces, positions)
```

**Advantages**:
- Flexible Python interface
- Easier debugging than plugin
- Can use external libraries

**Disadvantages**:
- Requires data marshaling (Python ↔ C++)
- Less direct MuJoCo integration

### 9.3 Recommended Approach

**Tiered strategy**:
1. **Prototype in Python**: Start with Python implementation
2. **Profile and identify bottlenecks**: Measure performance impact
3. **Selectively optimize**: Only move critical paths to C++
4. **Use pybind11 for most cases**: Easier development, sufficient performance
5. **Reserve MuJoCo plugins for extreme cases**: When direct MuJoCo access essential

### 9.4 Build System Integration

Add optional C++ extension compilation to `pyproject.toml`:

```toml
[project.optional-dependencies]
cpp_sensors = [
    "pybind11>=2.11.0",
    "cmake>=3.18"
]

[tool.scikit-build]
cmake.minimum-version = "3.18"
cmake.build-type = "Release"
```

Users can install with: `pip install -e .[cpp_sensors]`

---

## 10. Performance and Fidelity Considerations

### 10.1 Performance Targets

| Sensor Type | Target Update Rate | Acceptable Overhead |
|------------|-------------------|-------------------|
| Proprioceptive (baseline) | N/A | 0% (baseline) |
| Force/torque (MuJoCo native) | 1000 Hz | <5% |
| Tactile (Python, simple) | 100 Hz | <15% |
| Distance (ray-casting) | 100 Hz | <20% |
| Visual (RGB, 84x84) | 30 Hz | <50% |
| Visual (depth, 84x84) | 30 Hz | <30% |

### 10.2 Optimization Strategies

**Lazy evaluation**:
```python
class LazySensor(SensorBase):
    def __init__(self):
        self._cache = None
        self._cache_valid = False

    def update(self, env: SawyerXYZEnv) -> None:
        # Mark cache as invalid but don't compute yet
        self._cache_valid = False

    def read(self) -> npt.NDArray[np.float64]:
        # Compute only when actually read
        if not self._cache_valid:
            self._cache = self._compute(env)
            self._cache_valid = True
        return self._cache
```

**Sensor subsampling**:
- Not all sensors need per-step updates
- High-frequency sensors: every step
- Medium-frequency: every N steps
- Low-frequency: on-demand or every M steps

**Vectorized operations**:
- Use NumPy/JAX for batch computations
- Pre-allocate arrays
- Minimize Python loops

### 10.3 Fidelity vs Performance Trade-offs

**Configurable fidelity levels**:

```python
class SensorFidelity(Enum):
    LOW = 'low'      # Fast, simplified physics
    MEDIUM = 'medium'  # Balanced
    HIGH = 'high'    # Accurate, slow
    ULTRA = 'ultra'  # Research-grade, very slow

# Example: Tactile sensor with fidelity levels
class TactileSensor(SensorBase):
    def __init__(self, fidelity: SensorFidelity = SensorFidelity.MEDIUM):
        self.fidelity = fidelity
        if fidelity == SensorFidelity.LOW:
            self.resolution = (2, 2)  # 4 taxels
        elif fidelity == SensorFidelity.MEDIUM:
            self.resolution = (4, 4)  # 16 taxels
        elif fidelity == SensorFidelity.HIGH:
            self.resolution = (8, 8)  # 64 taxels
        elif fidelity == SensorFidelity.ULTRA:
            self.resolution = (16, 16)  # 256 taxels
```

---

## 11. Compatibility and Benchmarks

### 11.1 Backward Compatibility Requirements

**Strict guarantees**:
1. **Existing MetaWorld code runs unmodified**: All original benchmarks/tasks work
2. **Identical behavior when sensors disabled**: No performance regression
3. **Observation space unchanged** (when sensors off): Same dimensionality/bounds
4. **Determinism preserved**: Same seed → same trajectory

**Testing strategy**:
```python
# Test suite to verify backward compatibility
def test_backward_compatibility():
    """Ensure original MetaWorld behavior preserved."""

    # Create environment without sensors
    env_original = gym.make('Meta-World/MT1', env_name='reach-v3', seed=42)
    env_fork = gym.make('Meta-World-Sensory/MT1', env_name='reach-v3',
                        sensor_config=None, seed=42)

    # Run identical trajectories
    obs1, _ = env_original.reset()
    obs2, _ = env_fork.reset()
    assert np.allclose(obs1, obs2), "Reset observations differ"

    for _ in range(100):
        action = env_original.action_space.sample()
        obs1, r1, term1, trunc1, info1 = env_original.step(action)
        obs2, r2, term2, trunc2, info2 = env_fork.step(action)

        assert np.allclose(obs1, obs2), "Observations differ"
        assert np.isclose(r1, r2), "Rewards differ"
        assert term1 == term2 and trunc1 == trunc2, "Termination differs"
```

### 11.2 Sensory Benchmark Variants

Extend existing benchmarks with sensor configurations:

| Original Benchmark | Sensory Variant | Sensor Configuration |
|-------------------|----------------|---------------------|
| MT1 | MT1-Tactile | Tactile sensors only |
| MT10 | MT10-Force | Force/torque sensors only |
| MT50 | MT50-Multimodal | Tactile + force + distance |
| ML1 | ML1-Visual | RGB camera only |
| ML10 | ML10-Sensor-Only | No proprioception, sensors only |

**Registration**:
```python
register(
    id='Meta-World-Sensory/MT10-Tactile',
    entry_point=lambda: make_sensory_mt_envs(
        'MT10',
        sensor_config=TACTILE_CONFIG,
        observation_mode=ObservationMode.SENSOR_AUGMENTED
    )
)
```

### 11.3 Evaluation Metrics

**Standard metrics** (preserved):
- Success rate
- Episode return
- Convergence speed

**New sensory-specific metrics**:
- Sensor utilization (which sensors contributed to policy decisions)
- Observation entropy (information content)
- Sensor-only performance (ablation: proprioception off)

---

## 12. Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)
**Goal**: Establish sensor abstraction and integration points

- [ ] Create `metaworld/sensors/` module structure
- [ ] Implement `SensorBase` abstract class
- [ ] Implement `SensorManager` for lifecycle coordination
- [ ] Modify `SawyerXYZEnv._get_obs()` for sensor integration
- [ ] Add `observation_mode` configuration option
- [ ] Write backward compatibility test suite

**Deliverable**: Fork can load sensors but no sensors implemented yet

### Phase 2: Native MuJoCo Sensors (Weeks 3-4)
**Goal**: Leverage MuJoCo's built-in sensor system

- [ ] Add force/torque sensor XML definitions to task files
- [ ] Implement `MuJoCoNativeSensor` wrapper class
- [ ] Test on subset of environments (reach, push, pick-place)
- [ ] Benchmark performance overhead
- [ ] Document XML sensor configuration

**Deliverable**: Force/torque sensors working on all tasks

### Phase 3: Python-Computed Sensors (Weeks 5-7)
**Goal**: Implement computationally-derived sensors

- [ ] Implement `TactileSensor` (contact force aggregation)
- [ ] Implement `DistanceSensor` (ray-casting/geometric)
- [ ] Implement `ContactOnsetSensor` (event-based)
- [ ] Add configurable fidelity levels
- [ ] Optimize for performance (vectorization, caching)

**Deliverable**: Core Python sensor suite functional

### Phase 4: Visual Sensors (Weeks 8-9)
**Goal**: Integrate camera-based perception

- [ ] Implement `CameraSensor` (RGB, depth, segmentation)
- [ ] Add multi-camera support
- [ ] Test on reaching and manipulation tasks
- [ ] Profile rendering overhead
- [ ] Add image preprocessing utilities (downsampling, normalization)

**Deliverable**: Vision-based observation mode available

### Phase 5: C++ Integration Support (Weeks 10-12)
**Goal**: Enable high-performance custom sensors

- [ ] Set up pybind11 build system
- [ ] Implement example C++ sensor (high-res tactile)
- [ ] Document C++ sensor development workflow
- [ ] Add CMake build configuration
- [ ] Test cross-platform compilation (Linux, macOS, Windows)

**Deliverable**: Template for custom C++ sensors

### Phase 6: Configuration and Experimentation (Weeks 13-14)
**Goal**: Streamline sensor configuration for research

- [ ] Implement YAML/JSON config file parsing
- [ ] Create predefined sensor configurations (tactile-only, multimodal, etc.)
- [ ] Add sensor ablation utilities
- [ ] Create example notebooks/scripts
- [ ] Document best practices

**Deliverable**: User-friendly configuration system

### Phase 7: Benchmarking and Evaluation (Weeks 15-16)
**Goal**: Establish sensory benchmark suite

- [ ] Create sensory variants of MT1/MT10/MT50
- [ ] Create sensory variants of ML1/ML10/ML45
- [ ] Run baseline experiments (PPO, SAC)
- [ ] Generate learning curves for sensor ablations
- [ ] Write evaluation utilities

**Deliverable**: Sensory benchmark suite with baselines

### Phase 8: Documentation and Release (Weeks 17-18)
**Goal**: Prepare for public release

- [ ] Write comprehensive documentation (usage, API, examples)
- [ ] Create tutorial notebooks
- [ ] Record demo videos
- [ ] Set up CI/CD for testing
- [ ] Prepare academic paper/tech report
- [ ] Release v1.0

**Deliverable**: Public release-ready fork

---

## 13. File Structure

Proposed directory organization:

```
metaworld/
├── sensors/                    # New: Sensor system
│   ├── __init__.py
│   ├── base.py                # SensorBase, SensorManager
│   ├── native.py              # MuJoCo native sensor wrappers
│   ├── tactile.py             # Tactile sensors
│   ├── geometric.py           # Distance, ray-casting sensors
│   ├── visual.py              # Camera sensors
│   ├── derived.py             # Event-based, abstract sensors
│   ├── config.py              # SensorConfig, YAML parsing
│   └── cpp/                   # C++ sensor implementations
│       ├── tactile_hd.cpp
│       ├── CMakeLists.txt
│       └── bindings.cpp
│
├── sensory_env.py             # Modified: SensorySawyerXYZEnv
├── sawyer_xyz_env.py          # Modified: Add sensor hooks
├── envs/                      # Modified: Add sensor XMLs
│   └── assets/
│       └── sensors/           # New: Sensor-specific XML files
│           ├── force_sensors.xml
│           ├── tactile_sites.xml
│           └── cameras.xml
│
├── benchmarks/                # New: Sensory benchmark definitions
│   ├── mt_tactile.py
│   ├── mt_multimodal.py
│   └── ml_sensor_only.py
│
└── tests/
    └── sensors/               # New: Sensor tests
        ├── test_base.py
        ├── test_tactile.py
        ├── test_compatibility.py
        └── test_performance.py
```

---

## 14. Intended Use Cases

This fork is designed to support research such as:

### 14.1 Contact-Aware Manipulation Learning
- **Problem**: Learning dexterous manipulation requires understanding contact dynamics
- **Solution**: Tactile + force sensors provide rich contact information
- **Experiments**: Compare proprioceptive-only vs tactile-augmented policies on assembly tasks

### 14.2 Tactile-Driven Control
- **Problem**: Vision-based control fails in occlusion/poor lighting
- **Solution**: Touch-based perception enables blind manipulation
- **Experiments**: Sensor-only observation mode on in-hand manipulation tasks

### 14.3 Sensor-Based World Models
- **Problem**: Learning predictive models of contact dynamics
- **Solution**: Sensory observations as prediction targets
- **Experiments**: Train world model on sensor data, evaluate sim-to-real transfer

### 14.4 Multimodal Perception–Action Learning
- **Problem**: Humans use vision, touch, proprioception together
- **Solution**: Multimodal sensor fusion
- **Experiments**: Ablation study comparing unimodal vs multimodal policies

### 14.5 Sim-to-Real Transfer via Realistic Sensing
- **Problem**: Reality gap in sensing modalities
- **Solution**: Match simulation sensors to real robot hardware
- **Experiments**: Train in sensory-aware sim, deploy on real robot with matched sensors

---

## 15. Non-Goals

The following are explicitly **out of scope**:

### 15.1 Not Modifying Task Semantics
- ❌ Changing reward functions based on sensors
- ❌ Introducing sensor-dependent success criteria
- ❌ Task-specific sensor heuristics
- ✅ Sensors are purely observational

### 15.2 Not Redefining Action Spaces
- ❌ Haptic feedback in actions
- ❌ Sensor-based action primitives
- ✅ Action space remains XYZ + gripper

### 15.3 Not Optimizing for Specific Algorithms
- ❌ Sensor observations pre-formatted for specific RL algorithms
- ❌ Built-in sensor processing pipelines (e.g., CNNs)
- ✅ Raw sensor data; downstream processing is user's responsibility

### 15.4 Not Replacing Simulation Physics
- ❌ Custom contact dynamics
- ❌ Soft-body simulation
- ✅ MuJoCo physics unchanged; sensors observe existing physics

---

## 16. Success Criteria

The project is considered successful if:

### 16.1 Functional Requirements
- [ ] All original MetaWorld tasks work identically when sensors disabled
- [ ] Sensors can be added/removed without touching task files
- [ ] At least 3 sensor categories implemented (tactile, force, visual)
- [ ] C++ sensor integration pathway functional
- [ ] Observation modes (full, sensor-only, etc.) working

### 16.2 Performance Requirements
- [ ] <10% overhead for force/torque sensors
- [ ] <20% overhead for tactile sensors
- [ ] <50% overhead for visual sensors (84x84)
- [ ] Deterministic behavior maintained

### 16.3 Usability Requirements
- [ ] Sensors configurable via Python API and YAML
- [ ] Comprehensive documentation with examples
- [ ] Backward compatibility test suite passing
- [ ] Tutorial notebooks for common use cases

### 16.4 Research Enablement
- [ ] Sensory benchmark variants defined
- [ ] Baseline experiments conducted (PPO/SAC)
- [ ] Ablation study utilities available
- [ ] Example research paper experiments reproducible

---

## 17. Long-Term Vision

The long-term goal is to establish a **sensory-centric manipulation benchmark layer** built on MetaWorld, enabling systematic study of perception, contact, and uncertainty in robotic learning systems.

### 17.1 Evolution Path

**Year 1**: Core sensor infrastructure, basic tactile/force sensors

**Year 2**: Advanced sensors (optical tactile, slip detection), sim-to-real validation

**Year 3**: Multi-robot support (Franka, UR5), real-world sensor datasets

**Year 4**: Standardized sensory RL benchmark accepted by community

### 17.2 Community Engagement

- Open-source development (MIT license, matching MetaWorld)
- Integration with popular RL libraries (Stable-Baselines3, RLlib)
- Workshops at robotics/ML conferences
- Benchmarking competitions

### 17.3 Academic Impact

Target venues:
- **CoRL** (Conference on Robot Learning): Technical contribution
- **ICRA** (Robotics: Science and Systems): Benchmark paper
- **NeurIPS Datasets & Benchmarks**: Benchmark track
- **JMLR** (Journal of Machine Learning Research): Comprehensive benchmark description

---

## 18. Risk Mitigation

### 18.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Performance degradation | Medium | High | Extensive profiling, optimization, lazy evaluation |
| MuJoCo compatibility issues | Low | High | Pin MuJoCo version, test across versions |
| C++ integration complexity | High | Medium | Make C++ optional, provide Python fallbacks |
| Upstream MetaWorld divergence | Medium | Medium | Minimize core changes, modular design |

### 18.2 Research Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Limited adoption | Medium | High | Strong documentation, tutorials, baselines |
| Sensor configurations too complex | Low | Medium | Provide presets, sensible defaults |
| Not generalizable beyond MetaWorld | Low | Medium | Design sensor API to be library-agnostic |

---

## 19. Open Questions

### 19.1 Design Decisions Requiring User Input

1. **Observation format**: Dictionary vs flat array with metadata sidecar?
   - **Option A**: Always return dict when sensors enabled
   - **Option B**: Configurable (dict or flattened)
   - **Recommendation**: Option A (cleaner API)

2. **Sensor update frequency**: Per-step or configurable?
   - **Option A**: All sensors update every step
   - **Option B**: Per-sensor update rates
   - **Recommendation**: Option A initially, B later

3. **C++ integration priority**: MuJoCo plugin or pybind11 first?
   - **Option A**: Start with pybind11 (easier)
   - **Option B**: MuJoCo plugin (more performant)
   - **Recommendation**: Option A (development speed)

4. **Sensor failure handling**: Exception or zero-output?
   - **Option A**: Raise exception on sensor failure
   - **Option B**: Return zeros, set validity flag
   - **Recommendation**: Option B (robust training)

### 19.2 Future Extensions

- **Multi-agent support**: Multiple robots with independent sensors?
- **Sensor noise models**: Realistic noise injection for sim-to-real?
- **Temporal sensor processing**: Built-in filtering/smoothing?
- **Sensor attention mechanisms**: Which sensors are policy attending to?

---

## 20. Conclusion

This design plan outlines a comprehensive approach to extending MetaWorld with rich sensory awareness while maintaining strict backward compatibility and minimal upstream divergence. The proposed architecture enables:

- **Flexibility**: Sensors are optional, composable, configurable
- **Performance**: Tiered fidelity, lazy evaluation, C++ integration for critical paths
- **Research-grade quality**: Deterministic, reproducible, well-documented
- **Long-term sustainability**: Modular design, minimal core changes, extensibility

The implementation roadmap provides a structured 18-week path to a public release-ready sensory-aware MetaWorld fork, positioning it as a foundational platform for future research in sensor-driven robotic manipulation learning.

---

## Appendix A: Key Metrics and Targets

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Backward compatibility | 100% test pass | Automated test suite |
| Performance overhead (force) | <5% | Benchmark vs baseline |
| Performance overhead (tactile) | <15% | Benchmark vs baseline |
| Performance overhead (visual) | <50% | Benchmark vs baseline |
| Sensor update latency | <1ms (non-visual) | Profiling |
| Observation space validity | 100% Gym-compliant | Gym checkers |
| Documentation coverage | >90% | Docstring linting |
| Test coverage | >80% | Pytest-cov |

---

## Appendix B: References and Prior Art

### Tactile Simulation in RL
- **TACTO** (ICRA 2021): Fast tactile sensor simulation using depth rendering
- **Taxim** (RSS 2022): Differentiable tactile simulation
- **PyTouch** (CoRL 2021): Realistic tactile sensor models

### Sensory RL Benchmarks
- **DeepMind Control Suite**: Vision-based control
- **IKEA Furniture Assembly**: Contact-rich manipulation
- **MetaWorld (original)**: Proprioceptive multi-task RL

### C++ Integration in Python RL
- **MuJoCo Python bindings**: Reference implementation
- **PyBullet**: Mixed Python/C++ architecture
- **Isaac Gym**: GPU-accelerated simulation with Python API