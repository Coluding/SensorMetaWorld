import gymnasium as gym
from metaworld.sensors.visual import DepthCameraSensor
import numpy as np

env = gym.make('Meta-World/MT1', env_name='reach-v3', camera_id="corner3", render_mode="rgb_array")
sensor = DepthCameraSensor('gripper_depth_cam', 64, 64)

env.reset()
sensor.reset(env)

# Get initial depth
sensor.update(env)
depth1 = sensor.read().copy()

# Move gripper
for _ in range(100):
  action = [0.1, 0, 0, 0]  # Move forward in X
  x = env.step(action)
  env.render()
# Get new depth - should be DIFFERENT (camera moved)
sensor.update(env)
depth2 = sensor.read()

print(f"Depth changed: {not np.allclose(depth1, depth2)}")  # Should be True!


import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, action in enumerate([[0., 0, -1, 0], [0, 0.0, -1, 0], [0, 0, -1.0, 0]]):
  env.reset()
  sensor.reset(env)

  # Move gripper
  for _ in range(200):
      env.step(action)

  # Render depth
  sensor.update(env)
  depth_img = sensor.get_depth_as_image()

  axes[i].imshow(depth_img, cmap='viridis')
  axes[i].set_title(f"Action: {action[:3]}")
  axes[i].axis('off')

plt.tight_layout()
plt.show()