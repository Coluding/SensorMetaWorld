import gymnasium as gym
import numpy as np
import metaworld

env = gym.make('Meta-World/MT1', env_name='push-v3')
env.reset()

# Print all geometry names
print("All geometries:")
for i in range(env.unwrapped.model.ngeom):
    print(f"  {i}: {env.unwrapped.model.geom(i).name}")

# Find finger pad IDs
left_id = env.unwrapped.model.geom("leftpad_geom").id
right_id = env.unwrapped.model.geom("rightpad_geom").id
print(f"\nFinger IDs: left={left_id}, right={right_id}")

for step in range(50):
    action = env.action_space.sample()
    env.step(action)

    if env.unwrapped.data.ncon > 0:  # If there are contacts
        print(f"\nStep {step}: {env.unwrapped.data.ncon} contacts")

        for i in range(env.unwrapped.data.ncon):
            contact = env.unwrapped.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Get the names of the geometries in contact
            name1 = env.unwrapped.model.geom(geom1).name
            name2 = env.unwrapped.model.geom(geom2).name

            print(f"  Contact {i}: geom1={geom1} ({name1}) <-> geom2={geom2} ({name2})")

            # Check if it involves fingers
            if geom1 in [left_id, right_id] or geom2 in [left_id, right_id]:
                print(f"    *** FINGER CONTACT DETECTED! ***")