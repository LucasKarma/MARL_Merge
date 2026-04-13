import gymnasium as gym
import highway_env
import numpy as np

# ========== 创建 merge 环境 ==========
env = gym.make("merge-v0")
obs, info = env.reset()

# ========== Observation ==========
print("=" * 50)
print("OBSERVATION")
print("=" * 50)
print(f"Shape: {obs.shape}")
print(f"Features: [presence, x, y, vx, vy]")
print(f"Obs:\n{obs}\n")

# ========== Action ==========
print("=" * 50)
print("ACTION SPACE")
print("=" * 50)
print(f"Type: {env.action_space}")
print("0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER\n")

# ========== 跑几步看 reward ==========
print("=" * 50)
print("SAMPLE STEPS")
print("=" * 50)
for i in range(10):
    action = 1  # IDLE
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}: reward={reward:.4f}, terminated={terminated}")
    if terminated:
        print("  --> Episode ended (collision or passed merge zone)")
        break

# ========== 看车辆组成 ==========
print("\n" + "=" * 50)
print("VEHICLES ON ROAD")
print("=" * 50)
road = env.unwrapped.road
for i, v in enumerate(road.vehicles):
    vtype = type(v).__name__
    tag = "★ AV" if vtype == "MDPVehicle" else "  HDV"
    print(f"{tag} | Vehicle {i}: type={vtype}, "
          f"pos=({v.position[0]:.1f}, {v.position[1]:.1f}), "
          f"speed={v.speed:.1f}")

env.close()