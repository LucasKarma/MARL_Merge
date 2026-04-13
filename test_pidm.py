"""
验证 P-IDM 与静态 IDM 的行为差异。
场景：AV 在隔壁车道，target_lane_index 指向 HDV 所在车道，且 AV 在 HDV 前方。
预期：P-IDM 的 HDV 会减速（因为看到了切入的 AV），静态 IDM 不会。
"""
import gymnasium as gym
import numpy as np
from env_multi_agent import MultiAgentMergeEnv
from vehicle_pidm import PIDMVehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle
from gymnasium.envs.registration import register

# 注册环境
register(id="merge-multi-v0", entry_point="env_multi_agent:MultiAgentMergeEnv")

env = gym.make("merge-multi-v0")
obs, info = env.reset()

road = env.unwrapped.road

# ===== 找到一辆主道 HDV 和一辆 AV =====
hdv = None
av = None
for v in road.vehicles:
    if isinstance(v, IDMVehicle) and not isinstance(v, MDPVehicle) and hdv is None:
        hdv = v
    if isinstance(v, MDPVehicle) and av is None:
        av = v

print(f"AV:  pos=({av.position[0]:.1f}, {av.position[1]:.1f}), "
      f"lane={av.lane_index}, target_lane={av.target_lane_index}")
print(f"HDV: pos=({hdv.position[0]:.1f}, {hdv.position[1]:.1f}), "
      f"lane={hdv.lane_index}")

# ===== 手动构造切入场景 =====
# 确保 AV 和 HDV 不在同一条车道
# 先找到一条与 HDV 不同的车道
hdv_lane = hdv.lane_index
available_lanes = [v.lane_index for v in road.vehicles if v.lane_index != hdv_lane]
if available_lanes:
    different_lane = available_lanes[0]
else:
    # 手动指定一条不同的车道
    different_lane = (hdv_lane[0], hdv_lane[1], 1 - hdv_lane[2])

# 把 AV 放到不同车道上，纵向位于 HDV 前方 15m
av.lane_index = different_lane
av.position = hdv.position.copy()
av.position[0] += 15       # AV 在 HDV 前方 15m
av.position[1] += 4.0      # 横向偏移到另一条车道
av.target_lane_index = hdv_lane   # AV 的目标是切入 HDV 的车道

# ===== 对比 1：静态 IDM（原版 leader 查找）=====
front_static, _ = road.neighbour_vehicles(hdv, hdv.lane_index)
acc_static = hdv.acceleration(ego_vehicle=hdv, front_vehicle=front_static, rear_vehicle=None)
print(f"\n[静态 IDM]")
print(f"  leader: {type(front_static).__name__ if front_static else 'None'}, "
      f"pos=({front_static.position[0]:.1f})" if front_static else "  leader: None")
print(f"  acceleration: {acc_static:.3f}")

# ===== 对比 2：P-IDM（加上切入检测）=====
# 临时给 HDV 挂上 P-IDM 的方法来测试
pidm_finder = PIDMVehicle._find_cutting_in_av
cutting_av = pidm_finder(hdv)

if cutting_av is not None:
    # 跟原始 leader 比较，取更近的
    if front_static is None:
        effective_leader = cutting_av
    else:
        dist_original = front_static.position[0] - hdv.position[0]
        dist_cutting = cutting_av.position[0] - hdv.position[0]
        effective_leader = cutting_av if dist_cutting < dist_original else front_static
else:
    effective_leader = front_static

acc_pidm = hdv.acceleration(ego_vehicle=hdv, front_vehicle=effective_leader, rear_vehicle=None)
print(f"\n[P-IDM]")
print(f"  cutting_in AV detected: {cutting_av is not None}")
print(f"  effective leader: {type(effective_leader).__name__ if effective_leader else 'None'}, "
      f"pos=({effective_leader.position[0]:.1f})" if effective_leader else "  effective leader: None")
print(f"  acceleration: {acc_pidm:.3f}")

# ===== 结论 =====
print(f"\n{'='*50}")
print(f"加速度差异: {acc_static:.3f} (静态) vs {acc_pidm:.3f} (P-IDM)")
if abs(acc_static - acc_pidm) > 0.01:
    print("✅ P-IDM 检测到切入 AV，行为与静态 IDM 不同！")
else:
    print("⚠️  两者行为相同，需要检查切入检测逻辑")

env.close()