import numpy as np
from highway_env.envs import MergeEnv
from highway_env.vehicle.controller import MDPVehicle
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, SineLane, LineType
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.behavior import IDMVehicle
from vehicle_pidm import PIDMVehicle


# =====================================================================
# 三档难度配置
# =====================================================================
# road_ends: [汇合前, 收敛段, 平行合流段, 合流后]
# vehicles:  (lane_id, longitudinal, speed, role)

DIFFICULTY_CONFIGS = {
    "easy": {
        "controlled_vehicles": 2, # AV
        "vehicles_count": 5, # self + others(4)
        "road_ends": [150, 80, 80, 150],       # 默认道路，总长 460m
        "vehicles": [
            # --- 匝道 ---
            (("j", "k", 0),   5, 25, "av"), # 5：距起点 5m 处，初始速度 25m/s
            (("j", "k", 0),  30, 24, "hdv"),
            # --- 主道 ---
            (("a", "b", 0),  60, 30, "av"),
            (("a", "b", 0),  90, 29, "hdv"),
            (("a", "b", 0),  30, 31, "hdv"),
        ],
    },
    "medium": {
        "controlled_vehicles": 3, 
        "vehicles_count": 8,
        "road_ends": [200, 100, 100, 200],      # 总长 600m, 匝道 200m, 合流区 100m
        "vehicles": [
            # --- 匝道 AV ×2 ---
            (("j", "k", 0),  10, 25, "av"),
            (("j", "k", 0),  55, 23, "av"),
            # --- 主道 AV ×1 ---
            (("a", "b", 0),  80, 30, "av"),
            # --- 匝道 HDV ×1 ---
            (("j", "k", 0), 100, 22, "hdv"),
            # --- 主道 HDV ×3 ---
            (("a", "b", 0),  40, 31, "hdv"),
            (("a", "b", 0), 120, 29, "hdv"),
            (("a", "b", 0), 170, 28, "hdv"),
        ],
    },
    "hard": {
        "controlled_vehicles": 6,
        "vehicles_count": 11,
        "road_ends": [300, 120, 150, 200],      # 总长 770m, 匝道 300m, 合流区 150m
        "vehicles": [
            # --- 匝道 AV ×4（间距 ~50m）---
            (("j", "k", 0),  10, 25, "av"),
            (("j", "k", 0),  60, 24, "av"),
            (("j", "k", 0), 110, 23, "av"),
            (("j", "k", 0), 160, 22, "av"),
            # --- 主道 AV ×2 ---
            (("a", "b", 0),  80, 30, "av"),
            (("a", "b", 0), 200, 29, "av"),
            # --- 匝道 HDV ×2 ---
            (("j", "k", 0), 210, 21, "hdv"),
            (("j", "k", 0), 260, 20, "hdv"),
            # --- 主道 HDV ×3（间距 ~60m）---
            (("a", "b", 0),  40, 31, "hdv"),
            (("a", "b", 0), 140, 28, "hdv"),
            (("a", "b", 0), 270, 27, "hdv"),
        ],
    },
}


# =====================================================================
# HDV 合作程度配置
# =====================================================================

COOPERATION_PARAMS = {
    "cooperative": {
        "COMFORT_ACC_MAX": 2.0, # 最大舒适加速度，2.0相对温和
        "DISTANCE_WANTED": 7.0, # 期望跟车距离，7.0m相对宽松
        "TIME_WANTED": 2.0, # 期望跟车时距，2.0s意味礼让
    },
    "non_cooperative": {
        "COMFORT_ACC_MAX": 5.0,
        "DISTANCE_WANTED": 3.0,
        "TIME_WANTED": 0.8,
    },
}


def _sample_hdv_params(cooperation_level: str, rng: np.random.RandomState) -> dict:
    """根据合作程度返回一辆 HDV 的 IDM 参数。"""
    if cooperation_level == "mixed":
        key = rng.choice(["cooperative", "non_cooperative"])
    else:
        key = cooperation_level
    return COOPERATION_PARAMS[key].copy() # 每辆车单独一份字典对象，修改时不会互相影响


class MultiAgentMergeEnv(MergeEnv):
    """多智能体 merge 环境：支持三档难度 + 三档 HDV 合作程度 + 可变道路几何"""

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "difficulty": "easy",
            "cooperation_level": "cooperative",
            "use_pidm": True,
            "controlled_vehicles": 2,
            "action": {
                "type": "MultiAgentAction", # 多智能体场景，每一步接受的是多辆 AV 的动作元组，而不是单个动作
                "action_config": {"type": "DiscreteMetaAction"}, # 每辆AV的动作是离散的5选1
            },
            "observation": {
                "type": "MultiAgentObservation", # 多辆 AV 各自的 obs，而不是单个 obs
                "observation_config": {
                    "type": "Kinematics", # 运动学观测（位置+速度）
                    "vehicles_count": 5, # include myself
                    "features": ["presence", "x", "y", "vx", "vy"],
                },
            },
        })
        return config

    def _apply_difficulty(self):
        """根据 difficulty 设置 controlled_vehicles 和 vehicles_count"""
        diff = self.config["difficulty"]
        if diff not in DIFFICULTY_CONFIGS:
            raise ValueError(f"Unknown difficulty: {diff}. Choose from {list(DIFFICULTY_CONFIGS.keys())}")
        dc = DIFFICULTY_CONFIGS[diff]
        self.config["controlled_vehicles"] = dc["controlled_vehicles"]
        self.config["observation"]["observation_config"]["vehicles_count"] = dc["vehicles_count"]

    def reset(self, **kwargs):
        """在 reset 前先根据 difficulty 更新 config"""
        self._apply_difficulty() # 更换难度等级
        return super().reset(**kwargs)

    def _make_road(self) -> None:
        """根据 difficulty 配置生成不同长度的道路。"""
        diff = self.config.get("difficulty", "easy")
        ends = DIFFICULTY_CONFIGS[diff]["road_ends"] # 比如拿到 easy 是 [150, 80, 80, 150]

        net = RoadNetwork()

        # === 主道（双车道）===
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE # c - 实线；s - 虚线；n - 无线
        y = [0, StraightLane.DEFAULT_WIDTH] # 定义两个车道的中心线
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]] # 车道1的右线从实线变成了虚线（c → s），因为匝道车辆要从右侧并入，右边不能是实线。

        # 给每条车道各添加三段
        for i in range(2):
            net.add_lane("a", "b", StraightLane(
                [0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]
            ))
            net.add_lane("b", "c", StraightLane(
                [sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]
            ))
            net.add_lane("c", "d", StraightLane(
                [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]
            ))

        # === 匝道 ===
        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], # 两个路宽 + 额外的分隔距离（路肩+间距）
            line_types=[c, c], forbidden=True # forbidden = True 意味着这条道AV不可以停留，必须离开
        )
        # 使用正弦曲线，因为曲率变换连续，稳定，符合标准路况几何设计。直线连接会产生突变转角，车辆到转折点时方向盘需要瞬间打死，物理上不合理。
        lkb = SineLane( 
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc) # 和主道bc共享，通过forbidden区分

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0))) # 放一个障碍物。AV 必须在撞上这个障碍物之前完成变道并入主道。
        self.road = road

    def _make_vehicles(self):
        """根据 difficulty 配置生成所有车辆"""
        road = self.road
        self.controlled_vehicles = [] # 清空上一局的记录

        diff = self.config["difficulty"]
        dc = DIFFICULTY_CONFIGS[diff]
        cooperation = self.config["cooperation_level"]
        use_pidm = self.config["use_pidm"]

        rng = np.random.RandomState() # 用于后面给每辆 HDV 随机抽取参数，和环境全局的随机数分开，互不干扰。
        VehicleClass = PIDMVehicle if use_pidm else IDMVehicle

        for lane_id, longi, spd, role in dc["vehicles"]:
            if role == "av":
                vehicle = MDPVehicle.make_on_lane(
                    road, lane_id, longitudinal=longi, speed=spd,
                )
                road.vehicles.append(vehicle)
                self.controlled_vehicles.append(vehicle) # 只存 AV，RL 策略用这个列表取观测和发动作
            else:
                hdv = VehicleClass.make_on_lane(
                    road, lane_id, longitudinal=longi, speed=spd,
                )
                params = _sample_hdv_params(cooperation, rng) # 返回 "COMFORT_ACC_MAX"、"DISTANCE_WANTED"、"TIME_WANTED"
                for attr, val in params.items():
                    setattr(hdv, attr, val)
                road.vehicles.append(hdv) # hdv只加入road.vehicles，不加入controlled


# ===== 验证脚本 =====
if __name__ == "__main__":
    from gymnasium.envs.registration import register

    register(
        id="merge-multi-v0",
        entry_point="env_multi_agent:MultiAgentMergeEnv",
    )

    import gymnasium as gym

    for diff in ["easy", "medium", "hard"]:
        print(f"\n{'='*60}")
        print(f"  Difficulty: {diff}")
        dc = DIFFICULTY_CONFIGS[diff]
        ends = dc["road_ends"]
        print(f"  Road: highway {sum(ends)}m, ramp straight {ends[0]}m, merge zone {ends[2]}m")
        print(f"{'='*60}")

        env = gym.make("merge-multi-v0")
        env.unwrapped.config.update({
            "difficulty": diff,
            "cooperation_level": "cooperative",
        })
        obs, info = env.reset()

        n_av = len(env.unwrapped.controlled_vehicles)
        n_total = len(env.unwrapped.road.vehicles)
        n_hdv = n_total - n_av

        print(f"  AV: {n_av}  |  HDV: {n_hdv}  |  Total: {n_total}")
        print(f"  Obs: {len(obs)} agents, each {obs[0].shape}")

        for i, v in enumerate(env.unwrapped.road.vehicles):
            vtype = type(v).__name__
            is_av = v in env.unwrapped.controlled_vehicles
            tag = "★ AV" if is_av else "  HDV"
            print(f"  {tag} | {i}: {vtype}, "
                  f"pos=({v.position[0]:.1f}, {v.position[1]:.1f}), "
                  f"spd={v.speed:.1f}")

        # 跑 10 步 smoke test：全部 IDLE 看会不会立刻碰撞
        crashes = 0
        for step in range(10):
            actions = tuple([1] * n_av)
            obs, r, term, trunc, info = env.step(actions)
            if term or trunc:
                crashes += 1
                break
        print(f"  10-step smoke test: {'CRASHED at step ' + str(step) if crashes else 'SURVIVED all 10 steps'}")

        env.close()
