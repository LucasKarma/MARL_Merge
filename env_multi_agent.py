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
        "controlled_vehicles": 2,
        "vehicles_count": 5,
        "road_ends": [150, 80, 80, 150],       # 默认道路，总长 460m
        "vehicles": [
            # --- 匝道 ---
            (("j", "k", 0),   5, 25, "av"),
            # --- 主道 ---
            (("a", "b", 0),  60, 30, "av"),
            (("a", "b", 0),  90, 29, "hdv"),
            (("a", "b", 0),  30, 31, "hdv"),
            (("j", "k", 0),  30, 24, "hdv"),
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
        "COMFORT_ACC_MAX": 2.0,
        "DISTANCE_WANTED": 7.0,
        "TIME_WANTED": 2.0,
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
    return COOPERATION_PARAMS[key].copy()


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
                "type": "MultiAgentAction",
                "action_config": {"type": "DiscreteMetaAction"},
            },
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
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
        self._apply_difficulty()
        return super().reset(**kwargs)

    def _make_road(self) -> None:
        """根据 difficulty 配置生成不同长度的道路。"""
        diff = self.config.get("difficulty", "easy")
        ends = DIFFICULTY_CONFIGS[diff]["road_ends"]

        net = RoadNetwork()

        # === 主道（双车道）===
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]

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
            [0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4],
            line_types=[c, c], forbidden=True
        )
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
        net.add_lane("b", "c", lbc)

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self):
        """根据 difficulty 配置生成所有车辆"""
        road = self.road
        self.controlled_vehicles = []

        diff = self.config["difficulty"]
        dc = DIFFICULTY_CONFIGS[diff]
        cooperation = self.config["cooperation_level"]
        use_pidm = self.config["use_pidm"]

        rng = np.random.RandomState()
        VehicleClass = PIDMVehicle if use_pidm else IDMVehicle

        for lane_id, longi, spd, role in dc["vehicles"]:
            if role == "av":
                vehicle = MDPVehicle.make_on_lane(
                    road, lane_id, longitudinal=longi, speed=spd,
                )
                road.vehicles.append(vehicle)
                self.controlled_vehicles.append(vehicle)
            else:
                hdv = VehicleClass.make_on_lane(
                    road, lane_id, longitudinal=longi, speed=spd,
                )
                params = _sample_hdv_params(cooperation, rng)
                for attr, val in params.items():
                    setattr(hdv, attr, val)
                road.vehicles.append(hdv)


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