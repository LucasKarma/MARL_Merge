"""
Priority-based Safety Supervisor

核心职责：在 RL policy 选出动作后、环境执行前，检查每辆 AV 的动作
是否会在未来 T_n 步内导致碰撞。如果会，替换为最安全的动作。

流程：
1. 按优先级对 AV 排序
2. 对每辆 AV，预测其候选动作的未来轨迹
3. 预测周围车辆的未来轨迹
4. 检查是否碰撞
5. 碰撞 → 从合法动作中选安全边距最大的替换
"""
import numpy as np
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle


class SafetySupervisor:
    """Priority-based Safety Supervisor for multi-agent merge."""

    # 论文参数
    T_N = 6              # 预测步数
    DT = 0.2             # 每步时长（对应 5Hz 控制频率）
    SAFE_DISTANCE = 5.0  # 碰撞判定距离阈值 [m]

    # 优先级权重（来自 MARL 论文 Eq.10）
    ALPHA_MERGE = 1.0    # 匝道车优先
    ALPHA_DIST = 1.0     # 越靠近匝道末端越优先
    ALPHA_HEADWAY = 1.0  # 车头时距越小越优先

    # 匝道相关参数
    RAMP_LENGTH = 100.0  # 匝道总长度

    def __init__(self, env):
        self.env = env

    def safe_actions(self, actions):
        """
        输入：RL policy 给每辆 AV 选的动作 tuple，如 (3, 1)
        输出：安全检查后的动作 tuple，危险动作被替换
        """
        road = self.env.unwrapped.road
        controlled = self.env.unwrapped.controlled_vehicles
        actions = list(actions)

        # 第一步：按优先级排序
        priorities = [self._priority(av) for av in controlled] # 给每一个AV计算优先级
        sorted_indices = np.argsort(priorities)[::-1]  # 返回排序下标，高优先级在前

        # 存储每辆 AV 确认后的安全轨迹，用于 第i+1辆AV 与 第i辆AV 进行碰撞检测
        confirmed_trajectories = {}

        for idx in sorted_indices:
            av = controlled[idx]
            original_action = actions[idx]

            # 预测该动作下 AV 的未来轨迹
            av_traj = self._predict_av_trajectory(av, original_action)

            # 收集需要检查的其他车辆轨迹
            other_trajs = self._get_other_trajectories(
                av, road, confirmed_trajectories
            )

            # 检查是否碰撞
            if not self._check_collision(av_traj, other_trajs):
                # 安全，确认这个动作
                confirmed_trajectories[id(av)] = av_traj
            else:
                # 不安全，找最安全的替代动作
                best_action = original_action # 保底，假如5个候选都危险
                best_margin = -float("inf")
                best_traj = av_traj # 同样保底，先把最佳轨迹初始化为原始动作对应的轨迹。

                for candidate_action in range(5):  # 0-4 五个动作（LANE_LEFT / IDLE / LANE_RIGHT / FASTER / SLOWER）
                    cand_traj = self._predict_av_trajectory(
                        av, candidate_action
                    )
                    margin = self._safety_margin(cand_traj, other_trajs)
                    if margin > best_margin:
                        best_margin = margin
                        best_action = candidate_action
                        best_traj = cand_traj

                actions[idx] = best_action
                confirmed_trajectories[id(av)] = best_traj # 把替换后动作对应的轨迹存入字典，供后续 AV 审查时使用。

        return tuple(actions)

    def _priority(self, av):
        """
        计算 AV 的优先级分数（越高越先处理）。
        论文公式：p = α1·p_merge + α2·p_dist + α3·p_headway + noise
        """
        score = 0.0

        # p_merge：匝道上的车优先级更高
        lane = av.lane_index
        is_on_ramp = lane[0] == "j" or lane[0] == "k"
        if is_on_ramp:
            score += self.ALPHA_MERGE * 0.5

        # p_dist：越靠近匝道末端（纵向位置越大）越优先
        if is_on_ramp:
            x = av.position[0]
            score += self.ALPHA_DIST * (x / self.RAMP_LENGTH)

        # p_headway：车头时距越小越优先
        front, _ = self.env.unwrapped.road.neighbour_vehicles(
            av, av.lane_index
        )
        if front is not None and av.speed > 0:
            headway_dist = front.position[0] - av.position[0]
            time_headway = headway_dist / av.speed # 到前车的时间距离
            score += self.ALPHA_HEADWAY * max(
                0, -np.log(max(time_headway / 1.2, 0.01))
            ) # 用 -log 使得小 headway → 高分数

        # 加一点噪声打破平局
        score += np.random.normal(0, 0.01)

        return score

    def _predict_av_trajectory(self, av, action):
        """
        预测 AV 在给定动作下的未来 T_n 步位置。
        使用简单运动学：pos += speed * dt, speed += acc * dt
        """
        # 动作映射到加速度（highway-env DiscreteMetaAction 的默认设置）
        ACTION_TO_ACC = {
            0: 0.0,    # LANE_LEFT（横向，纵向不变）
            1: 0.0,    # IDLE
            2: 0.0,    # LANE_RIGHT（横向，纵向不变）
            3: 4.0,    # FASTER
            4: -6.0,   # SLOWER
        }
        acc = ACTION_TO_ACC.get(action, 0.0)

        trajectory = []
        x = av.position[0]
        speed = av.speed
        for _ in range(self.T_N):
            speed = max(0, speed + acc * self.DT)
            x = x + speed * self.DT
            trajectory.append(x)

        return trajectory # 返回比如[232.1, 234.3, 236.4, 238.4, 240.3, 242.1]，每个元素是对应时间步的预测 x 坐标。

    def _get_other_trajectories(self, av, road, confirmed_trajectories):
        """
        收集需要检查碰撞的其他车辆轨迹。
        - 已确认的 AV：使用 confirmed_trajectories 里的安全轨迹
        - HDV：使用 IDM 加速度预测
        - 未确认的其他 AV：使用当前状态匀速预测
        """
        trajs = []
        for v in road.vehicles:
            if v is av:
                continue

            if id(v) in confirmed_trajectories:
                # 已确认安全轨迹的 AV
                trajs.append(confirmed_trajectories[id(v)])
            elif isinstance(v, IDMVehicle):
                # HDV：用 IDM 公式预测
                trajs.append(self._predict_hdv_trajectory(v, road))
            else:
                # 其他未确认的 AV：匀速预测
                trajs.append(self._predict_constant_speed(v))

        return trajs

    def _predict_hdv_trajectory(self, hdv, road):
        """预测 HDV 未来 T_n 步位置，使用 IDM 加速度。"""
        front, rear = road.neighbour_vehicles(hdv, hdv.lane_index)
        acc = hdv.acceleration(
            ego_vehicle=hdv, front_vehicle=front, rear_vehicle=rear
        ) # AV 的 acc 来自动作映射表，HDV 的 acc 来自 IDM 公式

        trajectory = []
        x = hdv.position[0]
        speed = hdv.speed
        for _ in range(self.T_N):
            speed = max(0, speed + acc * self.DT)
            x = x + speed * self.DT
            trajectory.append(x)

        return trajectory

    def _predict_constant_speed(self, vehicle):
        """匀速预测：最简单的 fallback。用在未确认的其他 AV 上——它们的动作还没决定，无法预测加速度，用匀速是最保守的合理假设。"""
        trajectory = []
        x = vehicle.position[0]
        for step in range(self.T_N):
            x = x + vehicle.speed * self.DT
            trajectory.append(x)
        return trajectory

    def _check_collision(self, traj, other_trajs):
        """检查轨迹在任意时间步是否与其他车辆轨迹距离过近。"""
        for other in other_trajs:
            for t in range(self.T_N):
                if abs(traj[t] - other[t]) < self.SAFE_DISTANCE:
                    return True  # 碰撞
        return False

    def _safety_margin(self, traj, other_trajs):
        """计算轨迹与所有其他车辆轨迹的最小距离。"""
        min_dist = float("inf")
        for other in other_trajs:
            for t in range(self.T_N):
                dist = abs(traj[t] - other[t])
                min_dist = min(min_dist, dist)
        return min_dist
