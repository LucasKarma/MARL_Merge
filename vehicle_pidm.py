import numpy as np
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle


class PIDMVehicle(IDMVehicle):
    """
    Interaction-aware IDM Vehicle (P-IDM).
    
    与静态 IDM 的唯一区别：在选择 leader 时，额外检查是否有 AV
    正在切入自己的车道前方。如果有，将该 AV 视为动态 leader。
    """

    def act(self, action: dict | str = None):
        if self.crashed:
            return
        action = {}

        # ===== Lateral: MOBIL（与原版完全一致）=====
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        action["steering"] = self.steering_control(self.target_lane_index)
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )

        # ===== Longitudinal: P-IDM（核心改动在这里）=====
        # 第一步：用原始方法找当前车道的 leader
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
            self, self.lane_index
        )

        # 第二步：检查是否有 AV 正在切入我的车道前方
        cutting_in_vehicle = self._find_cutting_in_av()

        # 第三步：如果有切入的 AV，跟原始 leader 比较谁更近
        if cutting_in_vehicle is not None:
            if front_vehicle is None:
                # 当前车道前方没有车，切入的 AV 直接成为 leader
                front_vehicle = cutting_in_vehicle
            else:
                # 比较纵向距离，更近的那辆当 leader
                dist_original = front_vehicle.position[0] - self.position[0]
                dist_cutting = cutting_in_vehicle.position[0] - self.position[0]
                if dist_cutting < dist_original:
                    front_vehicle = cutting_in_vehicle

        action["acceleration"] = self.acceleration(
            ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
        )

        # 变道时检查目标车道（与原版一致）
        if self.lane_index != self.target_lane_index:
            front_vehicle_t, rear_vehicle_t = self.road.neighbour_vehicles(
                self, self.target_lane_index
            )
            target_idm_acceleration = self.acceleration(
                ego_vehicle=self,
                front_vehicle=front_vehicle_t,
                rear_vehicle=rear_vehicle_t,
            )
            action["acceleration"] = min(
                action["acceleration"], target_idm_acceleration
            )

        action["acceleration"] = np.clip(
            action["acceleration"], -self.ACC_MAX, self.ACC_MAX
        )
        from highway_env.vehicle.kinematics import Vehicle
        Vehicle.act(self, action)

    def _find_cutting_in_av(self):
        """
        检查场景中是否有 AV 正在切入自己的车道前方。
        
        三个判定条件：
        1. 是 MDPVehicle（AV）
        2. AV 的 target_lane_index == 我的 lane_index（它想来我的车道）
        3. AV 在我前方（纵向位置 > 我的位置）
        
        如果有多辆符合条件，返回最近的那辆。
        """
        my_lane = self.lane_index
        my_x = self.position[0]
        
        closest_av = None
        closest_dist = float("inf")

        for v in self.road.vehicles:
            # 条件 1：是 AV
            if not isinstance(v, MDPVehicle):
                continue
            # 条件 2：AV 的目标车道是我的车道
            if v.target_lane_index != my_lane:
                continue
            # 额外：AV 当前不在我的车道（还没切过来才需要特殊处理）
            if v.lane_index == my_lane:
                continue
            # 条件 3：AV 在我前方
            dist = v.position[0] - my_x
            if dist <= 0:
                continue
            # 取最近的
            if dist < closest_dist:
                closest_dist = dist
                closest_av = v

        return closest_av