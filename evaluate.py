"""
P3 统一评估脚本 — 完整实验矩阵（修正版 metric）
3 种方法 × 3 种 HDV 合作程度 = 9 组实验，全部在 Hard 难度下评估

Metric 定义（严格版）：
  - Collision Rate:     有任意 AV 碰撞的 episode 占比
  - Merge Success Rate: 所有匝道 AV 都成功并入主道且无碰撞的 episode 占比
  - Ramp Merge Ratio:   所有 episode 中匝道 AV 成功并入主道的个体比例
  - Avg Speed:          所有 AV 在所有 step 的速度均值
  - Time-to-Merge:      最后一辆匝道 AV 完成合流的平均 step 数
"""
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import torch
from train import QNetwork, MAX_OBS_DIM, flatten_and_pad
from safety_supervisor import SafetySupervisor


MODEL_FILES = {
    "marl":      "q_net_no_pidm.pth",
    "marl_pidm": "q_net_pidm.pth",
}


def _is_on_main_road(vehicle, road):
    """判断车辆是否已经在主道上。
    
    关键修复：匝道最后一段 ("b","c",2) 也以 "b" 开头，
    但它的 forbidden=True。主道车道的 forbidden=False。
    所以用 forbidden 属性区分，而不是看 lane_index 字符串。
    """
    lane = road.network.get_lane(vehicle.lane_index)
    return not getattr(lane, 'forbidden', False)


def _is_ramp_origin(lane_index):
    """判断车辆初始是否在匝道上"""
    return lane_index[0] == "j"


def evaluate(mode="idm", cooperation_level="cooperative",
             n_episodes=100, max_steps=50):
    """在 Hard 难度下跑 n_episodes 个 episode，记录严格版指标。"""
    use_pidm = (mode == "marl_pidm")

    try:
        register(id="merge-multi-v0", entry_point="env_multi_agent:MultiAgentMergeEnv")
    except Exception:
        pass

    env = gym.make("merge-multi-v0")
    env.unwrapped.config.update({
        "difficulty": "hard",
        "use_pidm": use_pidm,
        "cooperation_level": cooperation_level,
    })

    # 加载模型
    q_net = None
    if mode in ("marl", "marl_pidm"):
        q_net = QNetwork(obs_dim=MAX_OBS_DIM)
        model_path = MODEL_FILES[mode]
        q_net.load_state_dict(torch.load(model_path, weights_only=True))
        q_net.eval()

    # 指标容器
    collisions = 0
    all_ramp_merged_count = 0       # 所有匝道 AV 都合流成功的 episode 数
    total_ramp_avs = 0              # 所有 episode 中匝道 AV 的总数
    merged_ramp_avs = 0             # 成功合流的匝道 AV 总数
    total_speeds = []
    merge_times = []                # 最后一辆匝道 AV 合流的 step

    for ep in range(n_episodes):
        obs, info = env.reset()

        supervisor = SafetySupervisor(env) if mode in ("marl", "marl_pidm") else None

        # 识别哪些 controlled AV 是匝道出发的
        controlled = env.unwrapped.controlled_vehicles
        ramp_av_indices = []
        for i, av in enumerate(controlled):
            if _is_ramp_origin(av.lane_index):
                ramp_av_indices.append(i)

        n_ramp = len(ramp_av_indices)
        total_ramp_avs += n_ramp

        # 每辆匝道 AV 的合流状态：{av_index: merge_step or None}
        ramp_merge_step = {i: None for i in ramp_av_indices}
        any_crash = False

        for step in range(max_steps):
            if mode == "idm":
                actions = tuple(1 for _ in range(len(obs)))
            else:
                actions = []
                for o in obs:
                    with torch.no_grad():
                        o_t = torch.FloatTensor(flatten_and_pad(o)).unsqueeze(0)
                        q_vals = q_net(o_t)
                        actions.append(q_vals.argmax(dim=1).item())
                actions = tuple(actions)
                actions = supervisor.safe_actions(actions)

            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # 记录速度
            for av in controlled:
                total_speeds.append(av.speed)

            # 检查碰撞
            any_crash = any(av.crashed for av in controlled)

            # 检查每辆匝道 AV 是否已并入主道
            for i in ramp_av_indices:
                if ramp_merge_step[i] is None:
                    if _is_on_main_road(controlled[i], env.unwrapped.road):
                        ramp_merge_step[i] = step + 1

            if done:
                if any_crash:
                    collisions += 1
                break

        # 统计本 episode 的合流情况
        n_merged_this_ep = sum(1 for s in ramp_merge_step.values() if s is not None)
        merged_ramp_avs += n_merged_this_ep

        # 所有匝道 AV 都成功合流且无碰撞 → episode 级 success
        if n_merged_this_ep == n_ramp and not any_crash:
            all_ramp_merged_count += 1
            # TTM = 最后一辆匝道 AV 完成合流的 step
            last_merge_step = max(ramp_merge_step.values())
            merge_times.append(last_merge_step)

        # 进度日志
        if (ep + 1) % 10 == 0:
            curr_cr = collisions / (ep + 1)
            curr_ms = all_ramp_merged_count / (ep + 1)
            curr_ratio = merged_ramp_avs / total_ramp_avs if total_ramp_avs > 0 else 0
            print(f"    [{ep+1:3d}/{n_episodes}] collision={curr_cr:.0%}, "
                  f"all_merged={curr_ms:.0%}, ramp_ratio={curr_ratio:.0%}", flush=True)

    env.close()

    collision_rate = collisions / n_episodes
    merge_success_rate = all_ramp_merged_count / n_episodes
    ramp_merge_ratio = merged_ramp_avs / total_ramp_avs if total_ramp_avs > 0 else 0
    avg_speed = np.mean(total_speeds) if total_speeds else 0
    avg_merge_time = np.mean(merge_times) if merge_times else max_steps

    return {
        "mode": mode,
        "cooperation": cooperation_level,
        "collision_rate": collision_rate,
        "merge_success_rate": merge_success_rate,
        "ramp_merge_ratio": ramp_merge_ratio,
        "avg_speed": avg_speed,
        "avg_merge_time": avg_merge_time,
    }


if __name__ == "__main__":
    methods = ["idm", "marl", "marl_pidm"]
    coop_levels = ["cooperative", "non_cooperative", "mixed"]
    all_results = []

    for coop in coop_levels:
        print(f"\n{'━'*75}")
        print(f"  Cooperation Level: {coop.upper()}")
        print(f"{'━'*75}")

        for method in methods:
            print(f"  Running: {method} × {coop} (100 episodes)...")
            r = evaluate(mode=method, cooperation_level=coop, n_episodes=100)
            all_results.append(r)
            print(f"  → collision={r['collision_rate']:.0%}, "
                  f"all_merged={r['merge_success_rate']:.0%}, "
                  f"ramp_ratio={r['ramp_merge_ratio']:.0%}\n")

    # ===== 按 cooperation level 分组输出 =====
    for coop in coop_levels:
        subset = [r for r in all_results if r["cooperation"] == coop]
        print(f"\n{'='*85}")
        print(f"  HDV Cooperation: {coop.upper()}")
        print(f"{'='*85}")
        print(f"  {'方法':<15} {'Collision':>10} {'All Merged':>11} {'Ramp Ratio':>11} {'Speed':>8} {'TTM':>6}")
        print(f"  {'-'*61}")
        for r in subset:
            print(f"  {r['mode']:<15} {r['collision_rate']:>9.1%} "
                  f"{r['merge_success_rate']:>10.1%} "
                  f"{r['ramp_merge_ratio']:>10.1%} "
                  f"{r['avg_speed']:>7.2f} {r['avg_merge_time']:>5.1f}")

    # ===== 总汇总表 =====
    print(f"\n\n{'='*95}")
    print(f"  FULL EXPERIMENT MATRIX — Hard difficulty, 100 episodes, STRICT metrics")
    print(f"{'='*95}")
    print(f"  {'方法':<12} {'Cooperation':<16} {'Collision':>10} {'All Merged':>11} "
          f"{'Ramp Ratio':>11} {'Speed':>8} {'TTM':>6}")
    print(f"  {'-'*74}")
    for r in all_results:
        print(f"  {r['mode']:<12} {r['cooperation']:<16} {r['collision_rate']:>9.1%} "
              f"{r['merge_success_rate']:>10.1%} "
              f"{r['ramp_merge_ratio']:>10.1%} "
              f"{r['avg_speed']:>7.2f} {r['avg_merge_time']:>5.1f}")
    print(f"{'='*95}")