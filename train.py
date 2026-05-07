"""
P3 MARL Training Pipeline
DQN + Parameter Sharing + Local Reward + Safety Supervisor + Curriculum Learning

用法：
    python train.py                  # 默认 use_pidm=True
    python train.py --no-pidm        # 训练 MARL baseline（静态 IDM）
"""
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gymnasium.envs.registration import register
import gymnasium as gym

from safety_supervisor import SafetySupervisor


# ============================================================
# 全局常量
# ============================================================
MAX_OBS_DIM = 55          # Hard 模式 obs: 11×5 = 55，所有难度统一 pad 到此维度
N_ACTIONS = 5

CURRICULUM = [
    {"difficulty": "easy",   "episodes": 500,  "epsilon_start": 1.0},
    {"difficulty": "medium", "episodes": 800,  "epsilon_start": 0.8},
    {"difficulty": "hard",   "episodes": 1200, "epsilon_start": 0.6},
]


# ============================================================
# 工具函数：obs 展平 + 零填充
# ============================================================
def flatten_and_pad(obs, target_dim=MAX_OBS_DIM):
    """将单辆 AV 的 obs 展平并零填充到 target_dim。"""
    flat = obs.flatten()
    if len(flat) < target_dim:
        flat = np.concatenate([flat, np.zeros(target_dim - len(flat))])
    return flat


# ============================================================
# 组件 1：共享 Q 网络
# ============================================================
class QNetwork(nn.Module):
    """
    所有 AV 共享的 Q 网络。
    输入：55 维（统一 pad 后）
    输出：5 个动作的 Q 值
    """
    def __init__(self, obs_dim=MAX_OBS_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 组件 2：Replay Buffer
# ============================================================
class ReplayBuffer:
    """
    经验回放缓冲区。
    所有 AV 的经验混在一起存储（parameter sharing 的核心）。
    """
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity) 

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch) # 64*5 -> 5*64, 按特征分组
        return (
            torch.FloatTensor(np.array(obs)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_obs)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 组件 3：Local Reward（v2：系统级 merge shaping）
# ============================================================
def compute_local_rewards(env, obs_tuple, ramp_av_indices, merged_avs):
    """
    计算每辆 AV 的局部 reward。

    基础奖励（所有 AV）：
    - collision:       -10.0（碰撞惩罚）
    - high_speed:      speed / max_speed * 0.4（速度奖励）
    - on_road:         +0.1（存活奖励）

    Merge 奖励（仅限匝道 AV）：
    - merge_bonus:     +5.0（首次成功并入主道，一次性）
    - merge_progress:  position / merge_zone_start * 0.1（减弱，避免鼓励个体蛮冲）

    注：系统级 shaping 以 terminal bonus 形式在训练循环中发放，不在此函数内。
    """
    from env_multi_agent import DIFFICULTY_CONFIGS

    controlled = env.unwrapped.controlled_vehicles
    diff = env.unwrapped.config.get("difficulty", "easy")
    road_ends = DIFFICULTY_CONFIGS[diff]["road_ends"]
    merge_zone_start = sum(road_ends[:2])

    rewards = []
    for i, av in enumerate(controlled):
        r = 0.0
        if av.crashed:
            r -= 10.0
        else:
            # --- 基础奖励 ---
            max_speed = 30.0
            r += av.speed / max_speed * 0.4
            r += 0.1

            # --- Merge 奖励（仅限匝道 AV）---
            if i in ramp_av_indices:
                current_lane = env.unwrapped.road.network.get_lane(av.lane_index)
                on_main_road = not getattr(current_lane, 'forbidden', False)

                if on_main_road and i not in merged_avs:
                    r += 5.0
                    merged_avs.add(i) # # 记录已拿过奖励，防止重复发放
                elif not on_main_road and i not in merged_avs:
                    progress = min(av.position[0] / merge_zone_start, 1.0)
                    r += progress * 0.1

        rewards.append(r)
    return rewards


# ============================================================
# 组件 4：动作选择 + 训练步
# ============================================================
def select_actions(q_net, obs_tuple, epsilon):
    """Epsilon-greedy，每辆 AV 用同一个 Q 网络各自选动作。"""
    actions = []
    for obs in obs_tuple:
        if random.random() < epsilon:
            actions.append(random.randint(0, N_ACTIONS - 1))
        else:
            with torch.no_grad(): # 选动作时不需要反向传播
                obs_t = torch.FloatTensor(flatten_and_pad(obs)).unsqueeze(0) # 转化为批次格式 (batch_size, feature_dim)
                q_values = q_net(obs_t)
                actions.append(q_values.argmax(dim=1).item()) # 选取q值最大的动作
    return tuple(actions)


def train_step(q_net, target_net, buffer, optimizer, batch_size=64, gamma=0.99):
    """DQN 更新一步：采样 → TD target → MSE loss → 反向传播。"""
    if len(buffer) < batch_size: # buffer 里经验不够64条时不训练，直接返回。训练初期需要先积累经验。
        return None

    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size)

    q_values = q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad(): #  target_net 只用来计算目标值，不参与梯度更新，关闭梯度
        next_q = target_net(next_obs).max(dim=1)[0]
        td_target = rewards + gamma * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, td_target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()


# ============================================================
# 主训练入口
# ============================================================
def main():
    # ----- 命令行参数 -----
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pidm", action="store_true",
                        help="使用静态 IDM 而非 P-IDM（训练 MARL baseline）")
    args = parser.parse_args()

    use_pidm = not args.no_pidm
    model_name = "q_net_pidm.pth" if use_pidm else "q_net_no_pidm.pth"

    # ----- 环境 -----
    register(id="merge-multi-v0", entry_point="env_multi_agent:MultiAgentMergeEnv")
    env = gym.make("merge-multi-v0")

    # 初始配置（会在每个 curriculum 阶段开始时更新）
    env.unwrapped.config.update({
        "use_pidm": use_pidm,
        "cooperation_level": "mixed",      # 让 agent 见到合作和不合作的 HDV， 增强策略的泛化性。
    })

    # ----- 网络 -----
    q_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(q_net.state_dict()) # 两个网络初始参数必须完全一致，否则一开始 td_target 就是乱的。

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=50000)

    # ----- 超参数 -----
    max_steps = 50
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0 # 初始探索率（这里实际被每个 curriculum 阶段开始时会用 stage["epsilon_start"] 覆盖它。）
    epsilon_end = 0.05
    epsilon_decay = 0.995
    target_update_freq = 10 # # 每10episode同步target_net

    epsilon = epsilon_start

    # ----- 训练记录 -----
    episode_rewards = []
    losses = []
    global_ep = 0

    total_episodes = sum(stage["episodes"] for stage in CURRICULUM)

    print("=" * 60)
    print(f"P3 MARL Training — Curriculum Learning")
    print(f"  use_pidm:   {use_pidm}")
    print(f"  save as:    {model_name}")
    print(f"  curriculum: {' → '.join(s['difficulty'] + '(' + str(s['episodes']) + ', ε=' + str(s['epsilon_start']) + ')' for s in CURRICULUM)}")
    print(f"  total:      {total_episodes} episodes")
    print(f"  parameters: {sum(p.numel() for p in q_net.parameters())}")
    print(f"  device:     CPU")
    print("=" * 60)

    for stage in CURRICULUM:
        difficulty = stage["difficulty"]
        n_episodes = stage["episodes"]

        # 切换难度
        env.unwrapped.config.update({"difficulty": difficulty})

        # ★ 重置 epsilon：新难度 = 新环境，需要重新探索
        epsilon = stage["epsilon_start"]

        # 每个阶段需要重新创建 supervisor（因为 agent 数量可能变了）
        # supervisor 在 reset 后才能读到正确的 controlled_vehicles
        # 所以我们在第一次 reset 后再创建

        print(f"\n{'─'*60}")
        print(f"  Stage: {difficulty.upper()} ({n_episodes} episodes, ε reset → {epsilon:.2f})")
        print(f"{'─'*60}")

        for ep_in_stage in range(n_episodes):
            global_ep += 1
            obs, info = env.reset()

            # 每个 stage 的第一个 episode 时重建 supervisor
            if ep_in_stage == 0:
                supervisor = SafetySupervisor(env)

            # 识别匝道 AV（初始 lane_index 在 "j" 段的 controlled vehicle）
            controlled = env.unwrapped.controlled_vehicles
            ramp_av_indices = set()
            for i, av in enumerate(controlled):
                if av.lane_index[0] == "j":
                    ramp_av_indices.add(i)
            merged_avs = set()  # 跨 step 追踪已拿过 merge bonus 的 AV

            ep_reward = 0.0
            ep_losses = []

            for step in range(max_steps):
                # 1. 选动作
                raw_actions = select_actions(q_net, obs, epsilon)

                # 2. Safety Supervisor
                safe_actions = supervisor.safe_actions(raw_actions)

                # 3. 环境 step
                next_obs, global_reward, terminated, truncated, info = env.step(safe_actions)
                done = terminated or truncated

                # 4. Local reward（传入匝道 AV 索引和已合流追踪器）
                local_rewards = compute_local_rewards(env, next_obs, ramp_av_indices, merged_avs) # 传入的是 next_obs， 也就是执行动作之后的状态


                # 5. 存入 buffer（obs 统一 pad 到 55 维）
                for i in range(len(obs)):
                    buffer.push(
                        flatten_and_pad(obs[i]),
                        safe_actions[i],
                        local_rewards[i],
                        flatten_and_pad(next_obs[i]),
                        float(done), # 参与(1-done)计算
                    )

                # 6. 训练
                loss = train_step(q_net, target_net, buffer, optimizer, batch_size, gamma)
                if loss is not None:
                    ep_losses.append(loss)

                ep_reward += sum(local_rewards)
                obs = next_obs

                if done:
                    break

            # Epsilon 衰减
            epsilon = max(epsilon_end, epsilon * epsilon_decay) # 每个episode结束后衰减，

            # 更新 target 网络
            if global_ep % target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())

            # 记录
            avg_loss = np.mean(ep_losses) if ep_losses else 0
            episode_rewards.append(ep_reward)
            losses.append(avg_loss)

            if global_ep % 10 == 0:
                recent_reward = np.mean(episode_rewards[-10:])
                print(f"  [{difficulty:6s}] Ep {global_ep:4d} | "
                      f"Reward: {ep_reward:7.2f} | "
                      f"Avg(10): {recent_reward:7.2f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Eps: {epsilon:.3f} | "
                      f"Buf: {len(buffer)}")

    # ----- 保存 -----
    torch.save(q_net.state_dict(), model_name)
    print(f"\n✅ 训练完成，模型已保存至 {model_name}")
    print(f"最终 10 episode 平均 reward: {np.mean(episode_rewards[-10:]):.2f}")

    env.close()


if __name__ == "__main__":
    main()
