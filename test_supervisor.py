"""验证 Safety Supervisor 基本功能：接受动作、返回动作、不报错。"""
from gymnasium.envs.registration import register
import gymnasium as gym
from safety_supervisor import SafetySupervisor

register(id="merge-multi-v0", entry_point="env_multi_agent:MultiAgentMergeEnv")

env = gym.make("merge-multi-v0")
obs, info = env.reset()

supervisor = SafetySupervisor(env)

# 跑 20 步，看 supervisor 是否正常工作
for step in range(20):
    raw_actions = tuple(env.action_space.sample())  # 随机动作
    safe_actions = supervisor.safe_actions(raw_actions)

    changed = "← REPLACED" if raw_actions != safe_actions else ""
    print(f"Step {step:2d}: raw={raw_actions}, safe={safe_actions} {changed}")

    obs, reward, terminated, truncated, info = env.step(safe_actions)
    if terminated or truncated:
        print(f"  Episode ended at step {step}")
        obs, info = env.reset()

env.close()
print("\n✅ Safety Supervisor 运行正常")