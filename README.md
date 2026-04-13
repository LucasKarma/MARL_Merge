# Multi-Agent Cooperative Merging with Interaction-Aware Traffic Modeling

A MARL-based highway merging framework that investigates the impact of interaction-aware human driver modeling (P-IDM) on multi-AV cooperative merging in mixed-autonomy traffic.

## Problem

In highway merging scenarios with mixed autonomy (multiple AVs and human-driven vehicles sharing the road), AVs must coordinate lane changes into dense main-road traffic while avoiding collisions. Standard traffic models assume human drivers follow static car-following rules (IDM), ignoring the fact that human drivers observe and react to AV behavior. This project explores whether modeling this interaction — allowing HDVs to perceive AV lane-change intent — leads to safer and more effective cooperative merging strategies.

**One-line problem definition:** Design a multi-agent decision-making framework for highway merging under mixed autonomy, with interaction-aware human driver modeling, optimizing for collision avoidance and merge success rate.

## Environment

Built on [highway-env](https://github.com/Farama-Foundation/HighwayEnv), with a custom `MultiAgentMergeEnv` that extends the default `MergeEnv` with:

**Three difficulty levels** with scaled road geometry and vehicle counts:

| Difficulty | AVs | HDVs | Total | Ramp Length | Merge Zone | Road Total |
|-----------|-----|------|-------|------------|------------|------------|
| Easy | 2 | 3 | 5 | 150m | 80m | 460m |
| Medium | 3 | 4 | 7 | 200m | 100m | 600m |
| Hard | 6 | 5 | 11 | 300m | 150m | 770m |

**Three HDV cooperation levels**, controlled via IDM parameters:

| Level | COMFORT_ACC_MAX | DISTANCE_WANTED | TIME_WANTED | Behavior |
|-------|----------------|-----------------|-------------|----------|
| Cooperative | 2.0 | 7.0 | 2.0 | Loose following, yields easily |
| Non-cooperative | 5.0 | 3.0 | 0.8 | Tight following, does not yield |
| Mixed | Random per HDV | Random per HDV | Random per HDV | Heterogeneous traffic |

**Key implementation:** Road geometry (`_make_road()`) is overridden per difficulty to scale ramp length and merge zone, preventing immediate collisions at reset in high-density scenarios.

## Method

### P-IDM (Predictive IDM)

The core contribution. `PIDMVehicle` extends `IDMVehicle` by overriding `act()` to insert a cut-in detection step before IDM acceleration computation:

1. Find the standard leader on the current lane (`road.neighbour_vehicles()`)
2. Check if any AV has `target_lane_index` matching the HDV's lane and is positioned ahead — indicating an imminent lane change into the HDV's path
3. If a cutting-in AV is found and is closer than the standard leader, use it as the effective leader

This makes HDVs respond to AV intent (decelerate when an AV is about to merge in front), creating a more realistic and interactive traffic environment.

### Safety Supervisor

A priority-based safety layer that sits between the RL policy output and environment execution:

1. Rank all AVs by priority score (ramp vehicles first, closer to merge zone = higher priority, smaller headway = higher priority)
2. For each AV in priority order, predict future trajectories using kinematic equations (no simulation rollback)
3. Check predicted trajectories against all other vehicles for collision within T_n=6 steps
4. If collision detected, replace the action with the safest alternative (maximum safety margin)
5. Store confirmed safe trajectories for subsequent AV checks

**Key design decision:** The replay buffer stores supervisor-corrected actions (not raw policy actions), ensuring the Q-network learns from safe behavior rather than developing a false sense of safety.

### Training Pipeline

| Component | Implementation |
|-----------|---------------|
| Algorithm | DQN with target network (hard update every 10 episodes) |
| Parameter sharing | All AVs share one Q-network (input: 55-dim padded obs, output: 5 discrete actions) |
| Local reward | Per-AV: collision (-10), speed (0~0.4), survival (+0.1), merge bonus (+5.0 one-time), merge progress (0~0.1) |
| Curriculum learning | Easy (500 ep, ε=1.0) → Medium (800 ep, ε=0.8) → Hard (1200 ep, ε=0.6) |
| Training traffic | Mixed cooperation (HDVs randomly cooperative or non-cooperative) |
| Gradient clipping | `max_norm=10.0` to prevent Q-value overestimation cascades |

## Baselines

| Method | AV Decision | HDV Type | Description |
|--------|------------|----------|-------------|
| **IDM (weak baseline)** | All IDLE | IDMVehicle | No learning, no active merging — proves learning is necessary |
| **MARL (strong baseline)** | Trained Q-network | IDMVehicle (static) | Standard MARL without interaction-aware HDVs |
| **MARL+P-IDM (ours)** | Trained Q-network | PIDMVehicle | MARL with interaction-aware HDV modeling |

MARL and MARL+P-IDM are trained separately in their respective environments to ensure fair comparison (each policy is optimized for its own traffic dynamics).

## Metrics

All metrics use **strict definitions** with `forbidden`-attribute-based merge detection (see Failure Cases §4 for why this matters):

| Metric | Definition |
|--------|-----------|
| **Collision Rate** | Fraction of episodes where any AV crashed |
| **Ramp Merge Ratio** | Fraction of all ramp AVs (across all episodes) that successfully entered the main road (lane with `forbidden=False`) |
| **All Merged** | Fraction of episodes where all 4 ramp AVs merged and no collision occurred |
| **Average Speed** | Mean speed of all AVs across all steps and episodes |
| **Time-to-Merge** | Mean step at which the last ramp AV completes merging (only for All-Merged episodes) |

## Results

Evaluation on Hard difficulty (6 AVs + 5 HDVs, 770m road), 100 episodes per condition:

| Method | Cooperation | Collision | Ramp Ratio | Speed |
|--------|------------|-----------|------------|-------|
| IDM | Cooperative | 100% | 0% | 19.64 |
| MARL | Cooperative | **0%** | 25% | 22.68 |
| **MARL+P-IDM** | **Cooperative** | **0%** | **50%** | **22.13** |
| IDM | Non-cooperative | 100% | 0% | 19.17 |
| MARL | Non-cooperative | **0%** | 50% | 22.79 |
| MARL+P-IDM | Non-cooperative | **0%** | 25% | 21.95 |
| IDM | Mixed | 100% | 0% | 19.43 |
| MARL | Mixed | 7% | **72%** | 22.55 |
| MARL+P-IDM | Mixed | 9% | 41% | 22.04 |

**Key findings:**

- **IDM baseline fails completely** (100% collision) across all conditions, demonstrating that learning-based control is essential for merging.
- **P-IDM doubles merge ratio under cooperative traffic** (50% vs 25%) with 0% collision, showing that interaction-aware HDV modeling enables more effective cooperative merging when HDVs are willing to respond to AV intent.
- **Mixed traffic is the hardest setting.** Both MARL methods show non-zero collision rates (7-9%), with MARL achieving higher ramp ratio (72%) while both methods still showing non-zero collision rates. Training variance across runs is high in this condition.
- **All Merged remains 0%** across all methods — full system-level merging of all four ramp AVs was never achieved, indicating a major limitation of the DQN + local reward framework for multi-vehicle temporal coordination.

## Ablation: Reward Iteration

Three reward configurations were tested to investigate the safety-efficiency trade-off:

| Configuration | Collision Penalty | Merge Bonus | Merge Progress | Terminal Bonus | Outcome |
|--------------|------------------|-------------|----------------|---------------|---------|
| A: Base | -10 | — | — | — | Safe but no merging |
| B: Merge reward | -10 | +5.0 | 0.1 × progress | — | Active merging, best safety-efficiency balance |
| **C: Terminal bonus** | **-10** | **+5.0** | **0.1 × progress** | **0.5 × n_merged** | **Higher merge rate but collision rate exploded** |

Configuration B was selected as the final reward design. The terminal bonus experiment (C) confirmed that the performance bottleneck is not purely reward-driven — pushing harder toward merge success directly trades off against safety within the current framework.

## Failure Cases

See [Failure Analysis](P3_failure_analysis.md) for detailed documentation of six failure modes:

1. **System-level merge unsolved** — All Merged = 0% due to DQN + local reward coordination limits
2. **P-IDM training variance** — Interaction-aware environments create non-stationary dynamics that destabilize off-policy learning
3. **Safety-efficiency trade-off** — Merge-encouraging rewards inherently increase collision exposure
4. **Merge state misclassification** — Highway-env lane naming collision between ramp and main road, fixed via `forbidden` attribute
5. **Curriculum transition instability** — Epsilon decay across stages prevented re-exploration in harder environments
6. **Non-cooperative generalization gap** — Policies trained on mixed traffic struggle under purely aggressive HDVs

## Repo Structure

```
P3_MARL_Merge/
├── README.md                  # This file
├── P3_failure_analysis.md     # Detailed failure analysis
│
├── env_multi_agent.py         # MultiAgentMergeEnv (3 difficulty levels + cooperation levels)
├── vehicle_pidm.py            # P-IDM Vehicle (interaction-aware HDV)
├── safety_supervisor.py       # Priority-based Safety Supervisor
├── train.py                   # DQN + Parameter Sharing + Curriculum Learning
├── evaluate.py                # 3×3 experiment matrix evaluation
│
├── explore_merge.py           # Environment exploration script
├── test_pidm.py               # P-IDM behavior verification test
├── test_supervisor.py         # Safety Supervisor functional test
│
├── q_net_pidm.pth             # Trained weights (MARL+P-IDM)
├── q_net_no_pidm.pth          # Trained weights (MARL baseline)
│
└── visualization/
    └── experiment_matrix.jsx  # Interactive results visualization (React component)
```

## Tech Stack

Python 3.10 · PyTorch · highway-env 1.10.2 · gymnasium 1.2.3 · DQN · Multi-Agent RL · Parameter Sharing · Curriculum Learning

## References

- D. Chen, Z. Li, Y. Wang, L. Jiang, Y. Wang. "Deep Multi-agent Reinforcement Learning for Highway On-Ramp Merging in Mixed Traffic." *arXiv:2105.05701*, 2021. — Parameter sharing, local reward, action masking, priority-based safety supervisor, curriculum learning.
- B. Brito, J. Alonso-Mora. "Learning Interaction-aware Guidance Policies for Motion Planning in Dense Traffic Scenarios." *Delft University of Technology*. — Velocity reference, MPCC, P-IDM, SAC training pipeline.
- E. Leurent. "An Environment for Autonomous Driving Decision-Making." 2018. [GitHub](https://github.com/eleurent/highway-env).
