# Failure Analysis & Limitations

## 1. System-Level Merge Remains Unsolved

Across all methods, cooperation levels, and reward configurations, **All Merged = 0%** — no single episode achieved full system-level merging where all four ramp AVs successfully entered the main road. The best observed individual ramp merge ratio was 72% (MARL baseline under mixed cooperation), meaning at most approximately three out of four ramp AVs completed merging in any given episode.

**Root cause analysis:** The Hard scenario places four ramp AVs on a 300m ramp, each needing to find a gap in main-road traffic and execute a lane change within 50 steps. With DQN + parameter sharing, all AVs share one Q-network and make decisions based only on local observations (nearest 11 vehicles). No AV has information about what the other ramp AVs are doing or planning. This creates a fundamental coordination gap: two ramp AVs may simultaneously target the same gap in main-road traffic, leading to one succeeding and the other being forced to abort or collide.

**Implication:** Achieving full system-level merge likely requires either explicit inter-agent communication, a centralized critic with global state information (e.g., MAPPO / QMIX), or a sequential decision protocol where ramp AVs take turns attempting to merge.

---

## 2. P-IDM Training Variance

The MARL+P-IDM method exhibited significant training variance across runs. Under identical code, hyperparameters, and curriculum configuration, different random seeds produced markedly different evaluation outcomes. In the best run, MARL+P-IDM achieved 0% collision with 50% ramp merge ratio under cooperative traffic. In other runs, the same configuration produced 100% collision.

By contrast, the MARL baseline (static IDM) consistently produced low-collision strategies across all runs, suggesting that the static IDM environment presents a more stable optimization landscape for DQN.

**Root cause analysis:** P-IDM makes HDV behavior contingent on AV actions — when an AV signals a lane change, nearby HDVs adjust their acceleration. This creates a feedback loop: the AV's policy affects HDV behavior, which in turn affects the AV's next observation and reward. DQN, as an off-policy algorithm with a replay buffer, is particularly sensitive to this kind of non-stationarity because older experiences in the buffer were collected under a different effective environment (when the policy was different, HDVs responded differently). The result is high gradient variance during training, leading to inconsistent convergence across seeds.

**Implication:** On-policy algorithms (PPO, MAPPO) that do not reuse stale experiences may be better suited for interaction-aware environments where the agent's behavior directly modifies the dynamics of other traffic participants.

---

## 3. Safety-Efficiency Trade-Off in Reward Design

Multiple rounds of reward iteration revealed a persistent tension between merge success rate and collision rate. Three reward configurations were tested:

**Configuration A (baseline):** Collision penalty (-10), speed reward, survival bonus. Result: agents learned to avoid collisions but rarely attempted merging, as there was no incentive to do so.

**Configuration B (merge reward):** Added merge bonus (+5.0 one-time) and merge progress shaping (proportional to longitudinal position, weight 0.1). Result: agents began actively merging, with training reward increasing from ~+41 to ~+59. However, the merge progress component occasionally encouraged aggressive forward movement without sufficient gap assessment.

**Configuration C (terminal bonus):** Added episode-end shared reward proportional to the number of successfully merged ramp AVs. Result: ramp merge ratio increased in some conditions, but collision rate simultaneously worsened. The terminal bonus pushed the policy toward riskier merge attempts.

**Conclusion:** Within the DQN + local reward framework, improving merge success and maintaining safety appear to be competing objectives. The reward signal that encourages merging (forward progress, merge bonus) inherently increases exposure to collision risk. A more sophisticated approach — such as constrained optimization (Lagrangian methods) or separate safety and task critics — may be necessary to jointly optimize both objectives.

---

## 4. Merge State Misclassification Bug

During evaluation, an initial implementation classified a ramp AV as "successfully merged" based on its lane_index prefix: any vehicle whose lane_index started with nodes `("a", "b", "c", "d")` was considered to be on the main road. This produced inflated ramp merge ratios (e.g., 75% instead of the true 25%).

**Root cause:** In highway-env's road network, the ramp's final parallel segment is registered as `("b", "c", 2)` via `net.add_lane("b", "c", lbc)`. This shares the same `("b", "c")` prefix as the main road lanes `("b", "c", 0)` and `("b", "c", 1)`. A ramp AV that simply drove to the end of the ramp — without ever changing lanes into the main road — would be incorrectly classified as merged.

**Fix:** Replaced string-based lane classification with a check on the lane object's `forbidden` attribute. All ramp segments (straight, curve, and parallel merge zone) are constructed with `forbidden=True`, while main road lanes have `forbidden=False`. The corrected detection logic:

```python
lane = road.network.get_lane(vehicle.lane_index)
on_main_road = not getattr(lane, 'forbidden', False)
```

**Impact:** This bug simultaneously affected the evaluation metric (ramp_ratio) and the training reward signal (merge_bonus was being granted prematurely). After fixing, all models were retrained with correct merge detection, producing substantially different — and more trustworthy — evaluation results.

**Lesson:** In simulation environments with complex road network topologies, lane identity should never be inferred from node-name strings alone. Semantic properties of the lane object itself (such as `forbidden`, `speed_limit`, or lane type) provide more reliable classification.

---

## 5. Curriculum Transition Instability

When transitioning between curriculum stages (Easy → Medium → Hard), a naive approach of maintaining the epsilon decay schedule across stages caused the agent to enter harder environments with insufficient exploration. At the start of the Hard stage, epsilon had already decayed to ~0.21, leaving the agent committed to a policy learned in simpler settings. In the 11-vehicle Hard scenario, this policy produced identical collision patterns every episode, with reward locked at a constant negative value.

**Fix:** Epsilon is reset at each curriculum stage transition (Easy: 1.0, Medium: 0.8, Hard: 0.6), giving the agent adequate exploration budget to adapt to the new environment complexity.

---

## 6. Non-Cooperative Traffic Generalization

Both MARL and MARL+P-IDM were trained under mixed cooperation (HDVs randomly assigned cooperative or non-cooperative parameters). Despite this, performance under pure non-cooperative evaluation varied significantly between methods and across runs. In some configurations, methods that performed well under cooperative and mixed traffic showed 100% collision rates under non-cooperative traffic.

**Analysis:** Non-cooperative HDVs use aggressive IDM parameters (small following distance, high acceleration), creating a traffic environment with very tight gaps and fast-closing windows. The DQN policy, trained with mixed traffic where roughly half the HDVs are cooperative, may not have seen enough purely non-cooperative scenarios to develop robust gap-acceptance strategies for this extreme condition.

**Implication:** For safety-critical deployment, training should include dedicated curriculum stages with varying cooperation distributions, potentially including a pure non-cooperative phase, to ensure policy robustness across the full spectrum of human driver behaviors.
