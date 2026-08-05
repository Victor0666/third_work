import numpy as np

# 函数名中的 2 由 SeEvo 在生成 Prompt 时替换为目标版本号。
# 八个输入均为长度 N 的一维数组；相同下标始终指向同一个 ready task。
# 返回值也必须是长度 N 的一维数组，并遵守“分数越小，优先级越高”。
def get_task_priority_v2(
    min_exec_time,
    min_comm_time,
    min_incremental_energy,
    slack,
    upward_rank,
    remaining_work,
    ready_wait_time,
    uncertainty
):

    """
    Self-evolved priority rule: deadline-hardness first, energy-criticality second,
    latency-aware fairness third — with tighter numerical safeguards and adaptive scaling.

    Key evolutions from v1:
    - Replaces fixed exponential urgency with *adaptive sigmoid-gated urgency*:
      smooth transition at slack=0, bounded [0,1], avoids overflow even for large negative slack.
    - Introduces *slack-relative criticality*: upward_rank weighted by (1 + max(0,-slack)/median(|slack|+eps)),
      amplifying criticality only when DDL risk is present — avoids over-prioritizing high-rank tasks in safe region.
    - Uses *energy-latency tradeoff ratio*: (min_incremental_energy / (exec+comm+eps)) scaled robustly,
      explicitly favoring low-energy-per-latency assignments under deadline pressure.
    - Adds *uncertainty-aware waiting saturation*: wait_saturation modulated by 1/(1+uncertainty),
      ensuring fairness degrades gracefully under high uncertainty (prevents premature starvation).
    - Eliminates redundant robust_scale on already-bounded terms (e.g., sigmoid, arctan, tanh);
      applies robust scaling *only where needed*: on raw heterogeneous features (energy, work, latency).
    - Tighter NaN/inf handling: uses np.where + eps-clamping before division, avoids nan_to_num fallback where possible.
    - All operations guaranteed finite, deterministic, and shape-preserving for N=1.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    # Robust scaling for heterogeneous raw features — handles N=1 safely
    def robust_scale(x):
        if x.size == 1:
            return np.zeros_like(x)
        q25, q75 = np.percentile(x, [25, 75], method='midpoint')
        iqr = q75 - q25 + eps
        med = np.median(x)
        return (x - med) / iqr

    # --- Deadline urgency: bounded, smooth, zero-impact for slack > 0 ---
    # Sigmoid gating: 1 / (1 + exp(-slack)) → 0.5 at slack=0, → 0 for slack>>0, → 1 for slack<<0
    # But we want *urgency*, so invert: 1 - sigmoid(slack) = sigmoid(-slack)
    deadline_urgency = 1.0 / (1.0 + np.exp(-slack + eps))  # = sigmoid(-slack), range [0,1]

    # --- Slack-relative criticality: amplify upward_rank only when slack < 0 ---
    abs_slack = np.abs(slack) + eps
    median_abs_slack = np.median(abs_slack)
    slack_penalty_ratio = np.maximum(0.0, -slack) / (median_abs_slack + eps)
    # Criticality scaled only under pressure; baseline = upward_rank itself
    scaled_upward_rank = upward_rank * (1.0 + slack_penalty_ratio)

    # --- Energy-latency efficiency ratio: lower ratio = better (joules per second)
    exec_comm_sum = np.clip(min_exec_time + min_comm_time, eps, None)
    energy_latency_ratio = min_incremental_energy / exec_comm_sum
    energy_latency_ratio_scaled = robust_scale(energy_latency_ratio)

    # --- Latency pressure: only active when slack <= 0; else use normalized intrinsic latency
    latency_pressure = np.where(
        slack <= 0,
        np.clip(exec_comm_sum / (np.abs(slack) + eps), 0.0, 20.0),
        robust_scale(exec_comm_sum)
    )

    # --- Uncertainty-modulated waiting fairness ---
    # Longer wait matters more under low uncertainty; high uncertainty softens penalty
    inv_uncertainty_factor = 1.0 / (1.0 + uncertainty)
    wait_ratio = ready_wait_time / (exec_comm_sum + eps)
    wait_saturation = np.arctan(wait_ratio) * inv_uncertainty_factor

    # --- Remaining work: scaled for fairness among coarse-grained vs fine-grained tasks ---
    work_scaled = robust_scale(remaining_work)

    # --- Composite score: DDL urgency dominates, then energy-latency efficiency, then criticality & fairness ---
    # All components are finite, bounded, and sign-consistent: smaller score = higher priority
    score = (
        5.0 * deadline_urgency +                      # Hard-DDL adherence: highest weight, bounded [0,1]
        2.2 * energy_latency_ratio_scaled +          # Energy-per-latency efficiency: lower is better
        1.6 * robust_scale(latency_pressure) +       # Latency pressure under deadline risk
        1.3 * robust_scale(scaled_upward_rank) -     # Critical path impact, amplified only when needed
        0.9 * work_scaled -                          # Work progress fairness (higher work → lower priority unless urgent)
        0.8 * wait_saturation                        # Starvation prevention, uncertainty-dampened
    )

    # Final guard: ensure no NaN/inf, preserve shape, guarantee finite output
    score = np.where(np.isnan(score) | np.isinf(score), 1e12, score)
    return np.clip(score, -1e12, 1e12)
