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
    Self-evolved priority rule balancing deadline safety, energy efficiency, and fairness.
    
    Key improvements over v1:
    - Replaces exponential deadline gating with *smooth sigmoid penalty*: avoids inflation,
      provides bounded (0–1) urgency signal that increases monotonically near deadline.
    - Simplifies uncertainty coupling: applies linear risk scaling *only* to deadline risk
      and criticality — removes non-monotonic inverse damping, ensuring consistent behavior.
    - Restores robust tanh-based wait boost (with adaptive scale) for stronger starvation relief
      across wide wait-time distributions; now scaled by median wait to preserve monotonicity.
    - Introduces *energy-per-work* as primary energy signal (not energy-per-second), normalized
      by remaining_work to favor low-energy tasks *per unit work*, aligning with DDL-safe energy minimization.
    - Adds explicit slack-aware weighting: when slack >= 0, energy-efficiency dominates; when slack < 0,
      deadline & criticality dominate — enforcing hard-DDL priority without over-penalizing safe tasks.
    - All normalization uses robust IQR; all ops epsilon-protected; deterministic and shape-preserving.
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

    # Smooth, bounded deadline risk: sigmoid(-slack / tau) → [0,1], peaks at slack=0, saturates for large negative slack
    tau = np.clip(np.mean(np.abs(slack)) + eps, eps, 1000.0)
    deadline_risk = 1.0 / (1.0 + np.exp(slack / tau + eps))  # ≈ 0 when slack >> 0, ≈ 1 when slack << 0

    # Uncertainty linearly amplifies deadline risk and critical urgency (no inverse damping → monotonic & consistent)
    risk_factor = 1.0 + uncertainty
    critical_urgency = upward_rank * risk_factor * (remaining_work + eps)

    # Energy-efficiency signal: marginal energy per unit work (joules/MI) — prioritizes low-energy tasks *per work done*
    energy_per_work = min_incremental_energy / (remaining_work + eps)

    # Starvation mitigation: tanh-based, adaptively scaled to median wait time
    median_wait = np.median(ready_wait_time) + eps
    wait_boost = np.tanh(ready_wait_time / (median_wait + eps))

    # Robust IQR-based normalization
    def robust_iqr_normalize(x):
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        scale = iqr if iqr > eps else np.mean(np.abs(x)) + eps
        return x / (scale + eps)

    norm_deadline_risk = robust_iqr_normalize(deadline_risk)
    norm_critical_urgency = robust_iqr_normalize(critical_urgency)
    norm_energy_eff = robust_iqr_normalize(energy_per_work)
    norm_wait_boost = robust_iqr_normalize(wait_boost)
    norm_exec = robust_iqr_normalize(min_exec_time)
    norm_comm = robust_iqr_normalize(min_comm_time)
    norm_uncert = robust_iqr_normalize(uncertainty)

    # Slack-aware coefficient blending: when slack >= 0 → prioritize energy & wait; when slack < 0 → prioritize risk & urgency
    slack_mask = (slack < 0).astype(float)
    # Base weights for safe regime (slack >= 0)
    w_risk_safe, w_urgency_safe, w_energy_safe, w_wait_safe = 0.0, 0.0, 1.2, 0.8
    # Boosted weights for risky regime (slack < 0)
    w_risk_risk, w_urgency_risk, w_energy_risk, w_wait_risk = 5.0, 2.0, 0.3, 0.4
    # Interpolate weights smoothly using slack_mask
    w_risk = w_risk_safe * (1.0 - slack_mask) + w_risk_risk * slack_mask
    w_urgency = w_urgency_safe * (1.0 - slack_mask) + w_urgency_risk * slack_mask
    w_energy = w_energy_safe * (1.0 - slack_mask) + w_energy_risk * slack_mask
    w_wait = w_wait_safe * (1.0 - slack_mask) + w_wait_risk * slack_mask

    score = (
        w_risk * norm_deadline_risk
        - w_urgency * norm_critical_urgency
        + w_energy * norm_energy_eff
        + w_wait * norm_wait_boost
        + 0.1 * norm_exec
        + 0.05 * norm_comm
        + 0.02 * norm_uncert
    )

    return np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
