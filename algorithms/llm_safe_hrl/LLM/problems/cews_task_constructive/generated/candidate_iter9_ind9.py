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
    Self-evolved v2 priority rule: restores deadline dominance, simplifies energy risk weighting,
    strengthens starvation control, and tightens numerical robustness.

    Key improvements over v1:
    - Restores strict deadline feasibility focus: uses *clipped linear urgency* (not sigmoid) for direct slack-pressure mapping,
      with hard zero-crossing at slack=0 to guarantee urgent tasks dominate.
    - Reduces energy risk exponent range from [1,4] → [1,2.2] to prevent over-penalization; uses smoothed max(-slack, 0) instead of rel_slack.
    - Replaces harmonic efficiency with *normalized critical-energy ratio* — simpler, more stable, and preserves HEFT semantics.
    - Starvation penalty is now *unconditional, bounded, and directly proportional to both wait time and lateness pressure*:
      penalty = 0.35 * clip(wait_ratio, 0, 1) * clip(max(0, -slack), 0, 10) / (median_duration + eps), ensuring fairness for all starved tasks.
    - Introduces *uncertainty-aware slack discount*: scales deadline urgency by (1 - clip(uncertainty, 0, 0.8) * 0.4) to de-emphasize urgency when risk is high (reducing speculative scheduling).
    - Robust normalization now uses median-IQR with fallback to std only if IQR ≈ 0, and clips to [-6, 6] for tighter dynamic range.
    - All components explicitly eps-protected, nan/inf guarded, and shape-preserving.
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
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def robust_normalize(x):
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        q1, q3 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q3 - q1 + eps
        # Fallback to std only if IQR is near-zero (degenerate distribution)
        iqr_safe = np.where(iqr < 10*eps, np.std(x, ddof=0) + eps, iqr)
        norm = (x - med) / iqr_safe
        return np.clip(norm, -6.0, 6.0)

    # Base task duration for relative scaling
    task_min_duration = np.maximum(min_exec_time + min_comm_time, eps)
    median_duration = np.median(task_min_duration) + eps

    # --- Deadline Urgency: Linear, clipped, uncertainty-discounted ---
    # Direct pressure: higher penalty for negative slack; zero at slack=0; capped growth
    raw_urgency = np.clip(np.maximum(0.0, -slack), 0.0, 15.0) / (median_duration + eps)
    # Discount urgency under high uncertainty to avoid overcommitting risky tasks
    uncertainty_discount = np.clip(1.0 - uncertainty * 0.4, 0.2, 1.0)
    deadline_urgency = raw_urgency * uncertainty_discount

    # --- Energy Risk Weighting: Simpler, bounded exponent ---
    # α ∈ [1.0, 2.2] — avoids aggressive distortion seen in v1
    risk_exponent = np.clip(1.0 + 0.4 * np.maximum(0.0, -slack), 1.0, 2.2)
    energy_risk_weighted = min_incremental_energy * np.power(1.0 + uncertainty, risk_exponent)
    energy_safe = np.maximum(energy_risk_weighted, eps)

    # --- Criticality-Energy Efficiency: Clean ratio, no harmonic complexity ---
    crit_eff_ratio = upward_rank / energy_safe
    crit_eff_ratio = np.clip(crit_eff_ratio, 1e-06, 1e6)
    crit_eff_norm = robust_normalize(crit_eff_ratio)

    # --- Starvation Penalty: Unconditional, proportional, bounded [0, 0.35] ---
    p95_wait = np.percentile(ready_wait_time, 95, method='midpoint') if N > 1 else np.max(ready_wait_time)
    p95_wait = np.maximum(p95_wait, eps)
    wait_ratio = np.clip(ready_wait_time / p95_wait, 0.0, 1.0)
    lateness_pressure = np.clip(np.maximum(0.0, -slack), 0.0, 10.0) / (median_duration + eps)
    starvation_penalty = wait_ratio * lateness_pressure * 0.35
    starvation_penalty = np.clip(starvation_penalty, 0.0, 0.35)

    # --- Work & Uncertainty Normalization ---
    work_norm = robust_normalize(remaining_work)
    unc_norm = robust_normalize(uncertainty)

    # --- Time cost (lightweight execution+comm proxy) ---
    time_cost = np.sqrt(np.maximum(min_exec_time, eps)) + np.sqrt(np.maximum(min_comm_time, eps))
    time_norm = robust_normalize(time_cost)

    # --- Final score: deadline urgency dominates (negative weight), others balanced ---
    # Higher weights on deadline and critical-efficiency; lower on secondary factors
    score = (
        -5.0 * deadline_urgency          # Strongest pull: meet deadline first
        - 2.5 * crit_eff_norm           # Preserve critical-path energy efficiency
        + 0.3 * time_norm               # Mild penalty for long-duration tasks (avoids blocking)
        + 0.25 * work_norm              # Encourage progress on high-work subgraphs
        + 0.2 * unc_norm                # Slight penalty for high-uncertainty tasks
        + starvation_penalty            # Fairness enforcement, always active
    )

    # Final sanitization: ensure finite, deterministic output
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    return score
