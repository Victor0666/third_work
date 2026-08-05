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
    v2 priority rule: Deadline-robust urgency + criticality-aware energy gating + 
                      starvation-resilient work-density fairness + uncertainty-adaptive risk scaling.
    
    Key self-evolution improvements over v1:
    - Replaces tanh(-rel_slack) with *smooth clipped urgency* tanh(-clip(rel_slack, -5, 5)) 
      to prevent saturation distortion for extreme slack values (e.g., huge positive/negative).
    - Introduces *criticality-weighted energy penalty*: energy_penalty *= (upward_rank / (median_upward_rank + eps))
      to amplify energy optimization only on high-criticality tasks — avoids wasting low-energy bias on trivial nodes.
    - Fairness boost now uses *log-normalized wait density*: log(1 + ready_wait_time / (work_density + eps)),
      providing graceful saturation and better small-N stability vs. raw ratio division.
    - Uncertainty coupling replaces static median threshold with *adaptive percentile gate*: 
      uncertainty > percentile(uncertainty, 75) — more robust under skewed distributions.
    - Adds *deadline proximity booster*: when 0 < rel_slack < 0.2, apply negative bonus to critical_timing 
      to gently accelerate near-deadline critical tasks without triggering hard-penalty mode.
    - All normalizations use *stable quantile-based clipping* (5–95%) with explicit fallback for degenerate cases.
    - Final score bounded in [-1e12, 1e12] with deterministic NaN/inf handling; shape assertion preserved.
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64).copy()
    slack = np.asarray(slack, dtype=np.float64).copy()
    upward_rank = np.asarray(upward_rank, dtype=np.float64).copy()
    remaining_work = np.asarray(remaining_work, dtype=np.float64).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64).copy()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)

    def robust_5_95_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        if N == 1:
            return np.zeros_like(x)
        p05 = np.percentile(x, 5.0, method='lower')
        p95 = np.percentile(x, 95.0, method='higher')
        x_clipped = np.clip(x, p05, p95)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)

    # Hard lateness penalty: dominates all other terms when deadline violated
    lateness_mask = slack <= 0.0
    abs_slack = np.abs(slack)
    lateness_penalty = np.where(lateness_mask, -1000000000000.0 * (1.0 + 0.1 * abs_slack), 0.0)

    # Duration and normalized metrics
    duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.nan_to_num(rel_slack, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Smooth urgency: clip input to tanh to avoid saturation artifacts for extreme slack
    clipped_rel_slack = np.clip(rel_slack, -5.0, 5.0)
    urgency_base = np.tanh(-clipped_rel_slack)

    # Critical timing: duration × upward_rank, normalized
    critical_timing = duration * upward_rank
    norm_critical_timing = robust_5_95_norm(critical_timing)

    # Deadline proximity booster: gentle acceleration for near-deadline critical tasks
    proximity_mask = (rel_slack > 0.0) & (rel_slack < 0.2)
    proximity_booster = -0.15 * norm_critical_timing * proximity_mask

    # Energy density and gated penalty: now scaled by relative criticality
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.nan_to_num(energy_density, nan=0.0, posinf=0.0, neginf=0.0)
    norm_energy_density = robust_5_95_norm(energy_density)
    
    ur_median = np.median(upward_rank) if N > 1 else upward_rank[0]
    # Criticality-relative scaling: higher upward_rank → stronger energy optimization weight
    criticality_ratio = np.divide(upward_rank, ur_median + eps, out=np.ones_like(upward_rank), where=ur_median + eps != 0)
    energy_gate = (upward_rank > ur_median) & (slack > 0.0)
    energy_penalty = norm_energy_density * urgency_base * criticality_ratio * energy_gate

    # Fairness boost: log-normalized wait density to avoid division instability and saturate gracefully
    work_density = np.divide(remaining_work, duration + eps, out=np.zeros_like(remaining_work), where=duration + eps != 0)
    wait_density = np.divide(ready_wait_time, work_density + eps, out=np.zeros_like(ready_wait_time), where=work_density + eps != 0)
    log_wait_score = np.log1p(wait_density)  # log(1 + x) avoids -inf at zero
    norm_wait_score = robust_5_95_norm(log_wait_score)
    rw_median = np.median(remaining_work) if N > 1 else remaining_work[0]
    fairness_mask = (slack > 0.0) & (remaining_work > rw_median)
    fairness_boost = norm_wait_score * fairness_mask

    # Uncertainty coupling: adaptive 75th percentile gate instead of fixed median
    unc_p75 = np.percentile(uncertainty, 75.0, method='higher') if N > 1 else uncertainty[0]
    unc_mask = (slack > 0.0) & (upward_rank > ur_median) & (uncertainty > unc_p75)
    norm_uncertainty = robust_5_95_norm(uncertainty)
    unc_coupling = norm_uncertainty * unc_mask

    # Composite score with calibrated weights
    score = (
        0.38 * norm_critical_timing +
        0.22 * urgency_base +
        0.18 * energy_penalty +
        0.11 * fairness_boost +
        0.06 * unc_coupling +
        0.04 * robust_5_95_norm(remaining_work) +
        proximity_booster
    )

    # Apply hard penalty last to dominate all soft components
    score = lateness_penalty + score

    # Robust final sanitization
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    score = np.clip(score, -1000000000000.0, 1000000000000.0)

    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
