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
    v3 priority rule: Restores crisp hard-deadline dominance with zero-tolerance urgency gating,
    eliminates ambiguity in starvation guard via slack-conditional wait amplification (not thresholded),
    strengthens risk-aware uncertainty coupling using only slack-driven scaling (no exec_ratio dilution),
    and introduces criticality-normalized energy efficiency with workflow-relative percentile thresholding.
    
    Key improvements:
      - Urgency is strictly binary and zero-tolerance: slack <= 0 → score = -1.0 (highest priority), no soft margin.
      - Starvation guard now activates *only* when slack > 0, and scales linearly with normalized wait time — 
        no work-threshold masking to preserve fairness for all non-urgent tasks.
      - Uncertainty coupling uses pure |slack|^{-1} clamped scaling (0.1–10.0) — removes exec_ratio to refocus on deadline risk.
      - Energy penalty uses robust 90th-percentile threshold (not median) of criticality-weighted energy density,
        making it more selective against inefficient critical-path tasks.
      - All normalizations use 1%-99% clipping + min-max for stronger outlier resilience.
      - Final weights enforce strict hierarchy: urgency (-1.0 weight dominates) > energy (0.22) > latency (0.15) > 
        fairness (0.10) > uncertainty modulation (0.08) > baseline uncertainty (0.05) > remaining_work (0.04).
      - Score shifted to ensure negative urgency always wins: base = 1.0 + urgency_bias where urgency_bias = -1.0 if slack<=0.
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
    
    def robust_minmax_norm(x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0)
        p99 = np.percentile(x, 99.0)
        x_clipped = np.clip(x, p01, p99)
        x_min_c = np.min(x_clipped)
        x_max_c = np.max(x_clipped)
        if x_max_c - x_min_c < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min_c) / (x_max_c - x_min_c + eps)
    
    # Crisp urgency: zero-tolerance hard deadline enforcement → score offset = -1.0 for violation
    urgency_bias = np.where(slack <= 0, -1.0, 0.0)
    
    # Energy efficiency: criticality-weighted energy density, thresholded at 90th percentile
    duration = min_exec_time + min_comm_time + eps
    energy_density = min_incremental_energy / (duration + eps)
    eff_per_rank = energy_density / (upward_rank + eps)
    threshold_eff_per_rank = np.percentile(eff_per_rank, 90.0) + eps
    energy_penalty_mask = (eff_per_rank > threshold_eff_per_rank).astype(float)
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_penalty_mask
    
    # Latency: execution + communication, weighted by criticality and uncertainty
    comm_weight = 1.0 + 0.6 * robust_minmax_norm(upward_rank) + 0.4 * robust_minmax_norm(uncertainty)
    weighted_comm = min_comm_time * comm_weight
    latency_raw = min_exec_time + weighted_comm + eps
    norm_latency = robust_minmax_norm(latency_raw)
    
    # Fairness: starvation guard — activates only when slack > 0, no work-threshold
    starvation_mask = (slack > 0).astype(float)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_penalty = starvation_mask * norm_wait_time
    
    # Uncertainty modulation: pure slack-driven risk scaling (clamped inverse absolute slack)
    slack_abs = np.abs(slack) + 1.0
    slack_scale_factor = np.clip(1.0 / slack_abs, 0.1, 10.0)
    uncertainty_boost = uncertainty * slack_scale_factor
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Baseline uncertainty & remaining work (both penalized, lower is better)
    norm_uncertainty = robust_minmax_norm(uncertainty)
    norm_remaining_work = robust_minmax_norm(remaining_work)
    
    # Composite score: urgency_bias ensures violating tasks always win; others are positive penalties
    score = (
        1.0  # base offset to keep non-urgent scores positive
        + urgency_bias
        + 0.22 * energy_penalty
        + 0.15 * norm_latency
        + 0.10 * wait_penalty
        + 0.08 * norm_uncertainty_boost
        + 0.05 * norm_uncertainty
        + 0.04 * norm_remaining_work
    )
    
    # Final sanitization: clamp and replace NaN/inf
    score = np.clip(score, -1e12, 1e12)
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    
    return score
