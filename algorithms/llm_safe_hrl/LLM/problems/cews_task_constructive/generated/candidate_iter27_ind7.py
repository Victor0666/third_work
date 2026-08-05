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
    v3 priority rule: Hard urgency dominance + risk-duration-aware criticality gating +
                      wait-aware fairness with dynamic thresholding + unified monotonic
                      uncertainty scaling + slack-proportional progress pressure +
                      normalized energy density under feasibility-aware clipping.

    Key improvements over v1:
    - Replaces static percentile work_threshold with adaptive quantile (min(5%, max(1, N//10)%))
      to avoid zero-gating on small ready sets.
    - Introduces *feasibility-aware energy density clipping*: caps energy_density at 99th percentile
      of feasible values (not raw) to suppress outlier-driven misranking.
    - Uses *slack-proportional progress pressure*: upward_rank / (duration * (1 + |slack|/max_slack+eps))
      to dynamically amplify pressure near deadline without inversion.
    - Uncertainty coupling now uses *relative slack sensitivity* = clip(1/(|rel_slack|+eps), 0.05, 20)
      and applies only when slack <= 0 OR upward_rank > 0.8*max(upward_rank), avoiding median bias.
    - Adds *duration-penalized criticality*: criticality_gate = clip((0.15 - rel_slack)/0.15, 0, 1)
      tightened for stronger DDL enforcement at tight margins.
    - All normalization uses robust_minmax_norm with consistent 1st/99th percentile clipping;
      no z-score, no sigmoid, no unbounded functions.
    - Final score weights sum to 1.0 and prioritize urgency (0.40), energy (0.23), wait (0.12),
      uncertainty (0.10), progress (0.09), duration-criticality (0.04), and work (0.02).
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
    
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0, method='midpoint')
        p99 = np.percentile(x, 99.0, method='midpoint')
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    is_urgent = (slack <= 0.0).astype(np.float64)
    urgency_score = np.full(N, -1000000000000.0, dtype=np.float64)
    
    duration = min_exec_time + min_comm_time + eps
    risk_adjusted_duration = duration * (1.0 + uncertainty)
    
    # Feasibility-aware energy density: clip outliers *after* division to preserve monotonicity
    energy_density = np.divide(min_incremental_energy, risk_adjusted_duration, 
                              out=np.zeros_like(min_incremental_energy), 
                              where=risk_adjusted_duration != 0)
    energy_density = np.where(np.isfinite(energy_density), energy_density, 0.0)
    # Clip top 1% to suppress VM outlier effects while preserving relative ordering
    energy_p99 = np.percentile(energy_density, 99.0, method='midpoint') + eps
    energy_density_clipped = np.clip(energy_density, 0.0, energy_p99)
    norm_energy_density = robust_minmax_norm(energy_density_clipped)
    
    rel_slack = np.divide(slack, duration, out=np.zeros_like(slack), where=duration != 0)
    rel_slack = np.where(np.isfinite(rel_slack), rel_slack, 0.0)
    
    # Tighter criticality gate: activates earlier (rel_slack <= 0.15) for stronger DDL adherence
    criticality_gate = np.clip((0.15 - np.clip(rel_slack, -np.inf, 0.15)) / 0.15, 0.0, 1.0)
    
    # Adaptive high-rank mask: more robust than median on skewed distributions
    max_upward = np.max(upward_rank) + eps
    high_rank_mask = (upward_rank > 0.8 * max_upward).astype(np.float64)
    energy_penalty_mask = high_rank_mask * criticality_gate
    energy_penalty = norm_energy_density * energy_penalty_mask
    
    # Dynamic work threshold: min(5%, max(1, N//10)%) avoids zero-gating for N<20
    work_quantile = min(5.0, max(1.0, N // 10))
    work_threshold = np.percentile(remaining_work, work_quantile, method='midpoint') + eps
    wait_gate = (remaining_work >= work_threshold).astype(np.float64)
    norm_wait_time = robust_minmax_norm(ready_wait_time)
    wait_penalty = (1.0 - is_urgent) * norm_wait_time * wait_gate
    
    # Slack-proportional progress pressure: amplifies pressure as |slack| shrinks, no inversion
    max_abs_slack = np.max(np.abs(slack)) + eps
    progress_pressure = np.divide(upward_rank, duration * (1.0 + np.abs(slack) / max_abs_slack + eps),
                                  out=np.zeros_like(upward_rank), where=duration != 0)
    progress_pressure = np.where(np.isfinite(progress_pressure), progress_pressure, 0.0)
    norm_progress_pressure = robust_minmax_norm(progress_pressure)
    
    # Unified uncertainty coupling: active only when urgent OR highly ranked (0.8*max)
    uncertainty_active = np.maximum(is_urgent, high_rank_mask)
    abs_rel_slack = np.abs(rel_slack) + eps
    # Relative slack sensitivity bounded tightly to prevent explosion near zero
    slack_sensitivity = np.clip(1.0 / abs_rel_slack, 0.05, 20.0)
    rank_sensitivity = robust_minmax_norm(upward_rank)
    uncertainty_boost = uncertainty * slack_sensitivity * rank_sensitivity * uncertainty_active
    norm_uncertainty_boost = robust_minmax_norm(uncertainty_boost)
    
    # Duration-penalized criticality: higher duration + higher rank → higher penalty
    duration_penalty = (duration / (np.max(duration) + eps)) * robust_minmax_norm(upward_rank)
    
    # Normalize remaining work with robust scaling
    norm_remaining_work = robust_minmax_norm(remaining_work)
    
    # Weighted fusion: all terms monotonic, sum weights = 1.0
    score = (
        0.40 * (1.0 - robust_minmax_norm(np.clip(-slack, 0.0, np.inf))) +
        0.23 * energy_penalty +
        0.12 * wait_penalty +
        0.10 * norm_uncertainty_boost +
        0.09 * (1.0 - norm_progress_pressure) +
        0.04 * duration_penalty +
        0.02 * norm_remaining_work
    )
    
    # Apply hard urgency dominance
    score = np.where(is_urgent, urgency_score, score)
    
    # Final numeric safety
    score = np.clip(score, -1000000000000.0, 1000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000.0, posinf=1000000000000.0, neginf=-1000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
