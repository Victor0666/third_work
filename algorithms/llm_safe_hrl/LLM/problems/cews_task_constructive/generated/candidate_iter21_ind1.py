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

    '''
    v2 priority rule: Urgency-locked critical-path steering + adaptive energy gating + latency-aware starvation relief + risk-concentrated uncertainty penalty.

    Key advances over v1:
      - Urgency lockstep upgraded to *deadline proximity tiering*: slack <= 0 → fixed min score (-1e15); 
        0 < slack <= median(slack) → tiered urgency (linearly decreasing score), improving gradient near deadline.
      - Energy gating refined using *critical-path slack density*: slack / (upward_rank * (min_exec_time + min_comm_time + eps)), 
        with dynamic threshold = 0.75 * percentile(90, cp_slack_density) → adapts to workflow tightness.
      - Starvation relief now combines *wait-time efficiency* AND *latency pressure*, gated by both work volume AND slack deficit:
        only tasks with high remaining_work AND negative/low slack receive relief; avoids penalizing lightweight urgent tasks.
      - Uncertainty penalty fully decoupled from safe region: applied only when slack < 0 OR (slack <= 0.1 * median(|slack|)), 
        and scaled by upward_rank * uncertainty / (duration + eps) → concentrates risk mitigation where it matters.
      - All normalizations use robust percentile-clipped min-max (1%/99%) with degeneracy fallback; no z-score (reduces outlier sensitivity drift).
      - Final weights sum to 1.0: urgency_tier (0.40) > energy_gated (0.22) > progress_velocity (0.16) > starvation_relief (0.14) > risk_penalty (0.08).
    '''
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64)
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slack = np.asarray(slack, dtype=np.float64)
    upward_rank = np.asarray(upward_rank, dtype=np.float64)
    remaining_work = np.asarray(remaining_work, dtype=np.float64)
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    
    # Safeguard inputs against NaN/inf
    inputs = [min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty]
    for arr in inputs:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)
    
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p01 = np.percentile(x, 1.0)
        p99 = np.percentile(x, 99.0)
        x_clipped = np.clip(x, p01, p99)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # --- Urgency tiering: hard lockstep + linear proximity gradient ---
    is_urgent = (slack <= 0.0).astype(np.float64)
    is_proximal = ((slack > 0.0) & (slack <= np.median(np.abs(slack)) + eps)).astype(np.float64)
    urgency_score = np.full(N, -1000000000000000.0, dtype=np.float64)
    # Proximal tier: linear decay from -1e14 (most urgent) to 0 (least proximal)
    proximal_base = -100000000000000.0
    slack_range = np.maximum(np.median(np.abs(slack)) - np.min(slack[slack > 0]), eps)
    proximal_decay = np.divide(proximal_base, slack_range + eps, out=np.zeros_like(slack), where=slack_range != 0)
    proximal_score = proximal_base + (slack * proximal_decay)
    proximal_score = np.where(slack > 0.0, proximal_score, 0.0)
    proximal_score = np.where(is_proximal, proximal_score, 0.0)
    
    # --- Critical-path slack density for adaptive energy gating ---
    duration = min_exec_time + min_comm_time + eps
    cp_slack_density = np.divide(slack, upward_rank * duration + eps, out=np.zeros_like(slack), where=(upward_rank * duration + eps) != 0)
    cp_slack_density = np.where(np.isfinite(cp_slack_density), cp_slack_density, 0.0)
    # Dynamic threshold: 0.75 * 90th percentile of non-negative densities (focuses on constrained region)
    valid_density = cp_slack_density[cp_slack_density >= 0]
    dyn_threshold = 0.75 * (np.percentile(valid_density, 90.0) if len(valid_density) > 0 else 1.0)
    energy_gate = (cp_slack_density <= dyn_threshold).astype(np.float64)
    energy_density = np.divide(min_incremental_energy, duration + eps, out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_gate
    
    # --- Progress velocity: critical-path importance per unit duration ---
    progress_velocity = upward_rank / (duration + eps)
    norm_progress_velocity = robust_minmax_norm(progress_velocity)
    
    # --- Starvation relief: only for high-work AND latency-pressured tasks ---
    work_quantile_70 = np.quantile(remaining_work, 0.7, method='midpoint') + eps
    latency_pressure = np.clip(-slack, 0.0, np.inf) / (np.median(duration) + eps)
    starvation_condition = (remaining_work >= work_quantile_70) & (latency_pressure > 0.15)
    wait_efficiency = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    wait_efficiency = np.where(np.isfinite(wait_efficiency), wait_efficiency, 0.0)
    norm_wait_efficiency = robust_minmax_norm(wait_efficiency)
    starvation_relief = norm_wait_efficiency * starvation_condition.astype(np.float64)
    
    # --- Risk-concentrated uncertainty penalty: only under deadline stress ---
    # Activate when slack < 0 OR slack <= 10% of median |slack| (early warning zone)
    median_abs_slack = np.median(np.abs(slack)) + eps
    risk_activation = (slack < 0.0) | (slack <= 0.1 * median_abs_slack)
    risk_penalty_base = uncertainty * upward_rank / (duration + eps)
    risk_penalty = np.where(risk_activation, risk_penalty_base, 0.0)
    risk_penalty = np.clip(risk_penalty, 0.0, 10000.0)
    norm_risk_penalty = robust_minmax_norm(risk_penalty)
    
    # --- Weighted composite score ---
    # Non-urgent branch uses all components; urgent/proximal override dominates
    base_score = (
        0.22 * energy_penalty +
        0.16 * (-norm_progress_velocity) +
        0.14 * starvation_relief +
        0.08 * norm_risk_penalty
    )
    # Urgency dominates: fixed min score for urgent, tiered for proximal, base otherwise
    score = np.where(is_urgent, urgency_score,
                     np.where(is_proximal, proximal_score, base_score))
    
    # Final clamping and sanitization
    score = np.clip(score, -1000000000000000.0, 1000000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000000.0, posinf=1000000000000000.0, neginf=-1000000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
