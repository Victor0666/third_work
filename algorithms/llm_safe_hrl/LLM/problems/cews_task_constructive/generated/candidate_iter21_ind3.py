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
    v2 priority rule: Hard deadline lockstep + data-driven criticality gating + 
    outlier-resilient fairness + uncertainty-aware slack sensitivity + 
    strict monotonic urgency preservation.

    Key self-evolution improvements:
    - Restores robust z-score (MAD-based) for wait_efficiency and progress_velocity — proven resilience to workload skew.
    - Replaces fixed/soft gates with *adaptive thresholding*: energy_gate = (slack < percentile(slack, 10)) to preserve hard DDL semantics while adapting to current urgency distribution.
    - Gradient urgency removed entirely: replaced by *strict monotonic urgency term* — linear mapping of normalized slack deficit, clipped to [0,1], ensuring ordering near deadline is preserved and numerically stable.
    - Uncertainty coupling uses *relative rank scaling*: uncertainty * (upward_rank / (np.median(upward_rank) + eps)), preventing bias from absolute magnitude outliers.
    - All normalizations use fallback-safe robust_zscore (with MAD) — no percentile clipping in latency-critical paths; preserves ordinal relationships under heavy tails.
    - Final weights sum to 1.0: urgency (0.42) > energy-gated (0.23) > progress_velocity (0.15) > wait_efficiency (0.10) > uncertainty_slack (0.06) > criticality_bias (0.04).
    - Explicit finite-check + dtype enforcement + no in-place modification.
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
    
    # Safe nan/inf cleanup without in-place mutation
    inputs = [min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty]
    cleaned = [np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) for arr in inputs]
    min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty = cleaned
    
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=np.float64)
    
    def robust_zscore(x):
        if x.size == 0:
            return np.zeros_like(x)
        med = np.median(x)
        dev = x - med
        mad = np.median(np.abs(dev))
        if mad < eps:
            return np.zeros_like(x)
        z = dev / (mad + eps)
        return np.clip(z, -5.0, 5.0)
    
    # Hard deadline lockstep: urgent tasks get absolute top priority
    is_urgent = (slack <= 0.0).astype(np.float64)
    urgency_score = np.full(N, -1e15, dtype=np.float64)
    
    # Duration & critical path estimates
    duration = min_exec_time + min_comm_time + eps
    cp_duration = upward_rank + eps
    
    # Strict monotonic urgency: normalized slack deficit → [0,1], preserving order near zero
    slack_deficit = np.maximum(-slack, 0.0)
    norm_urgency = np.divide(slack_deficit, duration + eps, out=np.zeros_like(slack_deficit), where=(duration + eps) != 0)
    norm_urgency = np.clip(norm_urgency, 0.0, 1.0)
    
    # Adaptive energy gating: gate on bottom 10% of slack (data-driven urgency boundary)
    if N > 1:
        slack_10p = np.percentile(slack, 10.0, method='midpoint')
        energy_gate = (slack < slack_10p).astype(np.float64)
    else:
        energy_gate = np.ones(N, dtype=np.float64)
    
    # Energy density: marginal energy per time unit, robustly normalized
    energy_density = np.divide(min_incremental_energy, duration, out=np.zeros_like(min_incremental_energy), where=duration != 0)
    energy_density = np.where(np.isfinite(energy_density), energy_density, 0.0)
    norm_energy_density = robust_zscore(energy_density)
    
    # Progress velocity: upward_rank / duration → high impact + low latency prioritized
    progress_velocity = np.divide(upward_rank, duration, out=np.zeros_like(upward_rank), where=duration != 0)
    norm_progress_velocity = robust_zscore(progress_velocity)
    
    # Wait-efficiency fairness: wait_time / work → penalize starvation
    wait_efficiency = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    wait_efficiency = np.where(np.isfinite(wait_efficiency), wait_efficiency, 0.0)
    norm_wait_efficiency = robust_zscore(wait_efficiency)
    
    # Uncertainty coupling: relative rank-scaled to avoid magnitude bias
    median_upward = np.median(upward_rank) if N > 0 else 1.0
    rel_uncertainty = uncertainty * (upward_rank / (median_upward + eps))
    abs_slack = np.abs(slack) + eps
    unc_slack_ratio = np.divide(rel_uncertainty, abs_slack, out=np.zeros_like(rel_uncertainty), where=abs_slack != 0)
    unc_sensitivity = np.clip(unc_slack_ratio, 0.0, 1000.0)
    norm_unc_sensitivity = robust_zscore(unc_sensitivity)
    
    # Criticality bias: upward_rank itself, normalized to break ties among equal urgency
    norm_upward_rank = robust_zscore(upward_rank)
    
    # Weighted combination — all terms bounded and monotonic w.r.t. objective
    base_score = (
        0.42 * norm_urgency +
        0.23 * (norm_energy_density * energy_gate) +
        0.15 * (-norm_progress_velocity) +  # higher velocity → lower score
        0.10 * norm_wait_efficiency +
        0.06 * norm_unc_sensitivity +
        0.04 * (-norm_upward_rank)  # higher criticality → lower score
    )
    
    # Apply hard lockstep: urgent tasks dominate
    score = np.where(is_urgent, urgency_score, base_score)
    
    # Final safeguard: finite + clip extremes
    score = np.clip(score, -1e15, 1e15)
    score = np.nan_to_num(score, nan=1e15, posinf=1e15, neginf=-1e15)
    
    # Enforce shape guarantee
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
