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
    v2 priority rule: Unified urgency-energy-criticality with starvation-aware risk gating.
    
    Key self-evolution improvements:
    - Replaces slack bands with smooth, differentiable urgency scaling: exp(-slack / (duration + eps))
      to preserve gradient-like deadline pressure without hard thresholds.
    - Restores linear uncertainty coupling in energy density: energy / (duration * (1 + uncertainty)),
      avoiding quadratic suppression that harmed energy-efficiency tradeoff under tight deadlines.
    - Inverts starvation logic: wait relief now activates *only* when slack <= median_slack AND
      upward_rank > q75 — directly targeting high-criticality tasks suffering long waits under pressure.
    - Introduces "risk-weighted critical path load": (remaining_work * upward_rank) / (1 + uncertainty),
      biasing consolidation of uncertain but critical sub-DAGs on stable resources.
    - Uses robust median-based slack normalization instead of percentile bands for smoother behavior at small N.
    - All divisions guarded by eps; all NaN/inf replaced deterministically; no in-place mutation.
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
    
    # Sanitize inputs: replace NaN/inf with safe finite defaults
    inputs = [min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, 
              remaining_work, ready_wait_time, uncertainty]
    cleaned = [np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) for arr in inputs]
    min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, \
        remaining_work, ready_wait_time, uncertainty = cleaned
    
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p05 = np.percentile(x, 5.0)
        p95 = np.percentile(x, 95.0)
        x_clipped = np.clip(x, p05, p95)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    # Compute duration and normalized urgency signal: smooth exponential decay with slack
    duration = min_exec_time + min_comm_time + eps
    slack_clean = np.nan_to_num(slack, nan=0.0, posinf=0.0, neginf=0.0)
    median_slack = np.median(slack_clean) if N > 0 else 0.0
    # Urgency: higher when slack is negative or near zero → lower score
    urgency_scale = np.exp(-np.clip(slack_clean, -100.0, 100.0) / (duration + eps))
    urgency_scale = np.where(np.isfinite(urgency_scale), urgency_scale, 0.0)
    
    # Risk-adjusted energy density: linear uncertainty coupling, not quadratic
    effective_duration = duration * (1.0 + uncertainty + eps)
    risk_adj_energy_density = np.divide(min_incremental_energy, effective_duration, 
                                        out=np.zeros_like(min_incremental_energy), 
                                        where=effective_duration != 0)
    risk_adj_energy_density = np.where(np.isfinite(risk_adj_energy_density), risk_adj_energy_density, 0.0)
    
    # Critical-path workload density: reward high-rank, high-work tasks with low uncertainty
    cp_load = (remaining_work * (upward_rank + eps)) / (1.0 + uncertainty + eps)
    cp_load = np.where(np.isfinite(cp_load), cp_load, 0.0)
    
    # Starvation relief: only for high-criticality tasks under deadline pressure
    q75_rank = np.percentile(upward_rank, 75) if N > 0 else 0.0
    high_rank_mask = (upward_rank > q75_rank).astype(np.float64)
    tight_slack_mask = (slack_clean <= median_slack).astype(np.float64)
    # Wait relief scaled by uncertainty: longer waits matter more when uncertainty is high
    wait_relief_raw = np.divide(ready_wait_time, duration + eps, 
                                out=np.zeros_like(ready_wait_time), 
                                where=duration + eps != 0)
    wait_relief_raw = np.where(np.isfinite(wait_relief_raw), wait_relief_raw, 0.0)
    gated_wait_relief = wait_relief_raw * tight_slack_mask * high_rank_mask * (1.0 + uncertainty)
    
    # Normalize components
    norm_urgency = robust_minmax_norm(urgency_scale)
    norm_energy = robust_minmax_norm(risk_adj_energy_density)
    norm_cp_load = robust_minmax_norm(cp_load)
    norm_wait = robust_minmax_norm(gated_wait_relief)
    
    # Final score: minimize = higher priority; urgency dominates, then energy, then critical load, then fairness
    # Negative urgency term ensures urgent tasks get lowest score
    base_score = (
        -0.55 * norm_urgency +           # dominant: strong pull toward urgent tasks
         0.25 * norm_energy +            # secondary: prefer low-risk-energy options
         0.15 * norm_cp_load +           # tertiary: consolidate heavy critical work
         0.05 * norm_wait                # minimal: only relieve starvation where needed
    )
    
    # Ensure urgent tasks (slack <= 0) get absolute highest priority
    is_urgent = (slack_clean <= 0.0).astype(np.float64)
    score = np.where(is_urgent, -1e15, base_score)
    
    # Clip and sanitize final score
    score = np.clip(score, -1e15, 1e15)
    score = np.nan_to_num(score, nan=1e15, posinf=1e15, neginf=-1e15)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
