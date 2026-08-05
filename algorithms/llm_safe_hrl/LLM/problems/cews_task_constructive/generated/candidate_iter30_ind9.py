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
    v2 priority rule: Slack-monotonic urgency-dominant ranking with criticality-weighted energy fairness.
    
    Key self-evolution improvements:
    - Eliminates dual deadline pressures: uses *only* exp(-slack/duration) for urgency — strictly monotonic,
      dominant when slack ≤ 0, and preserves gradient pressure across all regimes.
    - Replaces percentile clipping with robust soft-clipping: tanh-scaled energy density to suppress outliers
      without distorting relative ordering of critical tasks.
    - Introduces criticality-aware energy fairness: scales energy density by (1 + upward_rank / max_rank),
      biasing lower-energy assignment to high-rank tasks without overriding urgency.
    - Removes wait-relief gating — instead integrates normalized ready_wait_time as a *small linear penalty*
      only for non-urgent tasks, preventing starvation while avoiding priority inversion.
    - Uses deterministic, zero-division-safe robust normalization with 5–95 percentile bounds and fallback.
    - All terms designed so that urgency dominates base_score; urgent tasks always get score ≈ -∞.
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
    
    # Clean all inputs deterministically
    inputs = [min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty]
    cleaned = [np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) for arr in inputs]
    min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty = cleaned
    
    def robust_minmax_norm(x):
        if x.size == 0:
            return np.zeros_like(x)
        p05 = np.percentile(x, 5.0, method='midpoint')
        p95 = np.percentile(x, 95.0, method='midpoint')
        x_clipped = np.clip(x, p05, p95)
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x_clipped - x_min) / (x_max - x_min + eps)
    
    duration = min_exec_time + min_comm_time + eps
    slack_clean = slack  # already cleaned
    # Strictly monotonic urgency: dominates all other terms when slack <= 0
    urgency_scale = np.exp(-np.clip(slack_clean, -100.0, 100.0) / (duration + eps))
    urgency_scale = np.where(np.isfinite(urgency_scale), urgency_scale, 0.0)
    
    # Criticality-weighted energy density: higher rank → stronger energy optimization pressure
    max_rank = np.max(upward_rank) + eps if N > 0 else eps
    rank_weight = 1.0 + (upward_rank / max_rank)
    effective_duration = duration * (1.0 + uncertainty + eps)
    risk_adj_energy_density = np.divide(min_incremental_energy, effective_duration, 
                                       out=np.zeros_like(min_incremental_energy), 
                                       where=effective_duration != 0)
    risk_adj_energy_density = np.where(np.isfinite(risk_adj_energy_density), risk_adj_energy_density, 0.0)
    # Soft-clipping via tanh to preserve discriminability while suppressing extremes
    energy_normalized = risk_adj_energy_density / (np.mean(risk_adj_energy_density) + eps)
    energy_soft_clipped = np.tanh(energy_normalized) * (np.max(np.abs(energy_normalized)) + eps)
    
    # Critical path load: consolidate uncertain but high-impact sub-DAGs
    cp_load = remaining_work * (upward_rank + eps) / (1.0 + uncertainty + eps)
    cp_load = np.where(np.isfinite(cp_load), cp_load, 0.0)
    
    # Starvation mitigation: small linear wait penalty *only* for non-urgent tasks
    wait_penalty = np.divide(ready_wait_time, duration + eps, 
                            out=np.zeros_like(ready_wait_time), 
                            where=duration != 0)
    wait_penalty = np.where(np.isfinite(wait_penalty), wait_penalty, 0.0)
    # Apply only where slack > 0 to avoid interfering with urgent scheduling
    wait_mask = (slack_clean > 0.0).astype(np.float64)
    wait_term = wait_penalty * wait_mask * 0.03  # light, bounded influence
    
    # Normalize components
    norm_urgency = robust_minmax_norm(urgency_scale)
    norm_energy = robust_minmax_norm(energy_soft_clipped * rank_weight)
    norm_cp_load = robust_minmax_norm(cp_load)
    
    # Base score: urgency dominates (1.0 - norm_urgency), others are secondary penalties
    base_score = (1.0 - norm_urgency) * 0.75 + norm_energy * 0.15 + norm_cp_load * 0.07 + wait_term
    
    # Hard urgency override: any task with slack <= 0 gets highest priority
    is_urgent = (slack_clean <= 0.0).astype(np.float64)
    urgency_score = np.full(N, -1e15, dtype=np.float64)
    score = np.where(is_urgent, urgency_score, base_score)
    
    # Final safeguard: clip & sanitize
    score = np.clip(score, -1e15, 1e15)
    score = np.nan_to_num(score, nan=1e15, posinf=1e15, neginf=-1e15)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
