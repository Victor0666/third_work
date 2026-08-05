import numpy as np

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
    v2 priority rule: Hybrid urgency-energy-criticality with adaptive starvation relief,
                     feasibility-aware energy density clipping, and slack-proportional progress pressure.
    
    Key innovations:
    - Combines Parent 2's smooth urgency scaling (exp(-slack/duration)) with Parent 1's
      slack-proportional progress pressure (upward_rank / (duration * (1 + |slack|/max_abs_slack)))
      for robust deadline pressure at all slack regimes.
    - Uses Parent 1's feasibility-aware energy density clipping (99th percentile cap) to suppress outliers,
      while retaining Parent 2's linear uncertainty coupling: energy / (duration * (1 + uncertainty)).
    - Gates wait relief using Parent 2's logic (high-rank AND tight-slack), but scales it by
      normalized ready_wait_time * (1 + uncertainty) for risk-aware fairness.
    - Introduces unified criticality-weighted load: (remaining_work * upward_rank) / (1 + uncertainty),
      normalized robustly and weighted to consolidate uncertain critical sub-DAGs.
    - All operations guarded against NaN/inf/zero; deterministic; no in-place mutation.
    """
    eps = 1e-08
    # Defensive copying and cleaning
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
    
    # Clean all inputs: replace NaN/inf with 0.0
    inputs = [min_exec_time, min_comm_time, min_incremental_energy, slack, 
              upward_rank, remaining_work, ready_wait_time, uncertainty]
    cleaned = [np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) for arr in inputs]
    min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, \
        remaining_work, ready_wait_time, uncertainty = cleaned
    
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
    
    # Duration and derived quantities
    duration = min_exec_time + min_comm_time + eps
    slack_clean = np.nan_to_num(slack, nan=0.0, posinf=0.0, neginf=0.0)
    max_abs_slack = np.max(np.abs(slack_clean)) + eps
    
    # Smooth urgency: exp(-slack/duration) — Parent 2 strength
    urgency_scale = np.exp(-np.clip(slack_clean, -100.0, 100.0) / (duration + eps))
    urgency_scale = np.where(np.isfinite(urgency_scale), urgency_scale, 0.0)
    
    # Slack-proportional progress pressure: Parent 1 insight, adapted
    progress_pressure = np.divide(
        upward_rank, 
        duration * (1.0 + np.abs(slack_clean) / max_abs_slack + eps),
        out=np.zeros_like(upward_rank), 
        where=duration != 0
    )
    progress_pressure = np.where(np.isfinite(progress_pressure), progress_pressure, 0.0)
    
    # Linear uncertainty-coupled energy density: Parent 2 (avoids quadratic suppression)
    effective_duration = duration * (1.0 + uncertainty + eps)
    risk_adj_energy_density = np.divide(
        min_incremental_energy,
        effective_duration,
        out=np.zeros_like(min_incremental_energy),
        where=effective_duration != 0
    )
    risk_adj_energy_density = np.where(np.isfinite(risk_adj_energy_density), risk_adj_energy_density, 0.0)
    
    # Feasibility-aware clipping: Parent 1's 99th percentile cap on energy density
    energy_p99 = np.percentile(risk_adj_energy_density, 99.0, method='midpoint') + eps
    energy_density_clipped = np.clip(risk_adj_energy_density, 0.0, energy_p99)
    
    # Critical path load: Parent 2's (work * rank) / (1 + uncertainty), risk-weighted
    cp_load = (remaining_work * (upward_rank + eps)) / (1.0 + uncertainty + eps)
    cp_load = np.where(np.isfinite(cp_load), cp_load, 0.0)
    
    # Wait relief gating: Parent 2's logic (tight slack AND high rank)
    median_slack = np.median(slack_clean) if N > 0 else 0.0
    q75_rank = np.percentile(upward_rank, 75) if N > 0 else 0.0
    tight_slack_mask = (slack_clean <= median_slack).astype(np.float64)
    high_rank_mask = (upward_rank > q75_rank).astype(np.float64)
    wait_relief_raw = np.divide(
        ready_wait_time, 
        duration + eps, 
        out=np.zeros_like(ready_wait_time), 
        where=duration + eps != 0
    )
    wait_relief_raw = np.where(np.isfinite(wait_relief_raw), wait_relief_raw, 0.0)
    gated_wait_relief = wait_relief_raw * tight_slack_mask * high_rank_mask * (1.0 + uncertainty)
    
    # Normalize components
    norm_urgency = robust_minmax_norm(urgency_scale)
    norm_progress = robust_minmax_norm(progress_pressure)
    norm_energy = robust_minmax_norm(energy_density_clipped)
    norm_cp_load = robust_minmax_norm(cp_load)
    norm_wait = robust_minmax_norm(gated_wait_relief)
    
    # Final weighted score: smaller = better
    # Prioritize urgency (0.45), energy efficiency (0.22), critical load (0.16), progress (0.09), wait relief (0.08)
    base_score = (
        0.45 * (1.0 - norm_urgency) +          # Urgency: higher urgency -> lower score
        0.22 * norm_energy +                   # Energy: lower density -> lower score
        0.16 * norm_cp_load +                  # Critical load: consolidate high-load sub-DAGs
        0.09 * (1.0 - norm_progress) +         # Progress pressure: higher pressure -> lower score
        0.08 * norm_wait                       # Wait relief: only when justified
    )
    
    # Hard urgency override: immediately schedule overdue/urgent tasks
    is_urgent = (slack_clean <= 0.0).astype(np.float64)
    urgency_score = np.full(N, -1e15, dtype=np.float64)
    score = np.where(is_urgent, urgency_score, base_score)
    
    # Clip and sanitize
    score = np.clip(score, -1e15, 1e15)
    score = np.nan_to_num(score, nan=1e15, posinf=1e15, neginf=-1e15)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
