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
    v2 priority rule: Hybrid urgency-lockstep + critical-path energy gating + fairness-aware starvation relief + uncertainty-normalized slack sensitivity.
    
    Key innovations:
    - Hard deadline lockstep preserved: slack <= 0 → fixed min score (-1e15) for deterministic urgency dominance
    - Critical-path energy gating enhanced: uses relative slack margin (slack / (upward_rank * duration)) with adaptive threshold (<= 0.8)
    - Starvation relief via robust wait-efficiency z-score, gated by work-threshold (30%-quantile) and latency pressure
    - Uncertainty coupling refined: uncertainty * max(0, -slack) / (duration + eps) * upward_rank → focuses risk penalty only on late/critical tasks
    - All normalizations use percentile-clipped min-max (1%/99%) for stability under sparse outliers; fallback to zero on degeneracy
    - Final weights sum to 1.0: urgency (0.45) > energy_gated (0.20) > progress_velocity (0.15) > wait_efficiency (0.12) > uncertainty_slack (0.08)
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=np.float64)
    min_comm_time = np.asarray(min_comm_time, dtype=np.float64)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slack = np.asarray(slack, dtype=np.float64)
    upward_rank = np.asarray(upward_rank, dtype=np.float64)
    remaining_work = np.asarray(remaining_work, dtype=np.float64)
    ready_wait_time = np.asarray(ready_wait_time, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    
    # Defensive nan/inf handling
    inputs = [min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty]
    for arr in inputs:
        arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    
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
    
    # 1. Hard deadline lockstep: absolute priority for overdue/urgent tasks
    is_urgent = (slack <= 0.0).astype(np.float64)
    urgency_score = np.full(N, -1000000000000000.0, dtype=np.float64)
    
    # 2. Duration & critical path context
    duration = min_exec_time + min_comm_time + eps
    cp_duration = upward_rank * duration + eps
    
    # 3. Critical-path–aware slack margin (adaptive tightness detection)
    rel_slack_margin = np.divide(slack, cp_duration, out=np.zeros_like(slack), where=cp_duration != 0)
    rel_slack_margin = np.where(np.isfinite(rel_slack_margin), rel_slack_margin, 0.0)
    energy_gate = (rel_slack_margin <= 0.8).astype(np.float64)  # tighter than Parent 2's 1.0 for stronger feasibility pressure
    
    # 4. Energy density: marginal energy per unit critical path time
    energy_density = np.divide(min_incremental_energy, duration + eps, out=np.zeros_like(min_incremental_energy), where=duration + eps != 0)
    norm_energy_density = robust_minmax_norm(energy_density)
    energy_penalty = norm_energy_density * energy_gate
    
    # 5. Progress velocity: high-impact low-latency prioritization
    progress_velocity = upward_rank / (duration + eps)
    norm_progress_velocity = robust_minmax_norm(progress_velocity)
    
    # 6. Fairness-aware starvation relief: wait-efficiency gated by work threshold and urgency
    wait_efficiency = np.divide(ready_wait_time, remaining_work + eps, out=np.zeros_like(ready_wait_time), where=remaining_work + eps != 0)
    wait_efficiency = np.where(np.isfinite(wait_efficiency), wait_efficiency, 0.0)
    work_threshold = np.quantile(remaining_work, 0.3, method='midpoint') + eps
    latency_pressure = np.clip(-slack, 0.0, np.inf) / (np.median(duration) + eps)
    starvation_gate = (remaining_work >= work_threshold).astype(np.float64) * (latency_pressure > 0.1).astype(np.float64)
    norm_wait_efficiency = robust_minmax_norm(wait_efficiency) * starvation_gate
    
    # 7. Uncertainty-slack coupling: focused risk penalty only on late/critical tasks
    unc_slack_penalty = uncertainty * np.clip(-slack, 0.0, np.inf) / (duration + eps) * upward_rank
    unc_slack_penalty = np.clip(unc_slack_penalty, 0.0, 10000.0)
    norm_unc_slack = robust_minmax_norm(unc_slack_penalty)
    
    # 8. Weighted linear combination (weights sum to 1.0)
    score = (
        0.45 * np.where(is_urgent, urgency_score, 0.0) +
        0.20 * energy_penalty +
        0.15 * (-norm_progress_velocity) +
        0.12 * norm_wait_efficiency +
        0.08 * norm_unc_slack
    )
    
    # 9. Apply urgent override and final clamping
    score = np.where(is_urgent, urgency_score, score)
    score = np.clip(score, -1000000000000000.0, 1000000000000000.0)
    score = np.nan_to_num(score, nan=1000000000000000.0, posinf=1000000000000000.0, neginf=-1000000000000000.0)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
