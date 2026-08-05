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
    Evolved priority rule v2: Combines adaptive urgency (Parent 2) with robust critical-path alignment and starvation safety (Parent 1).
    
    Key innovations:
      - Adaptive urgency cliff via tanh, but anchored to *normalized slack distance* (slack / (exec+comm+eps)) for scale invariance.
      - Critical-path synergy: upward_rank * remaining_work weighted by risk-aware slack penalty (1 + max(0,-slack)*uncertainty), not just division.
      - Energy fairness: min_incremental_energy normalized by *task's share of total remaining work*, ensuring low-energy tasks aren't penalized when work is small.
      - Starvation boost: only activated when slack > 0 AND uncertainty is below median, scaled by relative wait time and clipped tightly (0–0.25).
      - Robust normalization: uses MAD scaling with median centering (stable for N≥1), fallback to zero for N==1.
      - Final weights enforce strict deadline dominance (urgency: 0.65), critical-path synergy second (-0.22), energy fairness third (0.08), starvation relief fourth (0.03), uncertainty & time as weak regularizers (0.01 each).
      - All operations eps-protected, NaN/inf-clipped, and bounded to finite range.
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
    
    # Robust normalization: median center + MAD scale; safe for N=1
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        center = np.median(x)
        abs_devs = np.abs(x - center)
        mad = np.median(abs_devs) + eps
        z = (x - center) / mad
        return np.clip(z, -8.0, 8.0)
    
    # Task duration baseline for normalization and urgency scaling
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    
    # Adaptive urgency cliff: tanh of normalized slack pressure → smooth, bounded, monotonic
    norm_slack_pressure = -slack / task_duration
    urgency_cliff = np.tanh(norm_slack_pressure * 0.7)
    deadline_urgency = 1.0 + 0.9 * np.maximum(0.0, urgency_cliff)  # dominant term, no binary gating
    
    # Risk-weighted critical-path synergy: high upward_rank AND high remaining_work prioritized under tight slack
    # Penalty factor increases with both negative slack and uncertainty
    slack_risk_factor = np.clip(1.0 + np.maximum(0.0, -slack) * uncertainty, 1.0, 4.0)
    crit_synergy_raw = upward_rank * remaining_work / (slack_risk_factor * (min_incremental_energy + eps))
    crit_synergy_raw = np.clip(crit_synergy_raw, eps, 1e8)
    
    # Energy fairness: normalize incremental energy by task's proportional share of total remaining work
    total_work = np.sum(np.maximum(remaining_work, eps))
    work_share = np.maximum(remaining_work, eps) / (total_work + eps)
    energy_per_share = min_incremental_energy / (work_share + eps)
    energy_per_share = np.clip(energy_per_share, eps, 1e10)
    
    # Starvation boost: only when slack > 0 AND uncertainty is low (below median)
    median_uncertainty = np.median(uncertainty) + eps
    wait_eligible = (slack > 0.0) & (uncertainty < median_uncertainty)
    max_wait_eligible = np.max(ready_wait_time[wait_eligible]) if np.any(wait_eligible) else eps
    wait_boost_raw = np.where(wait_eligible, ready_wait_time / (max_wait_eligible + eps), 0.0)
    wait_penalty = np.clip(wait_boost_raw, 0.0, 0.25)  # tight cap prevents over-prioritization
    
    # Normalize components
    norm_urgency = robust_normalize(deadline_urgency)
    norm_crit = robust_normalize(crit_synergy_raw)
    norm_energy = robust_normalize(energy_per_share)
    norm_wait = robust_normalize(wait_penalty)
    norm_uncertainty = robust_normalize(uncertainty)
    norm_time = robust_normalize(np.sqrt(task_duration))
    
    # Weighted fusion: urgency dominates, then critical path, then energy fairness, then starvation, others weak regularizers
    score = (
        0.65 * norm_urgency
        - 0.22 * norm_crit
        + 0.08 * norm_energy
        + 0.03 * norm_wait
        + 0.01 * norm_uncertainty
        + 0.01 * norm_time
    )
    
    # Final sanitization
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
