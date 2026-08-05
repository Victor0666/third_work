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
    v4 priority rule: Hard-feasibility-first + criticality-gated energy efficiency +
                      dynamic starvation rescue + uncertainty-calibrated urgency scaling +
                      monotonic deadline dominance with lateness amplification.

    Key innovations:
    - Strict feasibility dominance: violated tasks get deterministic -1e12 boost (not approx -inf) for stable argmin.
    - Criticality-weighted energy penalty: energy term scaled by (upward_rank / max_upward_rank) * (1 + 0.5*uncertainty),
      prioritizing energy reduction on high-criticality, high-uncertainty paths.
    - Dynamic starvation detection: uses wait_ratio > 2.0 AND upward_rank > median_ur + 0.75*IQR(ur) AND slack < 600s,
      with starvation relief proportional to both wait_ratio and normalized ur.
    - Uncertainty-calibrated urgency: inverse-slack magnitude scaled by (1 + 0.8*uncertainty) only for non-violated tasks,
      preserving urgency fidelity while penalizing risky slack estimates.
    - Lateness amplification: linear penalty for slack <= 0, unclipped but bounded by 1e12, ensuring hard constraint enforcement.
    - All norms use robust IQR-based z-clipping with ±6σ bounds; all divisions guarded; all outputs finite & shape-(N,).
    - Layered additive composition with sum-to-1 weights: 0.5 urgency, 0.25 synergy, 0.15 energy, 0.1 starvation.
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

    def robust_zclip(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75], method='midpoint')
        iqr = q75 - q25 + eps
        z = (x - q50) / iqr
        return np.clip(z, -6.0, 6.0)

    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    
    # Hard feasibility dominance: violated tasks get strict top priority
    violated_mask = slack < 0
    feasibility_boost = np.full(N, 0.0, dtype=np.float64)
    feasibility_boost[violated_mask] = -1e12
    
    # Lateness amplification: linear penalty for violated tasks, unclipped but numerically bounded
    lateness_penalty = np.where(violated_mask, -1e12 + 1000.0 * np.abs(slack), 0.0)
    
    # Urgency term: inverse-slack with uncertainty scaling for non-violated tasks only
    inv_slack_raw = np.divide(1.0, np.abs(slack) + eps, out=np.zeros_like(slack), where=np.abs(slack) + eps != 0)
    inv_slack_confidence = 1.0 + 0.8 * uncertainty
    inv_slack = np.where(violated_mask, 0.0, inv_slack_raw * inv_slack_confidence)
    inv_slack = np.clip(inv_slack, 0.01, 500.0)
    
    # Synergy term: criticality-aware latency-efficiency tradeoff
    ur_max = np.max(upward_rank) + eps
    ur_normalized = np.clip(upward_rank / ur_max, 0.1, 1.0)
    base_synergy = ur_normalized * task_duration / (min_incremental_energy + eps)
    synergy_amplifier = np.where(violated_mask, 1.0 + 0.9 * uncertainty, 1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, eps, 1e10)
    
    # Energy term: criticality-weighted and uncertainty-adjusted
    energy_density = np.divide(min_incremental_energy, task_duration, out=np.zeros_like(min_incremental_energy), where=task_duration != 0)
    energy_density_adj = energy_density * ur_normalized * (1.0 + 0.5 * uncertainty)
    energy_density_adj = np.clip(energy_density_adj, eps, 1e10)
    
    # Starvation rescue: dynamic criticality threshold and stricter wait ratio
    if N > 1:
        ur_median = np.median(upward_rank)
        ur_q25, ur_q75 = np.quantile(upward_rank, [0.25, 0.75], method='midpoint')
        ur_iqr = ur_q75 - ur_q25 + eps
        crit_threshold = ur_median + 0.75 * ur_iqr
    else:
        crit_threshold = upward_rank[0]
    wait_ratio = np.divide(ready_wait_time, task_duration, out=np.zeros_like(ready_wait_time), where=task_duration != 0)
    wait_ratio = np.nan_to_num(wait_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    is_starvable = (wait_ratio > 2.0) & (upward_rank >= crit_threshold) & (slack < 600.0)
    starvation_boost = np.where(is_starvable, wait_ratio * ur_normalized * (1.0 + 0.2 * uncertainty), 0.0)
    starvation_boost = np.clip(starvation_boost, 0.0, 3.0)
    
    # Normalize each component independently
    norm_urgency = robust_zclip(-inv_slack)  # higher inv_slack => lower score => higher priority
    norm_synergy = robust_zclip(latency_crit_synergy)
    norm_energy = robust_zclip(energy_density_adj)
    norm_starvation = robust_zclip(-starvation_boost)  # higher boost => lower score => higher priority
    
    # Weighted sum with sum-to-1 weights
    score = (
        0.50 * norm_urgency +
        0.25 * norm_synergy +
        0.15 * norm_energy +
        0.10 * norm_starvation
    )
    
    # Apply feasibility boost and lateness penalty
    score = score + feasibility_boost + lateness_penalty
    
    # Final sanitization
    score = np.nan_to_num(score, nan=1e12, posinf=1e12, neginf=-1e12)
    score = np.clip(score, -1e12, 1e12)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
