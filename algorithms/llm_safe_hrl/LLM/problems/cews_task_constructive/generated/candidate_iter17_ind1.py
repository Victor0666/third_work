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
    Self-evolved priority rule v2: Hybrid adaptive urgency + criticality-energy synergy + robust starvation control.
    
    Key integrations:
    - Adaptive urgency bands (Parent 2) with quantile-based thresholds, but enhanced by Parent 1's latency-critical gating
      (only activate severe urgency for tasks with slack < median_slack) to avoid overreaction in relaxed regimes.
    - MAD normalization (Parent 2) retained for stability, but augmented with Parent 1's trimmed-mean fallback for N=1/2 cases.
    - Criticality-energy synergy refined: uses slack-attenuated upward_rank *and* uncertainty-gated energy efficiency,
      where energy penalty amplification is activated only under tight slack (< 0.5 * median_duration) *and* high uncertainty.
    - Starvation boost redesigned: joint condition on (normalized wait > 0.7) AND (remaining_work > 0.3 * median_rw)
      AND (slack >= 0) AND (upward_rank < median_ur), capped at 1.2x wait ratio — preventing CP starvation while
      ensuring fairness only for non-urgent, low-criticality, long-waiting tasks.
    - Communication pressure now weighted by uncertainty *and* relative comm overhead, gated by violation/tightness,
      with stronger penalty for violated tasks (x2.5) and moderate penalty for tight tasks (x1.2).
    - Final convex combination weights tuned to DDL-hardness (0.54), synergy leverage (0.28), comm-awareness (0.09),
      risk-energy (0.06), and starvation mitigation (0.03).
    """
    eps = 1e-08
    min_exec_time = np.asarray(min_exec_time, dtype=float).copy()
    min_comm_time = np.asarray(min_comm_time, dtype=float).copy()
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float).copy()
    slack = np.asarray(slack, dtype=float).copy()
    upward_rank = np.asarray(upward_rank, dtype=float).copy()
    remaining_work = np.asarray(remaining_work, dtype=float).copy()
    ready_wait_time = np.asarray(ready_wait_time, dtype=float).copy()
    uncertainty = np.asarray(uncertainty, dtype=float).copy()
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    
    # Robust normalization supporting small-N: use MAD when N>2, trimmed-mean fallback otherwise
    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        elif N == 2:
            # Trimmed-mean fallback for tiny N
            x_sorted = np.sort(x)
            center = np.mean(x_sorted)
            rng = np.max(x_sorted) - np.min(x_sorted) + eps
            return np.clip((x - center) / rng, -10.0, 10.0)
        else:
            # MAD normalization for larger N
            center = np.median(x)
            mad = np.median(np.abs(x - center)) + eps
            normed = (x - center) / mad
            return np.clip(normed, -10.0, 10.0)
    
    task_duration = np.maximum(min_exec_time + min_comm_time, eps)
    rel_slack = np.divide(slack, task_duration, out=np.zeros_like(slack), where=task_duration != 0)
    
    # Adaptive urgency bands using quantiles of rel_slack
    if N > 1:
        q20, q50, q80 = np.quantile(rel_slack, [0.2, 0.5, 0.8], method='midpoint')
    else:
        q20 = q50 = q80 = rel_slack[0]
    
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (rel_slack < q50)
    relaxed_mask = ~violated_mask & (rel_slack >= q50)
    
    # Urgency penalty: clipped linear ramp per band, gated by global latency-critical condition (slack < median_slack)
    median_slack = np.median(slack) if N > 0 else 0.0
    latency_critical_mask = (slack < median_slack).astype(float)
    
    urgency_penalty = np.zeros_like(slack)
    urgency_penalty[violated_mask] = np.clip(-slack[violated_mask] / (task_duration[violated_mask] + eps), 0.0, 6.0)
    urgency_penalty[tight_mask] = np.clip(q50 - rel_slack[tight_mask], 0.0, 3.5)
    urgency_penalty[relaxed_mask] = np.clip(q80 - rel_slack[relaxed_mask], 0.0, 1.0)
    urgency_penalty = urgency_penalty * latency_critical_mask
    
    # Criticality attenuation: dampen upward_rank only when slack > median_slack
    slack_factor = np.clip(1.0 - np.maximum(0.0, slack - median_slack) / (task_duration + eps), 0.1, 1.0)
    dampened_ur = upward_rank * slack_factor
    
    # Synergy: dampened_ur * duration / energy, amplified under violation
    base_synergy = dampened_ur * task_duration / (min_incremental_energy + eps)
    synergy_amplifier = np.where(violated_mask, 1.0 + 0.7 * uncertainty, 1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, eps, 1e8)
    
    # Risk-weighted energy: exponentiated penalty only under tight slack AND high uncertainty
    tight_slack_mask = (slack < 0.5 * np.median(task_duration) if N > 0 else 0.0).astype(float)
    high_uncert_mask = (uncertainty > 0.3).astype(float)
    rel_slack_distance = np.maximum(0.0, -slack) / (task_duration + eps)
    energy_exponent = 1.2
    risk_weighted_energy = min_incremental_energy * np.power(
        1.0 + uncertainty * rel_slack_distance * tight_slack_mask * high_uncert_mask + eps,
        energy_exponent
    )
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e9)
    
    # Communication pressure: scaled by uncertainty and comm overhead, gated by violation/tightness
    comm_overhead_ratio = min_comm_time / (task_duration + eps)
    comm_pressure = comm_overhead_ratio * (1.0 + 0.5 * uncertainty)
    comm_pressure = np.where(violated_mask, comm_pressure * 2.5,
                            np.where(tight_mask, comm_pressure * 1.2, comm_pressure * 0.3))
    
    # Starvation boost: only for non-urgent, low-criticality, long-waiting, sufficient-work tasks
    median_task_dur = np.median(task_duration) + eps
    norm_wait_ratio = np.clip(ready_wait_time / median_task_dur, 0.0, 5.0)
    median_rw = np.median(remaining_work) + eps
    rw_normalized = np.clip(remaining_work / median_rw, 0.01, 100.0)
    median_ur = np.median(upward_rank) + eps
    is_starvable = (
        (slack >= 0.0) &
        (norm_wait_ratio > 0.7) &
        (rw_normalized > 0.3) &
        (upward_rank < median_ur)
    )
    starvation_boost = np.where(
        is_starvable,
        np.clip(norm_wait_ratio * (1.0 + 0.2 * np.minimum(60.0, -slack + 60.0) / 60.0), 0.0, 1.2),
        0.0
    )
    
    # Normalize all components robustly
    norm_urgency = robust_normalize(urgency_penalty)
    norm_synergy = robust_normalize(latency_crit_synergy)
    norm_energy = robust_normalize(risk_weighted_energy)
    norm_comm = robust_normalize(comm_pressure)
    norm_starvation = robust_normalize(starvation_boost)
    
    # Final convex combination: prioritize DDL compliance, then synergistic efficiency, then fairness
    score = (
        0.54 * norm_urgency +
        -0.28 * norm_synergy +
        0.09 * norm_comm +
        0.06 * norm_energy +
        0.03 * norm_starvation
    )
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
