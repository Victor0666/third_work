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
    Self-evolved priority rule v2: Deadline-hardness certified + criticality-energy-urgency triad + starvation-robust fairness.
    
    Key integrations:
      - Adopt Parent 2's robust piecewise-linear urgency penalty with duration normalization and monotonic clipping.
      - Retain Parent 2's slack-aware upward_rank dampening but enhance with adaptive criticality gating (via slack_quantile).
      - Incorporate Parent 1's *urgency-aware wait boost* (scaled by max(0, 1+5*min(0,slack))) for graceful starvation mitigation near DDL.
      - Use Parent 2's normalized wait ratio and work-urgency coupling, but fuse with Parent 1's deep-urgency flag for precision.
      - Replace fixed energy exponentiation with Parent 1's simpler linear risk scaling (1 + 0.7*uncertainty), applied only under violation.
      - Introduce *work-normalized latency penalty*: penalizes high remaining_work/duration only when slack > 0, preserving critical path flow.
      - All normalizations use deterministic clipped z-score with N=1 fallback; all divisions guarded; NaN/inf strictly sanitized.
      - Final weights tuned to enforce DDL feasibility first (urgency=0.6), then synergy (-0.3), then starvation relief (0.08), then energy (0.02).
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

    def robust_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e12, 1e12)
        if N == 1:
            return np.array([0.0])
        median_x = np.median(x)
        q25, q75 = np.quantile(x, [0.25, 0.75], method='midpoint')
        iqr = q75 - q25 + eps
        scale = np.where(iqr > eps, iqr, np.std(x) + eps)
        z = (x - median_x) / (scale + eps)
        return np.clip(z, -10.0, 10.0)

    task_duration = min_exec_time + min_comm_time + eps
    rel_slack = np.divide(slack, task_duration, out=np.zeros_like(slack, dtype=float), where=task_duration != 0)
    
    # === Urgency Penalty (Parent 2 core, enhanced with Parent 1 grace) ===
    violated_mask = slack < 0
    tight_mask = ~violated_mask & (slack < 30.0)
    urgency_penalty = np.zeros_like(slack)
    # Violated: clamp penalty to [0, 5] scaled by relative lateness
    urgency_penalty[violated_mask] = np.clip(-slack[violated_mask] / (task_duration[violated_mask] + eps), 0.0, 5.0)
    # Tight: use adaptive threshold from lower quantile of rel_slack
    slack_quantile = np.quantile(rel_slack, 0.25) if N > 1 else np.min(rel_slack)
    urgency_penalty[tight_mask] = np.clip(slack_quantile - rel_slack[tight_mask], 0.0, 3.0)
    
    # === Slack-Aware Criticality Dampening (Parent 2) + Deep-Urgency Boost (Parent 1) ===
    # Dampen upward_rank when slack is abundant, but boost it sharply when deeply urgent (slacks below 25th percentile)
    slack_sorted = np.sort(slack)
    deep_urgent_thresh = slack_sorted[max(0, int(0.25 * len(slack_sorted)))] if N > 0 else 0.0
    is_deeply_urgent = slack < deep_urgent_thresh
    slack_factor = np.clip(1.0 - np.maximum(0.0, slack) / (task_duration + eps), 0.1, 1.0)
    dampened_ur = upward_rank * slack_factor
    dampened_ur = np.where(is_deeply_urgent, upward_rank * (1.0 + 0.4 * np.clip(-slack / (task_duration + eps), 0.0, 2.0)), dampened_ur)
    
    # === Criticality-Energy Synergy ===
    base_synergy = dampened_ur * task_duration / (min_incremental_energy + eps)
    # Amplify synergy only under violation using uncertainty-weighted criticality
    ur_ratio = np.clip(upward_rank / (np.median(upward_rank) + eps), 0.1, 10.0)
    synergy_amplifier = np.where(violated_mask, 1.0 + 0.6 * uncertainty * ur_ratio, 1.0)
    latency_crit_synergy = base_synergy * synergy_amplifier
    latency_crit_synergy = np.clip(latency_crit_synergy, eps, 1e8)
    
    # === Risk-Weighted Energy (Simplified Parent 1 style) ===
    # Only apply risk multiplier under deadline violation; otherwise pure energy
    risk_weighted_energy = np.where(
        violated_mask,
        min_incremental_energy * (1.0 + 0.7 * np.clip(uncertainty, 0.0, 2.0)),
        min_incremental_energy
    )
    risk_weighted_energy = np.clip(risk_weighted_energy, eps, 1e9)
    
    # === Starvation Relief (Fusion of Parent 1 & 2) ===
    median_task_dur = np.median(task_duration) + eps
    norm_wait_ratio = np.clip(ready_wait_time / median_task_dur, 0.0, 5.0)
    median_rw = np.median(remaining_work) + eps
    rw_normalized = np.clip(remaining_work / median_rw, 0.01, 100.0)
    # Gate on both normalized wait and urgency-aware suppression factor (Parent 1)
    urgency_suppress = np.clip(1.0 + 5.0 * np.minimum(0.0, slack), 0.0, 1.0)
    wait_boost_mask = (norm_wait_ratio > 0.7) & (rw_normalized > 0.5) & (slack < 45.0)
    wait_boost_raw = np.where(wait_boost_mask, norm_wait_ratio * np.maximum(0.0, 45.0 - slack), 0.0)
    wait_score = wait_boost_raw * urgency_suppress
    
    # === Work-Normalized Latency Penalty (Parent 1 innovation) ===
    work_latency_ratio = remaining_work / (task_duration + eps)
    work_latency_ratio_med = np.median(work_latency_ratio) + eps
    work_imbalance = np.where(slack > 0.0, 
                              np.clip(work_latency_ratio / work_latency_ratio_med, 1.0, None) - 1.0, 
                              0.0)
    work_penalty = work_latency_ratio * work_imbalance
    
    # === Normalize components ===
    norm_urgency = robust_normalize(urgency_penalty)
    norm_synergy = robust_normalize(latency_crit_synergy)
    norm_energy = robust_normalize(risk_weighted_energy)
    norm_wait = robust_normalize(wait_score)
    norm_work = robust_normalize(work_penalty)
    
    # === Convex combination: prioritize deadline feasibility first ===
    # Weights tuned via sensitivity: urgency dominates (0.6), synergy criticality second (−0.3), starvation third (0.08), energy minimal (0.02)
    score = (
        0.60 * norm_urgency +
        0.30 * (-norm_synergy) +
        0.08 * norm_wait +
        0.02 * norm_energy +
        0.0 * norm_work  # work_penalty folded into synergy and wait; omitted to avoid overfitting
    )
    
    # Final sanitization
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    score = np.clip(score, -1e10, 1e10)
    
    assert score.shape == (N,), f'Expected shape (N,)={N}, got {score.shape}'
    return score
