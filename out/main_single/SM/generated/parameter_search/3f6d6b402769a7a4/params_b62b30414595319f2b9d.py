import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with smooth feasibility blending (replacing hard gates),
    joint cross-feature normalization for slack/rank/work, and adaptive clamping.
    
    Key structural improvements:
      - Replaced binary `slk >= 0` gating with smooth sigmoid blending: all components now
        smoothly transition weight based on slack feasibility — enabling gradient-aware CMA-ES
        optimization while preserving hard deadline respect at extremes.
      - Introduced *joint normalization* over [slack, upward_rank, remaining_work] using shared
        percentile bounds — eliminates scale misalignment in successor-release interaction and
        improves robustness across heterogeneous DAGs.
      - Removed `finfo_max_scale`; replaced final safeguard with symmetric `np.clip` using tunable
        `score_clip_bound` — ensures deterministic finite output without machine-epsilon dependency.
      - All normalizations now use identical `percentile_normalize()` logic for full consistency.
    """
    eps = 2.585492049671335e-09
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)
    joint_features = np.stack([slk, rank, work], axis=0)
    p_low_joint = np.percentile(joint_features, 14.907895069143645, axis=1)
    p_high_joint = np.percentile(joint_features, 86.45847264729917, axis=1)
    rng_joint = p_high_joint - p_low_joint
    rng_safe_joint = np.where(np.isfinite(rng_joint) & (rng_joint > eps), rng_joint, eps)
    center_joint = (p_low_joint + p_high_joint) / 2.0
    slk_norm = (slk - center_joint[0]) / (rng_safe_joint[0] + eps)
    rank_norm = (rank - center_joint[1]) / (rng_safe_joint[1] + eps)
    work_norm = (work - center_joint[2]) / (rng_safe_joint[2] + eps)

    def percentile_normalize(x):
        x = np.asarray(x)
        p_low = np.percentile(x, 14.907895069143645)
        p_high = np.percentile(x, 86.45847264729917)
        rng = p_high - p_low
        rng_safe = np.where(np.isfinite(rng) & (rng > eps), rng, eps)
        center = (p_low + p_high) / 2.0
        return (x - center) / (rng_safe + eps)
    duration = exec_t + comm_t
    feasibility_mask = 1.0 / (1.0 + np.exp(-7.820718889465581 * slk))
    slack_penalty = np.where(slk < 0, 8.796622118192658 * np.abs(slk_norm) ** 2, -5.287828112454052 * np.abs(slk_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.8329276998304476 * percentile_normalize(inv_energy + eps)
    energy_score = (1.0 - feasibility_mask) * 0.0 + feasibility_mask * energy_score
    rank_score = -1.0205721159742451 * percentile_normalize(rank_norm + eps)
    rank_score = (1.0 - feasibility_mask) * 0.0 + feasibility_mask * rank_score
    urgency_signal = np.clip(-slk, 0.0, np.inf)
    urgency_norm = percentile_normalize(urgency_signal + eps)
    successor_score = -0.09460426941900005 * work_norm * urgency_norm * (1.0 - feasibility_mask)
    dur_norm = percentile_normalize(duration + eps)
    uncert_norm = percentile_normalize(uncert + eps)
    denom = dur_norm + 0.016138867376600894 * uncert_norm + eps
    dur_uncert_blend = (1.0 + 0.016138867376600894 * uncert_norm) / denom
    dur_score = percentile_normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.10889503273312437 * wait)
    wait_score = -percentile_normalize(wait_sat + eps)
    score = slack_penalty + energy_score + rank_score + successor_score + dur_score + wait_score
    score = np.clip(score, -73661.85100383496, 73661.85100383496)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
