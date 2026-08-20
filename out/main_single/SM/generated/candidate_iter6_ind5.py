import numpy as np
RULE_METADATA = {'structure_hash': '3f6d6b402769a7a4594e121a7c64345f20e0c5ffbfd40ab749f01d4316bb52d4', 'parameter_schema_hash': '3be9621274706f8e34df8a1ffa46214b9265fe2157603476f5bfe25f531cd1f4', 'best_parameter_hash': '0cfd3e8beff6ce279e355e1ffd0ab1f92c285d7cec85be4b50135f171b4bf394', 'best_parameters': {'epsilon': 4.047909826509812e-08, 'slack_risk_penalty': 3.4343697789571705, 'slack_urgency_gain': 2.272247798893609, 'energy_efficiency_weight': 0.0669392369326867, 'criticality_weight': 1.6314702457719212, 'duration_uncertainty_ratio': 0.8710636762466666, 'wait_decay_rate': 0.16029287410127158, 'pctl_low': 4.343406086947623, 'pctl_high': 92.77891496681609, 'successor_release_weight': 0.6417134334564195, 'feasibility_blend_steepness': 16.035161051511054, 'score_clip_bound': 1308935.7491024819}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'b034594e90d608355f2568407f1419cbe6f8d629026fabe6e61e27eb7f889144', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

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
    eps = 4.047909826509812e-08
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
    p_low_joint = np.percentile(joint_features, 4.343406086947623, axis=1)
    p_high_joint = np.percentile(joint_features, 92.77891496681609, axis=1)
    rng_joint = p_high_joint - p_low_joint
    rng_safe_joint = np.where(np.isfinite(rng_joint) & (rng_joint > eps), rng_joint, eps)
    center_joint = (p_low_joint + p_high_joint) / 2.0
    slk_norm = (slk - center_joint[0]) / (rng_safe_joint[0] + eps)
    rank_norm = (rank - center_joint[1]) / (rng_safe_joint[1] + eps)
    work_norm = (work - center_joint[2]) / (rng_safe_joint[2] + eps)

    def percentile_normalize(x):
        x = np.asarray(x)
        p_low = np.percentile(x, 4.343406086947623)
        p_high = np.percentile(x, 92.77891496681609)
        rng = p_high - p_low
        rng_safe = np.where(np.isfinite(rng) & (rng > eps), rng, eps)
        center = (p_low + p_high) / 2.0
        return (x - center) / (rng_safe + eps)
    duration = exec_t + comm_t
    feasibility_mask = 1.0 / (1.0 + np.exp(-16.035161051511054 * slk))
    slack_penalty = np.where(slk < 0, 3.4343697789571705 * np.abs(slk_norm) ** 2, -2.272247798893609 * np.abs(slk_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.0669392369326867 * percentile_normalize(inv_energy + eps)
    energy_score = (1.0 - feasibility_mask) * 0.0 + feasibility_mask * energy_score
    rank_score = -1.6314702457719212 * percentile_normalize(rank_norm + eps)
    rank_score = (1.0 - feasibility_mask) * 0.0 + feasibility_mask * rank_score
    urgency_signal = np.clip(-slk, 0.0, np.inf)
    urgency_norm = percentile_normalize(urgency_signal + eps)
    successor_score = -0.6417134334564195 * work_norm * urgency_norm * (1.0 - feasibility_mask)
    dur_norm = percentile_normalize(duration + eps)
    uncert_norm = percentile_normalize(uncert + eps)
    denom = dur_norm + 0.8710636762466666 * uncert_norm + eps
    dur_uncert_blend = (1.0 + 0.8710636762466666 * uncert_norm) / denom
    dur_score = percentile_normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.16029287410127158 * wait)
    wait_score = -percentile_normalize(wait_sat + eps)
    score = slack_penalty + energy_score + rank_score + successor_score + dur_score + wait_score
    score = np.clip(score, -1308935.7491024819, 1308935.7491024819)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
