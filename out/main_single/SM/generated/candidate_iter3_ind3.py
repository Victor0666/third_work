import numpy as np
RULE_METADATA = {'structure_hash': 'ad735003aea1030e2b7d266d2755b19ba7393029194a1611ae833bc4e12cd119', 'parameter_schema_hash': '43047d05f2d490555c0b4a53c121d02ae24f916a8df07ae9c8454ed6c57f03b5', 'best_parameter_hash': '02b8dab0725c503e62b4521c8ada7abf306087f4ea11c0632d8e70e24dc04c7f', 'best_parameters': {'epsilon': 9.123103168170531e-07, 'slack_risk_penalty': 9.0314979403049, 'slack_urgency_gain': 3.5917572454366806, 'energy_efficiency_weight': 0.5116989503344185, 'criticality_weight': 0.026298110114146164, 'duration_uncertainty_ratio': 1.3843903810695921, 'wait_decay_rate': 0.18882589809988326, 'uncertainty_slack_interaction': 0.0011198887710595247, 'ddl_protection_threshold': -0.7055730300617955, 'successor_release_penalty': 1.2578594043930242, 'robust_scale_factor': 1.130232401231624}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'a8825a32eb8ca9e7129ea600a90d9a06137e9a7094ae79c3da6a7887c9f95ff2', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three key structural improvements:
      - DDL protection gate: hard slack threshold disables non-deadline terms when slack < ddl_protection_threshold.
      - Successor-release interaction: penalizes tasks blocking critical-path successors (direct starvation mitigation).
      - Robust conditional activation: all non-slack terms activate only under safe slack margin, ensuring deadline compliance first.
    All components remain finite, deterministic, and shape-strict (N,)."""
    eps = 9.123103168170531e-07
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def robust_normalize(x):
        x = np.asarray(x)
        center = np.median(x)
        dev = np.abs(x - center)
        scale = np.median(dev) if len(dev) > 0 else 1.0
        scale = max(scale * 1.130232401231624, eps)
        return (x - center) / (scale + eps)
    duration = exec_t + comm_t
    slack_norm = robust_normalize(slk)
    slack_penalty = np.where(slk < 0, 9.0314979403049 * slack_norm ** 2, -3.5917572454366806 * np.abs(slack_norm))
    is_safe_slack = slk >= -0.7055730300617955
    inv_energy = 1.0 / (energy + eps)
    energy_score = np.where(is_safe_slack, -0.5116989503344185 * robust_normalize(inv_energy), 0.0)
    rank_active = np.where(slk > -eps, rank, 0.0)
    rank_score = np.where(is_safe_slack, -0.026298110114146164 * robust_normalize(rank_active + eps), 0.0)
    dur_norm = robust_normalize(duration + eps)
    uncert_norm = robust_normalize(uncert + eps)
    dur_uncert_blend = (1.0 + 1.3843903810695921 * uncert_norm) / (dur_norm + eps + 1.3843903810695921 * uncert_norm + eps)
    dur_score = np.where(is_safe_slack, robust_normalize(dur_uncert_blend), 0.0)
    wait_sat = 1.0 - np.exp(-0.18882589809988326 * wait)
    wait_score = np.where(is_safe_slack, -robust_normalize(wait_sat + eps), 0.0)
    slack_stress = np.where((slk < 0) & (slk >= -0.7055730300617955), np.abs(slack_norm), 0.0)
    unc_slack_interaction = 0.0011198887710595247 * uncert_norm * slack_stress
    successor_penalty = np.where(is_safe_slack, 1.2578594043930242 * robust_normalize(work), 0.0)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + successor_penalty
    score = np.nan_to_num(score, nan=np.median(score), posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
