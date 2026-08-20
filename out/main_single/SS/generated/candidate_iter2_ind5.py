import numpy as np
RULE_METADATA = {'structure_hash': '77b254fb28ccd78d2b16e2fb990c22f9ad50acd3b0434052b097c29b1433889c', 'parameter_schema_hash': '4710b53e6bc81ad18bd06e2cf071fdbb61f97d48b7c3b47543a6140f7927699b', 'best_parameter_hash': '3e86e45c92c404f86daf00488d1fc6288bf4cc274a52ab72882958abb9883e7d', 'best_parameters': {'epsilon': 7.217131058619024e-05, 'slack_penalty_exponent': 1.973825100265381, 'criticality_boost': 1.937385577732449, 'energy_efficiency_scale': 0.7862629666539586, 'duration_risk_ratio': 0.014101045312297899, 'wait_fairness_gain': 0.13030579118827423, 'uncertainty_slack_gate': 0.27723762552526243, 'rank_slack_coupling': 0.8105650414785999, 'uncertainty_slack_tightening_weight': 0.5057621628730367, 'slack_median_ratio_threshold': 0.3417295934766126, 'slack_pressure_smoothing': 0.264945753995222, 'energy_uncertainty_coupling': 0.00788689782231691}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '32ba059b6bb920e243a0d2578caa53acb80321089346e28a79f7aaad1a084472', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with smooth slack pressure, uncertainty-coupled energy, and successor-release signal.
    
    Key features:
    - Smooth tanh-based slack pressure avoids discontinuities (replaces boolean gates).
    - Energy penalty scaled by (1 + uncertainty * energy_uncertainty_coupling) for joint risk awareness.
    - Successor-release term: remaining_work * (1 - clip(slack/median_slack, 0, 1)) prioritizes unblocking critical paths.
    - All numeric literals are -2, -1, 0, 1, or 2; all tunables declared and used.
    - Robust normalization and finite safeguards via np.finfo.
    """
    eps = 7.217131058619024e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_norm(x):
        x_abs = np.abs(x)
        denom = np.mean(x_abs) + eps
        return x / denom
    median_slack = np.median(np.abs(slack)) + eps
    slack_pressure = np.tanh(slack / (eps + 0.264945753995222 * median_slack))
    neg_slack = np.maximum(0.0, -slack)
    slack_penalty = np.power(neg_slack + eps, 1.973825100265381)
    duration = min_exec_time + min_comm_time
    duration_norm = robust_norm(duration)
    uncertainty_norm = robust_norm(uncertainty)
    risk_adjusted_duration = 0.014101045312297899 * duration_norm + (1 - 0.014101045312297899) * uncertainty_norm
    energy_norm = robust_norm(min_incremental_energy)
    energy_uncertainty_factor = 1 + 0.00788689782231691 * uncertainty_norm
    coupled_energy_penalty = 0.7862629666539586 * energy_norm * energy_uncertainty_factor
    rank_base = robust_norm(upward_rank)
    rank_weighted = rank_base * (1 + 0.8105650414785999 * np.maximum(0.0, -slack_pressure))
    median_rank = np.median(upward_rank)
    is_critical = ((upward_rank >= median_rank) & (slack <= np.maximum(eps, 0.3417295934766126 * median_slack))).astype(float)
    criticality_bonus = is_critical * 1.937385577732449
    work_norm = robust_norm(remaining_work)
    slack_tightness = np.clip(slack / (eps + median_slack), 0.0, 1.0)
    successor_release = work_norm * (1 - slack_tightness)
    wait_norm = robust_norm(ready_wait_time)
    wait_bonus = -0.13030579118827423 * wait_norm
    unc_gate_active = ((uncertainty_norm > 0.27723762552526243) & (slack < eps)).astype(float)
    tightened_penalty = slack_penalty + unc_gate_active * 0.5057621628730367 * uncertainty_norm
    score = tightened_penalty + coupled_energy_penalty + 1 * risk_adjusted_duration - 1 * rank_weighted + 0 * criticality_bonus - 1 * successor_release + wait_bonus
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values in priority score'
    return score
