import numpy as np
RULE_METADATA = {'structure_hash': '747838d5c9ed269e9231fa7c6aa4c0054d98808baab214cd749910613e8bc8bb', 'parameter_schema_hash': 'd0360d8b3532d0ece3ad1b2aafcc4d6ca4e8ef522c7baf3836562c59a351311e', 'best_parameter_hash': '1512b4f13fe851d300c3f8a262bc591af1199bb250a4797f2b5c6d6878945ff1', 'best_parameters': {'epsilon': 0.0038065167293076796, 'slack_penalty_exponent': 1.8111413195945765, 'criticality_scale': 1.8173918575446293, 'energy_sensitivity': 0.3685814630303057, 'duration_robustness': 0.5574835348565593, 'wait_decay': 0.13167935804650743, 'uncertainty_gate_threshold': 0.736136877139391, 'slack_pressure_gate_steepness': 1.377947078358818, 'remaining_work_weight': 0.8285342308023778, 'wait_saturation_offset': 2.164901537848564e-05, 'energy_uncertainty_interaction': 0.4875857421881178, 'starvation_relief_cap': 0.9996499887791982}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '70dd12c008a2e8f3bb38ad0666c6c6acea78120605fbae275847c4101e5620ce', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: integrates binary DDL breach response with calibrated urgency boost;
       replaces unbounded waiting bonuses with capped logistic relief;
       refines successor-release coupling using slack-gated residual work impact;
       uses N-aware median-MAD normalization with explicit scalar fallback for N=1;
       adds explicit starvation cap to preserve DDL feasibility hierarchy."""
    eps = 0.0038065167293076796
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        if N == 1:
            center = x[0]
            spread = np.abs(x[0]) + eps
        else:
            center = np.median(x)
            spread = np.median(np.abs(x - center))
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    ddl_breach = (slack <= 0.0).astype(float)
    slack_gap = np.maximum(-slack, 0.0)
    slack_pressure_gate = 1.0 / (1.0 + np.exp(-1.377947078358818 * (slack_gap - eps)))
    residual_work_impact = norm_rank * norm_work * (ddl_breach + (1.0 - ddl_breach) * slack_pressure_gate)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.8111413195945765
    breach_amplified_penalty = raw_slack_penalty * (1.0 + 0.9996499887791982 * ddl_breach)
    norm_slack_penalty = robust_normalize(breach_amplified_penalty)
    boosted_rank = norm_rank * (1.0 + 1.8173918575446293 * slack_pressure_gate)
    uncert_gate = (norm_uncert > 0.736136877139391).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    duration_risk_score = norm_duration * norm_uncert * slack_pressure_gate
    wait_benefit = np.clip(1.0 - np.exp(-0.13167935804650743 * (norm_wait + 2.164901537848564e-05)), 0.0, 0.9996499887791982)
    score = +norm_slack_penalty - residual_work_impact - boosted_rank - 0.3685814630303057 * norm_energy - wait_benefit + 0.5574835348565593 * duration_risk_score + 0.4875857421881178 * energy_uncert_penalty + 0.8285342308023778 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
