import numpy as np
RULE_METADATA = {'structure_hash': '1c19a52cc935b40ef0e7d9775d71bd695234357ad3cc9636a01b36363f9e31b1', 'parameter_schema_hash': 'f7dc1e28d95f3893981b171a34712e1c059809ae42beeb252f1209de2cef440c', 'best_parameter_hash': 'dee666245f501945f8fabafbc5ff416cb0ea4dce53b803f92b7b4678f8ae2aa6', 'best_parameters': {'epsilon': 0.02203513389661572, 'slack_penalty_exponent': 3.542066224730915, 'criticality_scale': 1.4791525863444572, 'energy_sensitivity': 1.2857116817848475, 'duration_robustness': 1.6718174666753391, 'wait_decay': 0.0019468879381769374, 'uncertainty_gate_threshold': 0.6578727524421824, 'slack_sensitivity': 0.5817121528476737, 'energy_uncertainty_interaction': 0.8241653657281743, 'work_leverage_factor': 1.834519649268639, 'congestion_penalty_weight': 0.4048782557352542}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '5aadc2436ef39f4388f50637eb8fc42186135a0832078a4651645d6f1a41feb3', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: adds DDL-protection gate (criticality boost only when rank > median AND slack < 0);
       introduces congestion_penalty using ready_wait_time as load proxy under uncertainty;
       replaces linear slack_reward with piecewise-linear ramp: zero for slack < 0, linear from 0 to max over [0, median_slack], then saturates;
       removes unused slack_gate_width; all literals are -2,-1,0,1,2; epsilon via PARAMS; no side effects."""
    eps = 0.02203513389661572
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
        center = np.median(x)
        scale = np.median(np.abs(x - center)) + eps
        return (x - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    median_slack = np.median(slack)
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 3.542066224730915)
    median_norm_rank = np.median(norm_rank)
    ddl_protection_gate = ((norm_slack < 0.0) & (norm_rank > median_norm_rank)).astype(float)
    boosted_rank = norm_rank * (1.0 + 1.4791525863444572 * ddl_protection_gate)
    uncert_gate = ((norm_uncert > 0.6578727524421824) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * uncert_gate * 1.6718174666753391
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * 0.8241653657281743
    wait_benefit = np.tanh(0.0019468879381769374 * norm_wait)
    slack_reward = np.zeros_like(norm_slack)
    positive_slack_mask = slack >= 0.0
    if N > 0 and np.any(positive_slack_mask):
        ramp_region = (slack > 0.0) & (slack <= median_slack)
        saturation_region = slack > median_slack
        slack_reward[ramp_region] = 0.5817121528476737 * (slack[ramp_region] / (median_slack + eps))
        slack_reward[saturation_region] = 0.5817121528476737
    work_leverage_gate = (slack >= 0.0).astype(float)
    work_leverage = norm_work * work_leverage_gate * 1.834519649268639
    congestion_penalty = norm_wait * uncert_gate * 0.4048782557352542
    score = +slack_pressure + 1.2857116817848475 * norm_energy - boosted_rank + duration_risk_interaction - wait_benefit - slack_reward + energy_uncert_penalty - work_leverage + congestion_penalty
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
