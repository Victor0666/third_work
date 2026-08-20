import numpy as np
RULE_METADATA = {'structure_hash': '326483a7ab2477e6432f27646981194afa206ad22b0171006abf986e31672483', 'parameter_schema_hash': '3f7ebc1ad1ee6b08fe0dd7adaec19767f11fd1b203b017271dca8636141236c9', 'best_parameter_hash': 'a286df764d5cd03752734964d2841b2e49bcecef07243389666a901522f5ae40', 'best_parameters': {'epsilon': 0.010174273423718978, 'slack_penalty_exponent': 1.0287065656896248, 'criticality_scale': 1.9532026884692162, 'energy_sensitivity': 1.096423454214752, 'duration_robustness': 0.6251116134173355, 'wait_decay': 0.8567257748393174, 'uncertainty_gate_threshold': 0.40102988052344163, 'slack_pressure_gate_steepness': 3.928298103023734, 'remaining_work_weight': 0.6276904920199133, 'wait_saturation_offset': 0.00016927832301937668, 'energy_uncertainty_interaction': 0.305079147608293, 'ddl_protection_strength': 0.47191076216011785}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '36d728e5a100c45e2be1aab67bbda6f7fd31a4f35828b7aeb9bd2b8d59e5733c', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces fragile blended slack with conditional DDL-protection gate.
       When slack < 0, raw-slack penalty dominates via convex weighting — preserving hard constraint fidelity.
       When slack >= 0, robust normalization governs ranking stability. Eliminates dilution of urgency signals
       while retaining outlier resilience. All parameters used; no literals except -2,-1,0,1,2."""
    eps = 0.010174273423718978
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
        x_abs = np.abs(x)
        center = np.median(x_abs) if N > 1 else x_abs[0]
        spread = np.median(np.abs(x_abs - center)) if N > 1 else np.abs(x_abs[0] - center) + eps
        return (x_abs - center) / (spread + eps)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.0287065656896248
    norm_slack = robust_normalize(slack)
    ddl_violated = (slack < 0.0).astype(float)
    primary_slack_score = 0.47191076216011785 * raw_slack_penalty * ddl_violated + (1.0 - 0.47191076216011785 * ddl_violated) * norm_slack
    slack_pressure = np.clip(-primary_slack_score, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.928298103023734 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.9532026884692162 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.40102988052344163, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * raw_slack_penalty
    wait_benefit = 1.0 - np.exp(-0.8567257748393174 * (norm_wait + 0.00016927832301937668))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +raw_slack_penalty - boosted_rank - 1.096423454214752 * norm_energy - norm_duration - wait_benefit + 0.6251116134173355 * duration_risk_score + 0.305079147608293 * energy_uncert_penalty + 0.6276904920199133 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
