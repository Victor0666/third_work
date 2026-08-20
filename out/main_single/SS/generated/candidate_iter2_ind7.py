import numpy as np
RULE_METADATA = {'structure_hash': 'a018fe9238dac226de10251c7e9c8a0dfa2037ee4bf2dc9f2790ab6801612d6a', 'parameter_schema_hash': '6f246c63fdc4cb1b75208fe280ee9163dc7cd3f18a589c5ed93c3a920234bb1f', 'best_parameter_hash': 'c63296d8ca6efb0d24f77dc676120c26cb5f1b0320ceff7ed50894e82c481a5c', 'best_parameters': {'epsilon': 0.00023882318054377314, 'slack_penalty_exponent': 2.1113818147838805, 'criticality_scale': 0.5462720192402539, 'energy_sensitivity': 0.5907345121324301, 'duration_robustness': 0.001637673826814248, 'wait_decay': 0.053083907154760246, 'uncertainty_gate_threshold': 0.665579170109786, 'slack_gate_width': 1.0399606209992085, 'energy_slack_interaction': 0.9706981799644561}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '355c07ee054f51cb4c0bc14337f20f96a5aa96456a6c99d78a4b5251998c317d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: removes unused 'rank_slack_coupling'; retains robust starvation relief, adaptive gating, and novel energy-slack interaction."""
    eps = 0.00023882318054377314
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
        center = np.median(x_abs)
        scale = np.median(np.abs(x_abs - center)) + eps
        return (x_abs - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 2.1113818147838805)
    rank_gate = np.clip(1.0 - norm_slack / (1.0399606209992085 + eps), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 0.5462720192402539 * rank_gate)
    uncert_gate = ((norm_uncert > 0.665579170109786) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * uncert_gate * 0.001637673826814248
    wait_benefit = 1.0 - np.exp(-0.053083907154760246 * (norm_wait + eps))
    energy_slack_penalty = norm_energy * (1.0 + 0.9706981799644561 * slack_pressure)
    score = +slack_pressure + energy_slack_penalty + 0.5907345121324301 * norm_energy - boosted_rank - wait_benefit + duration_risk_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
