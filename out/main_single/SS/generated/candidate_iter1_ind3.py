import numpy as np
RULE_METADATA = {'structure_hash': '3d6195bc382643bbc582440b41e79a7f5044533c16caf225cca9b5c68470550c', 'parameter_schema_hash': '1de198d881063a22a4798a1b21f71d807c5d29f99154c3dd8fa5046e2e6f2438', 'best_parameter_hash': '064916ef87b6c9503938d226225258bddeaa3285e962db35ccfadb01009e650d', 'best_parameters': {'epsilon': 0.0005062789643831252, 'slack_penalty_exponent': 2.693654773335341, 'criticality_scale': 0.9683133556194332, 'energy_sensitivity': 0.8741054510570501, 'duration_robustness': 1.9850898614382204, 'wait_decay': 0.1558346664256122, 'uncertainty_gate_threshold': 0.6704008904316704, 'rank_slack_coupling': 0.8006526347018035, 'slack_gate_width': 1.1031222900559807}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'f562369f8a22cebd187d74ed3b41dd837d09c74ed6b74adf6465ad1a824a9aea', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all numeric thresholds moved to PARAMETER_SCHEMA; only -2,-1,0,1,2 used as literals."""
    eps = 0.0005062789643831252
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
        x = np.abs(x)
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
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 2.693654773335341)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack, 0.0, 1.1031222900559807), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 0.9683133556194332 * rank_gate)
    uncert_gate = ((norm_uncert > 0.6704008904316704) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * uncert_gate * 1.9850898614382204
    wait_benefit = np.tanh(0.1558346664256122 * norm_wait)
    coupled_rank = norm_rank * (1.0 + 0.8006526347018035 * slack_pressure)
    score = +slack_pressure + 0.8741054510570501 * norm_energy - coupled_rank - wait_benefit + duration_risk_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(N)
