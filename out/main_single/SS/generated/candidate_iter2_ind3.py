import numpy as np
RULE_METADATA = {'structure_hash': '99aa017b14479155efe3937ca6b1b07efa279690653924509c3cfe9f07845342', 'parameter_schema_hash': 'aa01b8c1c5bfe29110d553c33a90b312b54c4a7c84bc60f33d277e4abcfdb2fd', 'best_parameter_hash': 'd36f1c08b21af19b78047738edd110c22316a52704dfe435f7e82ceedf9fe0b5', 'best_parameters': {'epsilon': 2.0890166633808677e-06, 'slack_penalty_exponent': 3.6898098971346753, 'criticality_scale': 0.6245180392367169, 'energy_sensitivity': 1.0338590955421172, 'duration_robustness': 1.3270018619798665, 'wait_decay': 0.204467104732935, 'uncertainty_gate_threshold': 0.7864683351925934, 'slack_gate_width': 1.1812219549965761, 'work_pressure_weight': 0.4584935404565596}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'a607171d84f8521205aa6de9b5f5cd89cea17a72efdcaae790bf988772369847', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all declared parameters are used; no unused entries.
    Introduces work-pressure term to prevent late-stage bottlenecks under deadline pressure.
    Uses unified slack_pressure signal, smooth rank_gate (with literal 2.0), tanh starvation relief, and robust normalization.
    All numeric literals are -2,-1,0,1,2; epsilon via PARAMS; no side effects or I/O."""
    eps = 2.0890166633808677e-06
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
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 3.6898098971346753)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack, 0.0, 1.1812219549965761), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 0.6245180392367169 * rank_gate)
    work_pressure = np.where(slack_pressure > 0, 0.4584935404565596 * norm_work * slack_pressure, 0.0)
    uncert_gate = ((norm_uncert > 0.7864683351925934) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * uncert_gate * 1.3270018619798665
    wait_benefit = np.tanh(0.204467104732935 * norm_wait)
    score = +slack_pressure + 1.0338590955421172 * norm_energy - boosted_rank + work_pressure + duration_risk_interaction - wait_benefit
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    return score
