import numpy as np
RULE_METADATA = {'structure_hash': 'e99ca13608d162da8c477cd7939352f7d1b8c3260862f50bade9af0d7d9791df', 'parameter_schema_hash': '0416426438243095881e11b91eab3122a0e35c67b5cc0b748c803147cd8b3d8c', 'best_parameter_hash': '1eb39b6a24828e668d3b4e8ee60a4b2479b946b185389f4602242aa1898fa666', 'best_parameters': {'epsilon': 0.00916457452264223, 'criticality_scale': 0.7507805413284196, 'energy_sensitivity': 0.6328746763535441, 'wait_decay': 0.02387361707112217, 'uncertainty_gate_threshold': 0.9816334519320715, 'remaining_work_weight': 0.5507617964947042, 'wait_saturation_offset': 1.001591869204927e-09, 'energy_uncertainty_interaction': 0.5246159277733023, 'successor_release_coupling_strength': 1.0377379752132905, 'urgency_smoothing_factor': 1.1028792434998063, 'quantile_alpha': 0.4223624212779489, 'load_aware_gate_bias': 0.015720576598595045}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '29a3a5a8dc2b199f9b7765637b3e18412506e1524d8776b1944c17f318ec9ebd', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
       - Quantile-based robust scaling (Q3-Q1 IQR + epsilon) → handles skew/outliers better.
       - Load-aware conditional gate decoupling DDL urgency from energy penalties.
       - Piecewise linear urgency with breakpoint at slack=0 → sharp deadline compliance.
       - All numeric literals are {-2,-1,0,1,2}; exactly 12 parameters; all used; no hidden constants.
    """
    eps = 0.00916457452264223
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def scale_feature(x):
        x_abs = np.abs(x)
        q1 = np.quantile(x_abs, 0.4223624212779489, method='linear')
        q3 = np.quantile(x_abs, 1.0 - 0.4223624212779489, method='linear')
        spread = q3 - q1 + eps
        return x / (spread + eps)
    norm_slack = scale_feature(slack)
    norm_energy = scale_feature(min_incremental_energy)
    norm_duration = scale_feature(min_exec_time + min_comm_time)
    norm_rank = scale_feature(upward_rank)
    norm_work = scale_feature(remaining_work)
    norm_wait = scale_feature(ready_wait_time)
    norm_uncert = scale_feature(uncertainty)
    urgency_signal = np.where(slack < 0.0, -slack * 1.1028792434998063, 0.0)
    load_proxy = (norm_duration + norm_energy + norm_uncert) / 2.0
    load_gate = np.where(load_proxy > 0.015720576598595045, 1.0, 0.0)
    uncert_gate = np.tanh((0.9816334519320715 - norm_uncert) * 2.0)
    ddl_protection_gate = np.clip(urgency_signal * uncert_gate, 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 0.7507805413284196 * ddl_protection_gate)
    successor_release_score = norm_work * norm_rank * 1.0377379752132905 * ddl_protection_gate
    wait_benefit = np.tanh(0.02387361707112217 * (norm_wait + 1.001591869204927e-09))
    risk_energy_penalty = norm_energy * norm_uncert * ddl_protection_gate * (1.0 - uncert_gate) * load_gate
    score = +urgency_signal - boosted_rank - successor_release_score - wait_benefit + 0.5246159277733023 * risk_energy_penalty - 0.6328746763535441 * norm_energy + 0.5507617964947042 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
