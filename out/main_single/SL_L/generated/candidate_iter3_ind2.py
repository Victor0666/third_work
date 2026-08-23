import numpy as np
RULE_METADATA = {'structure_hash': 'eb794e74b19f2a896a39c5cbb61f6703822a8d0974d90122258e912b557f1429', 'parameter_schema_hash': '1b5cdd4cd0369ce3ecc3fe3844a227524cd3164fede5047ddb7c0893b59b0a9d', 'best_parameter_hash': '1b4f3cdabaf627fb7ef5008f742faec9f165889a8c20c64ad758e617f2f718c3', 'best_parameters': {'epsilon': 2.9672283905202078e-08, 'slack_sigmoid_steepness': 6.089881732590966, 'slack_sigmoid_offset': -2.8757089140854517, 'criticality_weight': 1.345705031711511, 'energy_sensitivity': 0.12075415914369045, 'duration_balance': 0.5844969143454872, 'wait_decay_rate': 0.03659685855570918, 'uncertainty_slack_coupling': 0.680207128633365, 'rank_energy_interaction': 1.1476394659571711, 'urgency_clip_max': 3.0847853836233297}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '7ef8c2255dbd0b2dea69bfcd49f7c78dd9aa392b798ed249afa33512cd62cb19', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Self-evolved priority rule: replaces brittle threshold gates with smooth, bounded sigmoid slack gating;
    eliminates piecewise discontinuity and unstable early-reward terms; unifies urgency/criticality via monotonic risk scaling;
    retains robust normalization and finite safeguards.
    """
    eps = 2.9672283905202078e-08

    def robust_norm(x):
        x = np.asarray(x, dtype=np.float64)
        denom = np.mean(np.abs(x)) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    norm_duration = robust_norm(min_exec_time + min_comm_time)
    norm_rank = robust_norm(upward_rank)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    sigmoid_input = 6.089881732590966 * (norm_slack - -2.8757089140854517)
    sigmoid_input_clipped = np.clip(sigmoid_input, -3.0847853836233297, 3.0847853836233297)
    slack_gate = 1.0 / (1.0 + np.exp(-sigmoid_input_clipped))
    urgency_base = np.maximum(0.0, -norm_slack)
    urgency_clipped = np.clip(urgency_base, 0.0, 3.0847853836233297)
    urgency_penalty = np.exp(6.089881732590966 * urgency_clipped) * slack_gate
    critical_boost = 1.345705031711511 * norm_rank * (1.0 - slack_gate)
    deadline_pressure = np.maximum(0.0, -norm_slack)
    uncertainty_coupled_pressure = 0.680207128633365 * deadline_pressure * norm_uncert * slack_gate
    wait_boost = 1.0 - np.exp(-0.03659685855570918 * norm_wait)
    rank_energy_penalty = 1.1476394659571711 * norm_energy * norm_rank
    score = urgency_penalty + critical_boost + uncertainty_coupled_pressure + rank_energy_penalty + 0.12075415914369045 * norm_energy + 0.5844969143454872 * norm_duration - wait_boost
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=-finfo.max)
    return score
