import numpy as np
RULE_METADATA = {'structure_hash': 'b5b86706f951655bdb5f631259a456fc84a56728b4d6e011c9badd7e09910589', 'parameter_schema_hash': '5cb54649cfddd9fbaab00493028003906df3305e898c3bbc5be53bedb05e1168', 'best_parameter_hash': 'b675df9bc80ec6aa749b83835941d30dc3bc7dd133d3b000d015d5ef05e70919', 'best_parameters': {'epsilon': 0.0011420842303027728, 'slack_penalty_exponent': 1.7037028330020567, 'criticality_scale': 0.8297927935298611, 'energy_sensitivity': 0.7815710762883116, 'duration_robustness': 0.22679516112736903, 'wait_decay': 0.49723236240378843, 'uncertainty_gate_threshold': 0.6098296492965503, 'slack_gate_width': 0.31597243152981236, 'rank_slack_coupling': 0.7284431573545225, 'energy_uncertainty_interaction': 0.6354745086139708, 'ddl_stress_energy_relaxation': 0.33744685105737204}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'b074e0ba77221e8f2cc9ed396518bab87cd4493173c66aac891502fcbe832056', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's robust min-max normalization and raw slack fidelity
       with Parent 1's conditional energy-relaxation under DDL stress and explicit energy-uncertainty penalty.
       Introduces novel dual-gate structure: (1) slack-pressure-triggered energy relaxation,
       and (2) joint energy-uncertainty penalty only active under high risk AND negative slack.
       Uses bounded min-max scaling for all features except slack (kept raw for deadline signal integrity).
       All operations protected against NaN/inf; deterministic and shape-correct."""
    eps = 0.0011420842303027728
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def bounded_minmax(x):
        x_min = np.min(x)
        x_max = np.max(x)
        range_val = x_max - x_min + eps
        return (x - x_min) / range_val
    norm_energy = bounded_minmax(min_incremental_energy)
    norm_duration = bounded_minmax(min_exec_time + min_comm_time)
    norm_rank = bounded_minmax(upward_rank)
    norm_wait = bounded_minmax(ready_wait_time)
    norm_uncert = bounded_minmax(uncertainty)
    slack_pressure_raw = np.clip(-slack, 0.0, None)
    slack_pressure = np.power(slack_pressure_raw + eps, 1.7037028330020567)
    norm_slack_for_gate = bounded_minmax(slack)
    rank_gate = np.clip(1.0 - 2.0 * np.clip(norm_slack_for_gate, 0.0, 0.31597243152981236), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 0.8297927935298611 * rank_gate)
    joint_risk_gate = ((norm_uncert > 0.6098296492965503) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * joint_risk_gate * 0.22679516112736903
    wait_benefit = np.tanh(0.49723236240378843 * norm_wait)
    coupled_rank = np.clip(norm_rank * (1.0 + 0.7284431573545225 * slack_pressure), 0.0, 2.0)
    energy_weight = np.where(slack <= 0.0, 0.7815710762883116 * 0.33744685105737204, 0.7815710762883116)
    energy_uncert_active = (slack < 0.0) & (norm_uncert > 0.6098296492965503)
    energy_uncert_penalty = np.where(energy_uncert_active, norm_energy * norm_uncert, np.zeros_like(norm_energy))
    score = +slack_pressure + energy_weight * norm_energy - coupled_rank - wait_benefit + duration_risk_interaction + 0.6354745086139708 * energy_uncert_penalty
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
