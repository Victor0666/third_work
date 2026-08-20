import numpy as np
RULE_METADATA = {'structure_hash': '8d9ac0132d32ac0e4b9d3b41a60a8e6a8608a4e1b0154d11c7dda99e53999ab8', 'parameter_schema_hash': '919408ba8accc3db3a8b39d8331c17a8f0c82a8a47bcd138b5433dc4415ddd05', 'best_parameter_hash': 'dd6585e3f27f3f2130539bc6110f3c2e224c4571693ce7921cdcc24ad67e6a7b', 'best_parameters': {'epsilon': 0.00493738049634712, 'slack_penalty_exponent': 1.0549395086746804, 'criticality_scale': 1.5108844795869822, 'energy_sensitivity': 0.9849084062508724, 'remaining_work_weight': 0.7499635025174791, 'uncertainty_gate_threshold': 0.03343340371336249, 'energy_uncertainty_interaction': 0.9733159745120652, 'tanh_half_scale': 0.10685905817452228, 'slack_linear_threshold': 0.1509511068007311, 'bottleneck_activation_gate': 0.06204090217362541, 'duration_robustness_factor': 1.33262845021676}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '47237811cbc17b60fc91d22626ba756353a27a0f47c7fd454e48f8bd446adcef', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's structural advances with Parent 1's robust duration handling:
       - Keeps piecewise slack penalty (linear+quadratic) and dual-gated bottleneck release.
       - Integrates 'duration_robustness_factor' to dampen norm_duration and exec_penalty — empirically reduces sensitivity to estimation outliers.
       - Uses composite DDL gate: tanh-based feasibility signal AND hard zero for negative slack → ensures strict hard-deadline enforcement.
       - Adds robustified duration penalty: damped by duration_robustness_factor, fully gated by composite DDL condition, and scaled by slack_pressure.
       - Removes all starvation relief terms (confirmed destabilizing in Parent 2 diagnostics).
       - All normalizations use median-MAD; all outputs clipped to [-2,2] for bounded AST depth and stability."""
    eps = 0.00493738049634712
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_uncert = median_mad_normalize(uncertainty)
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    tanh_feasible = np.clip(0.10685905817452228 * (1.0 - np.tanh(slack / (eps + np.finfo(float).tiny))), 0.0, 1.0)
    hard_deadline_violated = (slack < 0.0).astype(float)
    ddl_gate = (1.0 - hard_deadline_violated) * tanh_feasible
    slack_pressure = np.clip(norm_slack, -1.0, 1.0)
    ddl_urgent = (slack <= eps).astype(float)
    low_uncert = (norm_uncert <= 0.06204090217362541).astype(float)
    successor_release = norm_work * norm_rank * ddl_urgent * low_uncert
    abs_norm_slack = np.abs(norm_slack)
    linear_penalty = abs_norm_slack
    quadratic_penalty = abs_norm_slack ** 2
    piecewise_slack_penalty = np.where(abs_norm_slack <= 0.1509511068007311, linear_penalty, quadratic_penalty)
    energy_penalty = norm_energy * (1.0 + 1.0549395086746804 * slack_pressure) * (1.0 - ddl_gate)
    uncert_gate = (norm_uncert > 0.03343340371336249).astype(float)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * (1.0 - ddl_gate)
    duration_penalty = 1.33262845021676 * norm_duration * slack_pressure * (1.0 - ddl_gate)
    score = +np.clip(piecewise_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(norm_rank * (1.0 + 1.5108844795869822 * slack_pressure), -2.0, 2.0) - 0.9849084062508724 * np.clip(energy_penalty, -2.0, 2.0) + 0.9733159745120652 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.7499635025174791 * np.clip(norm_work, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
