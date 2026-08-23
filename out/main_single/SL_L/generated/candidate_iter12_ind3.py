import numpy as np
RULE_METADATA = {'structure_hash': 'bb5fe32733be11cf7da7ff5efc0e1d063a95c2efe5e3ce0390ff667947c70d36', 'parameter_schema_hash': 'e53087e98bc5f43fc7309379dc263c5f2649414aa9f75aaa0e4e25d305fc7879', 'best_parameter_hash': 'c8e2158b28e2c945b306805a780f73051d6e073bc30eaad60ef8b322fa73d35d', 'best_parameters': {'epsilon': 4.136934127926047e-08, 'slack_penalty_exponent': 2.3618718055420924, 'slack_aware_criticality_scale': 0.9410954856916203, 'energy_efficiency_ratio_weight': 0.1388587070729272, 'uncertainty_slack_coupling': 0.24164658817237578, 'duration_risk_penalty': 1.2819891091137614, 'energy_uncertainty_interaction': 0.10207963100012407, 'uncertainty_tanh_steepness': 5.452991997815694, 'wait_fairness_gain': 0.7422492924777887, 'work_density_weight': 0.9750817807084352, 'tanh_bias': 0.5869426707207401, 'robust_quantile': 0.933181219696939}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '7a47bd847a33c827f235e30909fd1bf4d3f26888bb84ffc70447be78ab7150bc', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with smooth slack penalty, tanh-based uncertainty gating, and MAD-normalization.
    
    Key structural improvements:
    - Replaces piecewise slack scoring with continuous `max(0,-slack)**p` — eliminates discontinuities and instability.
    - Replaces IQR normalization with robust MAD (mean absolute deviation) to avoid outlier sensitivity of median/IQR.
    - Replaces sigmoid uncertainty gates with bounded `tanh(steepness * uncertainty)` for smoother, monotonic modulation.
    - Applies uncertainty gating *only* to energy/duration terms (not slack_score), decoupling deadline urgency from risk scaling.
    - All inputs sanitized with `np.nan_to_num`; no in-place mutation; strict shape-(N,) output.
    """
    eps = 4.136934127926047e-08
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        center = np.mean(x)
        abs_dev = np.abs(x - center)
        scale = np.mean(abs_dev) + eps
        return (x - center) / (scale + eps)
    slack_penalty = np.maximum(0.0, -slack) ** 2.3618718055420924
    duration_total = min_exec_time + min_comm_time + eps
    duration_norm = mad_normalize(duration_total)
    unc_gate = np.tanh(5.452991997815694 * uncertainty)
    duration_risk = duration_norm * (1.0 + unc_gate * 1.2819891091137614)
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_norm = mad_normalize(energy_per_sec)
    energy_eff_score = energy_eff_norm * 0.1388587070729272 * (1.0 + unc_gate)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = mad_normalize(uncertainty * deadline_pressure) * 0.24164658817237578
    rank_norm = mad_normalize(upward_rank)
    clipped_slack = np.maximum(0.0, -slack)
    critical_bonus = rank_norm * (1.0 + 0.9410954856916203 * clipped_slack)
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.10207963100012407 * energy_norm * unc_norm * unc_gate
    energy_base_score = mad_normalize(min_incremental_energy) * (1.0 + unc_gate)
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 0.5869426707207401 * (1.0 + np.tanh(0.7422492924777887 * wait_norm))
    duration_q = np.quantile(duration_total, 0.933181219696939) + eps
    work_density = remaining_work / duration_q
    work_density_norm = mad_normalize(work_density)
    work_density_bonus = 0.9750817807084352 * work_density_norm
    score = mad_normalize(slack_penalty) + duration_risk + energy_eff_score + unc_slack_coupling + energy_uncertainty_score + energy_base_score + wait_boost - critical_bonus - work_density_bonus
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
