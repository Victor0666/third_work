import numpy as np
RULE_METADATA = {'structure_hash': 'b46930fba0ba1c39edcb2e21fef48e8d73aebf83cb2d308aa4fd51116348dfd8', 'parameter_schema_hash': 'f03a311ce705b0595d3b130d1e6b15a21fb373c978f719e2f5bf7f09258a824a', 'best_parameter_hash': '129c3c8633a7f2c97c408dd599f65a18cfb7651c26fa511748d03fa14a0f96f3', 'best_parameters': {'epsilon': 1.1852049394051301e-05, 'slack_urgency_scale': 4.062739147784761, 'energy_efficiency_bias': 0.8927859776693965, 'critical_path_leverage': 0.3811406973435453, 'wait_fairness_gain': 0.26759482504577825, 'duration_balance': 0.3907055550712599, 'work_density_weight': 0.0360720215985227, 'robustness_mad_factor': 1.433147043591394, 'slack_pressure_tanh_scale': 0.22764270190698413, 'rank_uncertainty_coupling': 0.7910841871889606, 'energy_slack_coupling': 0.2290799789139804}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'fb569c6442506814656420ea07e6430d5b7569d47587da852971fd8b6bda118e', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with bounded tanh slack urgency (replacing piecewise),
       pre-gated rank normalization for stability, and decoupled energy-slack coupling.
       Key improvements:
       - Replaces fragile piecewise slack_norm with single smooth tanh urgency curve: 
         slack_urgency_scale * tanh(-slack * slack_pressure_tanh_scale) → negative slack yields strong penalty,
         near-zero slack yields sharp urgency, large positive slack saturates near zero.
       - Normalizes upward_rank *before* applying dual gating (slack >= 0 AND uncertainty), eliminating outlier-driven instability.
       - Introduces energy_slack_coupling to selectively amplify energy penalty only under slack pressure,
         avoiding over-penalization in slack-rich regimes.
       - Removes redundant uncertainty_amplifier + resilience_boost; replaces with unified rank_uncertainty_coupling
         applied *only* under feasibility (slack >= 0) for robust critical-path resilience.
    """
    eps = 1.1852049394051301e-05
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.433147043591394 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = -slack
    slack_urgency = 4.062739147784761 * np.tanh(slack_pressure * 0.22764270190698413)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_base = 0.8927859776693965 * energy_norm
    energy_slack_gate = np.clip(slack_pressure, 0.0, np.inf)
    energy_score = energy_base * (1.0 + 0.2290799789139804 * np.tanh(energy_slack_gate * 0.22764270190698413))
    rank_norm = mad_normalize(upward_rank)
    unc_med = np.median(uncertainty)
    unc_gate = np.clip(1.0 - (uncertainty - unc_med) / np.maximum(unc_med, eps), 0.0, 1.0)
    feasibility_gate = (slack >= 0.0).astype(float)
    critical_gate = feasibility_gate * unc_gate
    critical_boost = 0.3811406973435453 * rank_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_bonus = 0.0360720215985227 * work_density_norm * feasibility_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-0.26759482504577825 * np.maximum(wait_norm, 0.0))
    wait_score = wait_boost * feasibility_gate
    unc_norm = mad_normalize(uncertainty)
    resilience_boost = 0.7910841871889606 * rank_norm * unc_norm * feasibility_gate
    score = slack_urgency + 0.3907055550712599 * duration_norm + energy_score - critical_boost - work_density_bonus + wait_score - resilience_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
