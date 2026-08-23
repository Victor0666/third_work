import numpy as np
RULE_METADATA = {'structure_hash': 'f29bb2775c0ea16b523dcebb940efe4380c4c59c235db823dbe7329b4de6f4de', 'parameter_schema_hash': 'c4d0fedad8de1b630df0a21e84f8d3ba4b3efc6f947792941ad3083379a2af5e', 'best_parameter_hash': '14e2dabaf5e35bf16c296265c8dc0a65778fa214a2bb06377ec00cee6131556c', 'best_parameters': {'epsilon': 1.7311206754225584e-08, 'slack_pressure_tanh_scale': 0.8251162700349802, 'energy_sensitivity': 0.8783243404545725, 'critical_path_leverage': 0.7553259113146857, 'wait_fairness_gain': 0.8394199222114116, 'uncertainty_sensitivity': 0.03727834645831601, 'duration_balance': 0.5748416075532924, 'work_density_weight': 0.5347740604320409, 'robustness_mad_factor': 1.2826111663853537}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'c0fcf606ea9980f52cd020f62b1e3fb46b8f82c9c6309c81d1595281ed3e30fe', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Hybrid priority rule with unified tanh-based slack pressure, joint slack+uncertainty gating
    for critical path leverage, saturating wait fairness, and robust MAD normalization.
    All parameters declared in PARAMETER_SCHEMA are used; no unused or missing references.
    Numeric literals restricted to {-2,-1,0,1,2}; epsilon guarded via PARAMS["epsilon"].
    """
    eps = 1.7311206754225584e-08
    finfo = np.finfo(np.float64)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.2826111663853537 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = np.tanh((1 + 0.8251162700349802) * -slack)
    slack_norm = np.clip(1 + slack_pressure, 0.0, 2.0)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_score = 0.8783243404545725 * energy_norm
    rank_norm = mad_normalize(upward_rank)
    unc_med = np.median(uncertainty)
    unc_gate = np.clip(1.0 - (uncertainty - unc_med) / np.maximum(unc_med, eps), 0.0, 1.0)
    critical_gate = ((slack >= 0.0) * unc_gate).astype(float)
    critical_boost = 0.7553259113146857 * rank_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_gate = (slack >= 0.0).astype(float)
    work_density_bonus = 0.5347740604320409 * work_density_norm * work_density_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-0.8394199222114116 * np.maximum(wait_norm, 0.0))
    wait_score = wait_boost * (slack >= 0.0).astype(float)
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 0.03727834645831601 * unc_norm * np.abs(slack_pressure)
    score = slack_norm + 0.5748416075532924 * duration_norm + energy_score - critical_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=-finfo.max)
    return score
