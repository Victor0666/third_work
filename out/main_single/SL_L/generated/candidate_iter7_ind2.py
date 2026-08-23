import numpy as np
RULE_METADATA = {'structure_hash': '61f43b1991e705fb8da169949de8dae3822d75741d98ced64370ced8b9bf3c9f', 'parameter_schema_hash': '216b9ff845392037ea6d3fdc7a5e45f5451847e203d11c7ce5aa8158dbad8132', 'best_parameter_hash': '9b702550a6b5e47ffd7bb5e1c13b875ee9c7dae363cd4a9352711dbb849a6647', 'best_parameters': {'epsilon': 0.00014081193971887454, 'slack_risk_penalty': 10.486971243835773, 'slack_urgency_scale': 3.4322837297331437, 'energy_efficiency_bias': 1.4306947773018368, 'critical_path_leverage': 0.6956196603233422, 'wait_fairness_gain': 0.3873190364098029, 'uncertainty_sensitivity': 0.4426678632917128, 'duration_balance': 0.3540744598110419, 'slack_pressure_tanh_scale': 0.867816388440575, 'work_density_weight': 0.28209245328718435, 'robustness_mad_factor': 1.5699974211900876, 'slack_decay_scale': 0.49861578714265403}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '0a51d16c6d7af3ab4495cbe8a03ffdad6184d9d0d025b3f516aaff1f97f599b1', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating counterfactual evidence:
    - Adds conditional DDL protection gate (slack >= 0 AND uncertainty <= median_uncertainty)
    - Replaces linear wait fairness with saturating exponential: 1 - exp(-gain * norm_wait)
    - Introduces successor-release interaction: upward_rank * remaining_work, gated by slack feasibility
    - Uses fairness headroom (wait_time / (|slack| + eps)) instead of raw slack to gate energy savings
    - Robustly normalizes all features using MAD scaled by robustness_mad_factor
    - All operations are epsilon-guarded and finite-value safe.
    """
    eps = 0.00014081193971887454
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
        scale = 1.5699974211900876 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = np.tanh(-slack * 0.867816388440575)
    slack_norm = 10.486971243835773 * slack_pressure + 3.4322837297331437 * (1.0 - np.tanh(slack * 0.49861578714265403))
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    fairness_headroom = np.divide(ready_wait_time, np.abs(slack) + eps)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_weight_adj = 1.4306947773018368 * np.clip(fairness_headroom, 0.0, 1.0)
    rank_work_interaction = upward_rank * remaining_work
    rank_work_norm = mad_normalize(rank_work_interaction)
    median_uncertainty = np.median(uncertainty)
    ddl_protection_gate = ((slack >= 0.0) & (uncertainty <= median_uncertainty + eps)).astype(float)
    critical_successor_boost = 0.6956196603233422 * rank_work_norm * ddl_protection_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 0.3873190364098029 * (1.0 - np.exp(-0.3873190364098029 * (wait_norm + eps)))
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 0.4426678632917128 * unc_norm * slack_pressure * ddl_protection_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_bonus = 0.28209245328718435 * work_density_norm * ddl_protection_gate
    score = slack_norm + 0.3540744598110419 * duration_norm + energy_weight_adj * energy_norm - critical_successor_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
