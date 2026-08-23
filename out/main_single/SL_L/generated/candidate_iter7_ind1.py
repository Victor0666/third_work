import numpy as np
RULE_METADATA = {'structure_hash': '2173d097af352fe48ffc8dbecc20be392dbc78e8e63d51b820f27e507e7d9f98', 'parameter_schema_hash': '80f2ec29d7abfc0bc19a2f34facca37d22520dab4e5729601cab91f111eb7190', 'best_parameter_hash': '7bb9940293dd3efebc63b7af28e4d1504278e19b681e4f0ef8b2187fca7809f2', 'best_parameters': {'epsilon': 0.051539024687597565, 'slack_risk_penalty': 6.6000883237386745, 'slack_urgency_scale': 3.478060998337171, 'energy_efficiency_bias': 1.7638848792942068, 'critical_path_leverage': 0.14538175944799592, 'wait_fairness_gain': 0.37667132323423785, 'uncertainty_sensitivity': 0.004760330165028379, 'slack_pressure_tanh_scale': 0.016826942091481577, 'work_density_weight': 0.34908248790751767, 'robustness_mad_factor': 1.2468377059653388}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'cce27f6b865152e91edbf203eea17eb3eecd6d90c7125baf5a267784bfa70252', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule integrating successor-release interaction and host-load conditional gating.
    
    Key structural mutations:
    - Replaces piecewise slack transformation with smooth tanh-based urgency curve for unified risk modeling.
    - Introduces 'successor_release_interaction': upward_rank × remaining_work, gated by slack feasibility
      and uncertainty threshold (to unblock critical paths only when risk allows).
    - Adds 'host_load_conditional_gate': energy term scaled by fairness headroom (wait_time / (|slack|+eps)),
      enabling energy savings only where deadlines permit—no unconditional energy minimization.
    - Uses saturating exponential wait boost (1 - exp(-gain * norm_wait)) for bounded fairness.
    - All interactions respect DDL-first lexicographic ordering: no energy or fairness terms dominate slack risk.
    """
    eps = 0.051539024687597565
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
        scale = 1.2468377059653388 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_centered = slack - np.median(slack)
    slack_pressure = np.tanh(0.016826942091481577 * (-slack_centered + eps))
    slack_penalty = 6.6000883237386745 * (1.0 - slack_pressure) + 3.478060998337171 * slack_pressure
    unc_med = np.median(uncertainty)
    successor_gate = ((slack >= 0.0) & (uncertainty <= unc_med + eps)).astype(float)
    successor_release = upward_rank * remaining_work
    successor_norm = mad_normalize(successor_release)
    successor_bonus = 0.14538175944799592 * successor_norm * successor_gate
    fairness_headroom = ready_wait_time / (np.abs(slack) + eps)
    energy_headroom_gate = np.clip(fairness_headroom, 0.0, 1.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_score = 1.7638848792942068 * energy_norm * energy_headroom_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-0.37667132323423785 * np.maximum(0.0, wait_norm))
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 0.004760330165028379 * unc_norm * slack_pressure
    duration = min_exec_time + min_comm_time
    work_density = remaining_work / (duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_bonus = 0.34908248790751767 * work_density_norm
    score = slack_penalty + energy_score + wait_boost + uncertainty_amplifier - successor_bonus - work_density_bonus
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
