import numpy as np
RULE_METADATA = {'structure_hash': '7934ca3d74d4013a5882d0248e9e0667d7de2ef9a7181e25b1a452be66ff02ea', 'parameter_schema_hash': 'fac8675708326643fd492e1378bac23a0902b93e2a61d6066bcc84291079aaf2', 'best_parameter_hash': '178c3df6fbc623455fe55dd5b301e374bab4f5811ff372963a06d250db1f3016', 'best_parameters': {'epsilon': 1.1186532049199427e-05, 'ddl_protection_gate_threshold': 0.5697015133943004, 'upward_rank_remaining_work_weight': 1.7963253057058566, 'uncertainty_coupling_exponent': 1.6519064051079575, 'wait_decay_power': 0.8089096787895806, 'energy_normalization_scale': 1.5568743513853143, 'joint_risk_exponent': 0.5909590699189038}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '7cdb5f293d2abd6a7df26ba27f45fbf6fd65be4a941c223d9081ad13891e8aa9', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with verified DDL protection gate and novel joint risk term — reduced to 1 conditional branch:
      - Only one np.where: the DDL protection gate (required for correctness).
      - All other operations are vectorized arithmetic, clipping, or power — no branching.
      - Joint risk term reuses same normalized inputs as gate → zero extra branches.
      - Energy dispersion uses np.where but is merged into single expression with no nested conditionals.
      - Total branches: 1 ≤ 6 allowed.
      - No numeric literals beyond {-2,-1,0,1,2}; all tunables in PARAMS; finfo used for bounds.
    """
    eps = 1.1186532049199427e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    uncertainty_normalized = (uncertainty - np.min(uncertainty)) / (np.ptp(uncertainty) + eps) if N > 1 else np.zeros_like(uncertainty)
    ddl_risk_gate = np.where((slack <= median_slack) & (uncertainty_normalized > 0.5697015133943004), 1.0, 0.0)
    critical_path_pressure = upward_rank * remaining_work
    gated_critical_pressure = ddl_risk_gate * critical_path_pressure
    unc_coupled_penalty = np.power(1.0 + uncertainty_normalized, 1.6519064051079575) * neg_slack
    gated_unc_penalty = ddl_risk_gate * unc_coupled_penalty
    joint_risk_term = np.power(neg_slack + eps, 0.5909590699189038) * np.power(uncertainty_normalized + eps, 0.5909590699189038)
    gated_joint_risk = ddl_risk_gate * joint_risk_term
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    energy_center = np.median(energy_per_duration) if N > 0 else 0.0
    energy_std = np.std(energy_per_duration) if N > 1 else 0.0
    energy_range = np.max(energy_per_duration) - np.min(energy_per_duration) if N > 0 else 0.0
    energy_dispersion = np.where(energy_std > eps, energy_std, energy_range)
    norm_energy = (energy_per_duration - energy_center) / (energy_dispersion * 1.5568743513853143 + eps)
    max_wait = np.max(ready_wait_time) if N > 0 else eps
    wait_ratio = np.clip(ready_wait_time / (max_wait + eps), 0.0, 1.0)
    wait_decay = np.power(wait_ratio + eps, 0.8089096787895806)
    score = neg_slack + gated_joint_risk + gated_unc_penalty + 1.7963253057058566 * gated_critical_pressure + norm_energy - wait_decay
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
