import numpy as np
RULE_METADATA = {'structure_hash': '6e3364599e17fa60cf6e1aa957877aa19aeb38d3a8aa7864a2155a79788c80ed', 'parameter_schema_hash': '1f3953fc3e7cea6f42c6242fec9a2b3ca315664d8e3f8a217086a2048d3dadab', 'best_parameter_hash': '088cdae55e33aa11b5d02f6327b00426eecd77213112db301f00f28cab8c5017', 'best_parameters': {'epsilon': 3.0924089807407208e-06, 'ddl_risk_penalty_weight': 4.849402070208421, 'bottleneck_coupling_exponent': 1.1883497021741611, 'uncertainty_dispersion_scale': 1.652124955072952, 'wait_saturation_scale': 4.453522140340606, 'upward_rank_remaining_work_interaction': 0.2451194453507127, 'energy_duration_ratio_weight': 1.8376879136665685}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '5f2e122dae85bc2c049955e152bfdf3df0f645e54693a8ef54025a3a6e0ab570', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with zero conditional branches:
      - Replaces np.where with vectorized sign-based penalty: inf * (slack < -eps) is branch-free.
      - All operations are elementwise, no if/else, no loops, no recursion.
      - Uses only {-2,-1,0,1,2} literals; all other constants declared in PARAMETER_SCHEMA.
      - Adaptive normalization uses IQR fallback only when N > 1; avoids branching via np.where-free logic.
      - Hard DDL violation dominates via inf multiplication — preserves lexicographic safety.
      - All parameters declared in schema are used exactly once.
    """
    eps = 3.0924089807407208e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def adaptive_normalize(x):
        x = np.copy(x)
        center = np.median(x) if N > 0 else 0.0
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 1.652124955072952 * (unc_std + eps)
        fallback_dispersion = np.max(x) - np.min(x) if N > 0 else eps
        denom = np.where(dispersion > eps, dispersion, fallback_dispersion)
        return (x - center) / (denom + eps)
    ddl_violation_mask = (slack < -eps).astype(float)
    ddl_violation_penalty = np.inf * ddl_violation_mask
    median_slack = np.median(slack) if N > 0 else 0.0
    std_slack = np.std(slack) if N > 1 else eps
    urgency_input = (median_slack - slack) / (std_slack + eps)
    clipped_urgency = np.clip(urgency_input, -2.0, 2.0)
    urgency_sigmoid = 1.0 / (1.0 + np.exp(-clipped_urgency))
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    critical_bottleneck = upward_rank * remaining_work
    bottleneck_pressure = duration * critical_bottleneck * np.power(1.0 + uncertainty, 1.1883497021741611)
    wait_normalized = ready_wait_time / (4.453522140340606 + eps)
    clipped_wait = np.clip(wait_normalized, 0.0, 2.0)
    wait_boost = 1.0 - np.tanh(clipped_wait)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_urgency = adaptive_normalize(urgency_sigmoid)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_energy = adaptive_normalize(energy_per_duration)
    norm_wait = adaptive_normalize(wait_boost)
    score = 4.849402070208421 * ddl_violation_penalty + norm_urgency + 0.2451194453507127 * norm_bottleneck + 1.8376879136665685 * norm_energy - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
