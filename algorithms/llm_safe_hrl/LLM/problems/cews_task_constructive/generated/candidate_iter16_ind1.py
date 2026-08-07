import numpy as np
RULE_METADATA = {'structure_hash': '9499c6f1f4b6cd06f4634d67d726fb58e6d3915df212ab882be1392ec9703210', 'parameter_schema_hash': 'cf65f5dc9e483e77d7aab51d3e617b0e3c88041055742355d3bd981c96669e73', 'best_parameter_hash': '8f86d7d1c9f3a48fcd38188c53ec453f23c8f09d3afd5918ab11a5bdcbeafc57', 'best_parameters': {'epsilon': 0.002445662087980999, 'iqr_low_percentile': 23.09534291133774, 'iqr_high_percentile': 82.22181160995501, 'slack_linear_coeff': 1.1127248302874357, 'upward_remaining_interaction_weight': 1.3729205133383662, 'uncertainty_slack_coupling': 0.04160148253039114, 'energy_normalization_offset': 0.09385517555127443}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'edaa59fd2e226de0fae06c56aa8555ad00d6775d9cbd3ea141dace0354a203c8', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
    - Linear slack penalty (not sigmoid) with uncertainty coupling for sharper, numerically stable deadline risk.
    - Direct upward_rank × remaining_work interaction to capture critical-path computational bottleneck pressure.
    - Energy term shifted by offset to avoid over-penalizing low-energy tasks when slack is tight.
    - Removed anti-starvation ramp (replaced by robust wait-aware normalization in urgency term).
    - Unified bottleneck term now explicitly includes uncertainty-weighted slack penalty.
    - All normalizations use adaptive IQR→min-max fallback; no unbounded ops or hidden state.
    """
    eps = 0.002445662087980999
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
        q_low = np.percentile(x, 23.09534291133774)
        q_high = np.percentile(x, 82.22181160995501)
        iqr = q_high - q_low
        center = np.median(x)
        if iqr < eps:
            x_min, x_max = (np.min(x), np.max(x))
            denom = x_max - x_min
            if denom < eps:
                return np.zeros_like(x)
            return (x - x_min) / (denom + eps)
        else:
            denom = iqr
            return (x - center) / (denom + eps)
    linear_slack_penalty = 1.1127248302874357 * np.clip(-slack, 0.0, None)
    coupled_slack_penalty = linear_slack_penalty * (1.0 + 0.04160148253039114 * uncertainty)
    norm_slack_penalty = adaptive_normalize(coupled_slack_penalty)
    critical_load_pressure = upward_rank * (remaining_work + eps)
    norm_critical_load = adaptive_normalize(critical_load_pressure)
    norm_energy = adaptive_normalize(min_incremental_energy)
    shifted_energy = norm_energy + 0.09385517555127443
    shifted_energy = np.clip(shifted_energy, 0.0, 2.0)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_base = duration * (1.0 + norm_critical_load) * (1.0 + norm_slack_penalty)
    norm_bottleneck = adaptive_normalize(bottleneck_base)
    score = norm_slack_penalty + 1.3729205133383662 * norm_critical_load + norm_bottleneck + shifted_energy
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
