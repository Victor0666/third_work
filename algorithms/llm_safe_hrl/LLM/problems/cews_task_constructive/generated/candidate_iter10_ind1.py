import numpy as np
RULE_METADATA = {'structure_hash': '926881a9026282587c74da0d263dcb1930fa403189206b0677fa37d6625603a1', 'parameter_schema_hash': '055591bf36490a6e4803eada4a1ae644cd81ed6cadc706be0fc56d17f6a4a83d', 'best_parameter_hash': '9335fba1d2ecea208f27e0991c14fa669bb3bd5d68018fb4da379799e52e3a4e', 'best_parameters': {'epsilon': 1.3234785266787446e-06, 'ddl_protection_gate_threshold': 0.7214945276559104, 'energy_duration_ratio_weight': 1.0555532333885103, 'successor_bottleneck_coupling': 0.6914040762841797, 'iqr_low_percentile': 39.99390660284756, 'iqr_high_percentile': 86.2142519470087, 'slack_sigmoid_steepness': 4.779809793046667, 'slack_sigmoid_offset': 0.9596568213265635, 'sigmoid_clip_bound': 6.220923486446488}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '20aeb3f2630e60636de3689b39b61a48f0337f4be3a751a2f128084f0495a67f', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating counterfactual evidence:
    - Removed inactive parameters (wait_saturation_time, critical_rank_percentile) per inactivity analysis
    - Replaced arctan wait saturation with bounded linear ramp to avoid over-smoothing under starvation
    - Simplified DDL protection to joint condition: tight slack AND high uncertainty (no percentile gating)
    - Unified risk coupling: multiplicative urgency × uncertainty instead of additive conditional term
    - Used adaptive min-max fallback when IQR dispersion is low (improves robustness to near-constant features)
    - Removed critical bonus (evidence shows it causes energy degradation without DDL improvement)
    - Prioritizes smooth, bounded, and numerically stable operations only.
    """
    eps = 1.3234785266787446e-06
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
        q_low = np.percentile(x, 39.99390660284756)
        q_high = np.percentile(x, 86.2142519470087)
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
    slack_centered = slack - 0.9596568213265635
    sigmoid_input = np.clip(-4.779809793046667 * slack_centered, -6.220923486446488, 6.220923486446488)
    urgency = 1.0 / (1.0 + np.exp(sigmoid_input))
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    if N == 1:
        median_wait = ready_wait_time[0]
    else:
        median_wait = np.median(ready_wait_time)
    wait_ramp = np.clip(ready_wait_time / (median_wait + eps), 0.0, 1.0)
    norm_wait = adaptive_normalize(wait_ramp)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_risk_gate = ((slack < median_slack) & (uncertainty > 0.7214945276559104 * max_uncertainty)).astype(float)
    risk_coupled = urgency * uncertainty
    norm_risk = adaptive_normalize(risk_coupled)
    score = norm_urgency + 0.6914040762841797 * norm_bottleneck + ddl_risk_gate * norm_risk + 1.0555532333885103 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
