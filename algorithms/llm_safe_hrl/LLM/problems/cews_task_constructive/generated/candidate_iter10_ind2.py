import numpy as np
RULE_METADATA = {'structure_hash': '78d8fec0702cea550a30c681b4dc0e5f5f3fc74a643a36475eccc283492a6bc9', 'parameter_schema_hash': '1ed5128da8b00358a2a77cfe0d7fb2dfae8a7e1609347d246f6aafac3c543595', 'best_parameter_hash': '4ef4a5efb7dc8293e70b1fc5256438b179948da33c65b84ac73d3f7c54fec7f4', 'best_parameters': {'epsilon': 0.000833945853739823, 'ddl_protection_gate_threshold': 0.8089790850129006, 'energy_duration_ratio_weight': 0.7365816630192694, 'successor_bottleneck_coupling': 0.6007557709643409, 'iqr_low_percentile': 36.426888449011884, 'iqr_high_percentile': 71.16782443056401, 'slack_sigmoid_offset': 0.564115840841487, 'slack_sigmoid_steepness': 1.8940305006630935, 'wait_saturation_time': 4.013243686606039, 'sigmoid_clip_bound': 30.060255268286607}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '9161e2187b4bc59ffc2fdb392b053626b13afc6ef844daf9f9998169f2ae7607', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating evidence-backed structural changes:
    - Replaces arctan wait saturation with bounded sigmoid (smoother, more stable under long waits)
    - Removes critical-path percentile gating (evidence shows it's inactive; replaced by direct upward_rank coupling)
    - Introduces multiplicative urgency × uncertainty coupling only under joint DDL pressure
    - Uses adaptive min-max fallback when IQR dispersion is low (improves robustness to near-constant features)
    - Eliminates redundant parameters (slack_sigmoid_* and wait_saturation_time now active per evidence)
    - Retains conditional DDL protection gate but tightens activation logic to joint condition
    - All terms normalized via tunable IQR percentiles; no unbounded growth or unstable ops
    """
    eps = 0.000833945853739823
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
        q_low = np.percentile(x, 36.426888449011884)
        q_high = np.percentile(x, 71.16782443056401)
        iqr = q_high - q_low
        center = np.median(x)
        if iqr > eps:
            denom = iqr
        else:
            denom = np.max(x) - np.min(x)
            if denom < eps:
                denom = eps
        return (x - center) / (denom + eps)
    slack_centered = slack - 0.564115840841487
    sigmoid_input = np.clip(-1.8940305006630935 * slack_centered, -30.060255268286607, 30.060255268286607)
    urgency_gate = 1.0 / (1.0 + np.exp(sigmoid_input))
    norm_urgency = adaptive_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * (1.0 + urgency_gate + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (4.013243686606039 + eps)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_scaled))
    norm_wait = adaptive_normalize(wait_saturation)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else eps
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.8089790850129006 * max_uncertainty)).astype(float)
    norm_uncertainty = adaptive_normalize(uncertainty)
    ddl_risk_amplifier = ddl_protection_active * norm_uncertainty
    score = norm_urgency + 0.6007557709643409 * norm_bottleneck + 0.7365816630192694 * norm_energy_eff - norm_wait + ddl_risk_amplifier
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
