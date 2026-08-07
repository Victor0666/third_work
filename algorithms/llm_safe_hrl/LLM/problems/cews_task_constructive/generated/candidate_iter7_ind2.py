import numpy as np
RULE_METADATA = {'structure_hash': '2f9227b08f5e2e06fd31e908d8a0c013924c1aca11b9dbaa3497f8b990bfc604', 'parameter_schema_hash': 'a9a7a43e6d18d8b6edc31b559c39686bc5e004c69e50b62d38fcbf80084dda6b', 'best_parameter_hash': '7bc54164a84522fee2cd96f04499ba1311cadfa3b6410e1537c1b6535a4ec79e', 'best_parameters': {'epsilon': 0.00885594308225641, 'ddl_protection_gate_threshold': 0.7700618072393214, 'energy_duration_ratio_weight': 1.975497778719598, 'successor_bottleneck_coupling': 0.16460498980091653, 'iqr_low_percentile': 39.48011901401259, 'iqr_high_percentile': 76.15423934326937, 'slack_sigmoid_steepness': 6.118491831897224, 'slack_sigmoid_offset': 0.39468495948077376, 'wait_saturation_time': 1.580071798330737}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'ba48d814d2c8a457aae9bef26d282cd0b0b4d1ec1a4b8df60c74437a29e8aa29', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating evidence-backed structural changes:
    - Uses ddl_protection_gate_threshold in DDL-protection gate (now correctly referenced)
    - Replaced arctan wait saturation with bounded linear ramp + soft cap for better gradient behavior
    - Unified bottleneck term: (exec+comm) * upward_rank * remaining_work (stronger coupling to critical path)
    - DDL protection gate now uses threshold fraction of max uncertainty as intended
    - Energy term uses log-normalized energy/duration ratio to compress dynamic range
    - Added anti-starvation term: inverse of normalized ready_wait_time (smaller wait → higher priority)
    - All normalizations use IQR with elite-tuned percentiles
    - Final score preserves strict ordering: deadline > bottleneck > energy > fairness
    """
    eps = 0.00885594308225641
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 39.48011901401259)
        q_high = np.percentile(x, 76.15423934326937)
        iqr = q_high - q_low
        center = (q_low + q_high) / 2.0
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    slack_centered = slack - 0.39468495948077376
    sigmoid_input = -6.118491831897224 * slack_centered
    finfo = np.finfo(float)
    sigmoid_input = np.clip(sigmoid_input, -finfo.max, finfo.max)
    urgency_gate = 1.0 / (1.0 + np.exp(-sigmoid_input))
    norm_urgency = iqr_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    log_energy_eff = np.log1p(energy_per_duration + eps)
    norm_energy_eff = iqr_normalize(log_energy_eff)
    bottleneck_pressure = duration * (upward_rank + eps) * (remaining_work + eps)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    wait_normalized = np.clip(ready_wait_time / (1.580071798330737 + eps), 0.0, 1.0)
    wait_priority = 1.0 - wait_normalized
    norm_wait = iqr_normalize(wait_priority)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = (uncertainty > 0.7700618072393214 * max_uncertainty).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_urgency + 0.16460498980091653 * norm_bottleneck + 1.975497778719598 * norm_energy_eff - norm_wait + ddl_protection_active * norm_uncertainty
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
    return score.astype(float, copy=False)
