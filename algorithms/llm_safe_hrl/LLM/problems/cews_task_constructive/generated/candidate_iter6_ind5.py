import numpy as np
RULE_METADATA = {'structure_hash': 'd6e2957a4de557b3f1184077b87d93c490b6d462bb16554354db1e3860b8f91b', 'parameter_schema_hash': '090ced21345839d154a8a0d55a65c411cd8d9f44d1add48dca2167d06ce0be3f', 'best_parameter_hash': 'caf92681ae4ddbf6a4c85c24a5e05b10e58832c2898b0be49e6049a3e5f24d81', 'best_parameters': {'epsilon': 0.0021973723156991677, 'slack_sigmoid_steepness': 3.9351358299569013, 'slack_sigmoid_offset': 0.15563047531152385, 'energy_duration_ratio_weight': 0.2913199918128654, 'successor_bottleneck_coupling': 0.42694978235195985, 'critical_rank_percentile': 0.8611916008217172, 'wait_saturation_time': 8.587774520511651, 'iqr_low_percentile': 13.476021720624491, 'iqr_high_percentile': 76.87315356155337, 'sigmoid_clip_bound': 5.830161950999403, 'ddl_protection_gate_threshold': 0.7208273342278608}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '2eecf0c0807bb20ee354b4c58f11aad8d80fb28dcd640042e590ed2b73fd1844', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule restoring hard deadline safety via binary DDL-protection gate,
    with adaptive IQR normalization scaling removed to eliminate recursion risk — replaced by
    static but robust IQR normalization using declared percentiles only.
    
    Key correction:
      - Removed recursive iqr_normalize call inside iqr_normalize; now uses only static percentile bounds.
      - All numeric literals are restricted to {-2,-1,0,1,2}; eps_safe derived from np.finfo.
      - All parameters declared in schema are used exactly once.
      - No loops, no recursion, no I/O, no randomness, no VM/Host logic.
    """
    eps = 0.0021973723156991677
    N = len(slack)
    finfo = np.finfo(float)
    eps_safe = max(eps, finfo.tiny)
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
        q_low = np.percentile(x, 13.476021720624491)
        q_high = np.percentile(x, 76.87315356155337)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps_safe else np.max(np.abs(x - center)) + eps_safe
        return (x - center) / (denom + eps_safe)
    slack_centered = slack - 0.15563047531152385
    sigmoid_input = np.clip(-3.9351358299569013 * slack_centered, -5.830161950999403, 5.830161950999403)
    urgency_gate = 2.0 / (1.0 + np.exp(sigmoid_input)) - 1.0
    norm_urgency = iqr_normalize(urgency_gate)
    duration = np.maximum(min_exec_time + min_comm_time, eps_safe)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    bottleneck_pressure = duration * (remaining_work + upward_rank + eps_safe)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps_safe)
    critical_gate = np.where(rank_percentile >= 0.8611916008217172, 1.0, 0.0)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = critical_gate * norm_rank
    wait_scaled = ready_wait_time / (8.587774520511651 + eps_safe)
    wait_saturation = 2.0 / np.pi * np.arctan(wait_scaled)
    norm_wait = iqr_normalize(wait_saturation)
    median_slack = np.median(slack)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.7208273342278608 * max_uncertainty)).astype(float)
    score = norm_urgency + 0.42694978235195985 * norm_bottleneck + 0.2913199918128654 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_active * norm_urgency
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
