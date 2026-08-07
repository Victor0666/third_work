import numpy as np
RULE_METADATA = {'structure_hash': '0e20b9d4147fcf108c81821d2addc60ca52b5b64ace2adb14e31985171e50aff', 'parameter_schema_hash': '1d8f32fa1bea6c3c2821d7becb2cf41f227ff5b38951d01fa57855ba7efa3635', 'best_parameter_hash': '1c9cd32ca81db48fa971f0bebba076e0cdb079e44513e5ddcd8a748fba871cd8', 'best_parameters': {'epsilon': 0.0020757840090362394, 'energy_duration_ratio_weight': 1.2600863558408835, 'successor_bottleneck_coupling': 0.5737886451616656, 'iqr_low_percentile': 29.39823897601134, 'iqr_high_percentile': 87.51091204127798, 'wait_clip_threshold': 174.2439538136621, 'slack_sigmoid_steepness': 3.506017943256645, 'bottleneck_slack_modulation': 1.090762936474661, 'sigmoid_clip_bound': 36.33329549078061}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '2648221fdd5516c2654d7e3e4ce728c8474383fa8d93f88f4f66ad341bd35005', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's robust linear urgency & log wait with Parent 1's critical-path awareness,
    enhanced by performance analysis insights:
      - Replaces linear urgency with bounded sigmoid penalty on slack for smoother, more discriminative lateness response.
      - Introduces slack-modulated bottleneck coupling: (exec+comm)*upward_rank*(1 + sigmoid(slack)) — amplifies pressure only when deadlines are tight.
      - DDL-protection gate now uses strict condition (slack < 0 AND uncertainty > median_uncertainty), eliminating fragile percentile thresholds.
      - Keeps log-clipped wait term for bounded anti-starvation; removes arctan and percentile gating.
      - Retains direct upward_rank bonus (no gating) for critical path emphasis.
      - All normalization uses elite-tuned IQR percentiles from Parent 2.
    """
    eps = 0.0020757840090362394
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
        q_low = np.percentile(x, 29.39823897601134)
        q_high = np.percentile(x, 87.51091204127798)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    slack_centered = slack - np.median(slack) if N > 1 else slack
    sigmoid_input = 3.506017943256645 * -slack_centered
    sigmoid_input = np.clip(sigmoid_input, -36.33329549078061, 36.33329549078061)
    urgency_penalty = 1.0 / (1.0 + np.exp(-sigmoid_input))
    norm_urgency = iqr_normalize(urgency_penalty)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    slack_sigmoid = 1.0 / (1.0 + np.exp(-3.506017943256645 * np.clip(-slack, 0.0, 36.33329549078061)))
    bottleneck_pressure = duration * (upward_rank + eps) * (1.0 + 1.090762936474661 * slack_sigmoid)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    norm_rank = iqr_normalize(upward_rank)
    critical_bonus = norm_rank
    wait_clipped = np.clip(ready_wait_time, 0.0, 174.2439538136621)
    wait_log = np.log1p(wait_clipped)
    norm_wait = iqr_normalize(wait_log)
    median_uncertainty = np.median(uncertainty) if N > 0 else 0.0
    ddl_protection_active = ((slack < 0.0) & (uncertainty > median_uncertainty)).astype(float)
    score = norm_urgency + 0.5737886451616656 * norm_bottleneck + 1.2600863558408835 * norm_energy_eff - critical_bonus - norm_wait + ddl_protection_active * norm_urgency
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
