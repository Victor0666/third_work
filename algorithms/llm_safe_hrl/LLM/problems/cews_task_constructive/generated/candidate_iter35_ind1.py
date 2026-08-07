import numpy as np
RULE_METADATA = {'structure_hash': 'fcaca75ad2bc21a5af68549f126d0ad1167d9a8014206cdb68b93967e9bd92d1', 'parameter_schema_hash': '393df7685ecb07b2e84c60faa94fe20634d3ba4865f039adaa5c5411a3d34cba', 'best_parameter_hash': '6aec0398f8f83341f7a196e1b1d64adc57885f27313581aaf0dc7e659f1c98c4', 'best_parameters': {'epsilon': 2.858050173322122e-06, 'energy_duration_ratio_weight': 1.8811737959703065, 'successor_bottleneck_coupling': 0.10261001936935629, 'uncertainty_dispersion_scale': 1.0802051590415622, 'urgency_cap_exponent': 0.4600300013892082, 'bottleneck_uncertainty_amplification': 1.1338710249167743, 'wait_saturation_scale': 1.4661429313663126, 'ddl_protection_activation_threshold': 0.798732720794342}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': 'f077aa333162da22a64509f15a85b3faf7c075c3d4a9370a0a920d54a9215152', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's robust deadline dominance and adaptive normalization
    with Parent 1's conditional DDL-protection gating — now restructured as *fractional activation*:
      - Uses `ddl_protection_activation_threshold` to determine if >threshold% of ready tasks are at risk,
        enabling bottleneck/energy terms only when system-wide deadline pressure is high.
      - Retains pre-normalized `neg_slack` as dominant hard-DDL penalty (no normalization distortion).
      - Keeps bounded sigmoid wait saturation for anti-starvation (smooth, saturating, no inversion).
      - Replaces percentile-based gating with global risk fraction — more stable under sparse/small-N cases.
      - All normalization uses uncertainty-aware adaptive_normalize() with fallback to range.
      - Final composition preserves ordering: hard-DDL > urgency > bottleneck > energy > fairness.
    """
    eps = 2.858050173322122e-06
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
        dispersion = 1.0802051590415622 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = np.where(dispersion > eps, dispersion, fallback_range)
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.4600300013892082)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = adaptive_normalize(urgency_linear)
    risk_fraction = np.mean(slack <= median_slack) if N > 0 else 0.0
    ddl_protection_active = risk_fraction >= 0.798732720794342
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_base = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    unc_normalized = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_base * np.power(1.0 + unc_normalized, 1.1338710249167743)
    norm_bottleneck = np.where(ddl_protection_active, adaptive_normalize(bottleneck_pressure), np.zeros_like(bottleneck_pressure))
    wait_scaled = ready_wait_time / (1.4661429313663126 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    score = neg_slack + norm_urgency + 0.10261001936935629 * norm_bottleneck + 1.8811737959703065 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
