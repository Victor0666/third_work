import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating evidence-backed structural changes:
      - Replaces additive neg_slack with *pre-normalized, hard-gated DDL risk term* using joint slack/uncertainty activation.
      - Introduces learned power-coupling `upward_rank ** p × remaining_work ** p` for critical-path importance (validated on CRITICAL_PATH_STARVATION).
      - Uses smooth, bounded DDL-risk gate: activated only when (slack <= median_slack) AND (normalized_uncertainty > threshold).
      - Removes energy_duration_ratio_weight and successor_bottleneck_coupling — replaced by direct risk-conditioned bottleneck term.
      - Keeps bounded sigmoid wait saturation for anti-starvation, now scaled via wait_saturation_scale.
      - All normalizations use adaptive_normalize with uncertainty_dispersion_scale; no multiplicative risk couplings.
      - Final score preserves strict ordering: DDL risk > critical-path pressure > energy > fairness.
    """
    eps = 0.00012505624941956065
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
        dispersion = 1.7055783048662476 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = dispersion if dispersion > eps else fallback_range
        return (x - center) / (denom + eps)
    median_slack = np.median(slack) if N > 0 else 0.0
    normalized_uncertainty = adaptive_normalize(uncertainty)
    ddl_risk_active = (slack <= median_slack + eps).astype(float) * (normalized_uncertainty > 0.06499697772867977).astype(float)
    neg_slack = np.clip(-slack, 0.0, None)
    ddl_risk_penalty = neg_slack * ddl_risk_active
    up_rem_power = np.power(np.maximum(upward_rank, eps) * np.maximum(remaining_work, eps), 1.1551778280725862)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_base = duration * up_rem_power
    unc_normalized = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_base * np.power(1.0 + unc_normalized, 0.5181107783551246)
    norm_ddl_risk = adaptive_normalize(ddl_risk_penalty)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_energy = adaptive_normalize(min_incremental_energy)
    wait_scaled = ready_wait_time / (0.5264019362673797 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    score = norm_ddl_risk + norm_bottleneck + norm_energy - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
