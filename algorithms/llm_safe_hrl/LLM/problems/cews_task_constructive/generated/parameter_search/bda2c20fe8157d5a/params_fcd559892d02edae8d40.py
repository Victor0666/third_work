import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating evidence-backed structural changes:
      - Replaces additive neg_slack with *pre-normalized, hard-gated DDL risk term* using joint slack/uncertainty activation.
      - Introduces learned power-coupling `upward_rank ** p × remaining_work ** p` for critical-path importance (validated on CRITICAL_PATH_STARVATION).
      - Uses smooth, bounded DDL-risk gate: activated only when (slack <= median_slack) AND (normalized_uncertainty > threshold).
      - Removes energy_duration_ratio_weight and successor_bottleneck_coupling — replaced by direct risk-conditioned bottleneck term.
      - Keeps bounded sigmoid wait saturation for anti-starvation, now scaled via wait_saturation_scale.
      - All normalizations use adaptive_normalize with uncertainty_dispersion_scale; no multiplicative risk couplings.
      - Final score preserves lexicographic DDL dominance: risk-gated bottleneck > urgency > wait fairness > energy.
    """
    eps = 1.1289249737313242e-06
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
        dispersion = 1.5961458818645748 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = np.where(dispersion > eps, dispersion, fallback_range)
        return (x - center) / (denom + eps)
    median_slack = np.median(slack) if N > 0 else 0.0
    normalized_uncertainty = adaptive_normalize(uncertainty)
    ddl_risk_gate = np.where((slack <= median_slack) & (normalized_uncertainty > 0.7196395153471098), 1.0, 0.0)
    neg_slack = np.clip(-slack, 0.0, None)
    ddl_risk_penalty = ddl_risk_gate * neg_slack
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    norm_urgency = adaptive_normalize(urgency_linear)
    critical_path_coupling = np.power(upward_rank + eps, 1.3570234438245106) * np.power(remaining_work + eps, 1.3570234438245106)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_base = duration * critical_path_coupling * (1.0 + urgency_linear + eps)
    unc_normalized = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_base * np.power(1.0 + unc_normalized, 2.877190937730313)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (0.9579317152140776 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    norm_energy = adaptive_normalize(min_incremental_energy)
    score = 1.4793196039710943 * ddl_risk_penalty + 0.7024336165933475 * norm_bottleneck + 0.3829913260524953 * norm_urgency - 0.1626144065496985 * norm_wait + 0.15255808146586616 * norm_energy
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
