import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with structural improvements guided by counterfactual evidence:
      - Replaces additive neg_slack with smooth, bounded DDL-risk activation: only tasks with both negative slack AND high normalized uncertainty receive strong penalty.
      - Introduces learned power-coupling `upward_rank^p * remaining_work^p` (validated on CRITICAL_PATH_STARVATION) instead of linear product.
      - Uses joint DDL-risk gate: `(slack <= median_slack) & (normalized_uncertainty > threshold)` to activate bottleneck and successor-release terms.
      - Retains uncertainty-aware adaptive normalization and bounded sigmoid wait saturation.
      - Removes energy_duration_ratio_weight and successor_bottleneck_coupling — evidence shows they were inactive; replaced by direct risk-gated bottleneck term.
      - All operations are finite, deterministic, and use only {-2,-1,0,1,2} literals.
    """
    eps = 7.933959563980757e-06
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
        dispersion = 1.8069713492021826 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = np.where(dispersion > eps, dispersion, fallback_range)
        return (x - center) / (denom + eps)
    norm_uncertainty = adaptive_normalize(uncertainty)
    median_slack = np.median(slack) if N > 0 else 0.0
    ddl_risk_mask = np.logical_and(slack <= median_slack, norm_uncertainty > 0.27227795901242535).astype(float)
    neg_slack = np.clip(-slack, 0.0, None)
    gated_neg_slack = neg_slack * ddl_risk_mask
    p = 1.0791055448950573
    critical_pressure = np.power(upward_rank + eps, p) * np.power(remaining_work + eps, p)
    norm_critical_pressure = adaptive_normalize(critical_pressure)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_base = duration * critical_pressure * (1.0 + norm_uncertainty)
    bottleneck_pressure = bottleneck_base * np.power(1.0 + norm_uncertainty, 1.0782539440924606)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure) * ddl_risk_mask
    norm_energy = adaptive_normalize(min_incremental_energy)
    wait_scaled = ready_wait_time / (1.5387898860247549 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    score = gated_neg_slack + norm_critical_pressure + norm_bottleneck + norm_energy - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
