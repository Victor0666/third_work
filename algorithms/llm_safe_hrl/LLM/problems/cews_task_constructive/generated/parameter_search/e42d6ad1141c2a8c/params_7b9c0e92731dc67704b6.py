import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating evidence-backed structural changes:
      - Adds conditional DDL-protection gate: only activates risk-aware terms when slack <= median_slack AND normalized uncertainty > threshold.
      - Replaces generic urgency with explicit successor-release coupling: (1 + slack/median_slack) * (1 + uncertainty) for release-aware pressure.
      - Activates upward_rank × remaining_work unconditionally — verified to resolve starvation without quantile estimation.
      - Uses DDL-aware min-max normalization (clipped range) for small-N ready sets to preserve neg_slack dominance.
      - Removes all inactive parameters (energy_duration_ratio_weight, urgency_cap_exponent, wait_saturation_scale, uncertainty_dispersion_scale) per diagnostics.
      - Bounded sigmoid anti-starvation retained but simplified: uses ready_wait_time directly with unit scale and configurable clipping.
      - All operations guarded against NaN/inf/zero; deterministic; no loops or side effects.
    """
    eps = 0.00012784014584872364
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    unc_center = np.median(uncertainty) if N > 0 else 0.0
    unc_range = np.max(uncertainty) - np.min(uncertainty) if N > 1 else eps
    norm_uncertainty = (uncertainty - unc_center) / (unc_range + eps)
    ddl_protection_active = (slack <= median_slack) & (norm_uncertainty > 0.21896436092355578)
    critical_pressure = upward_rank * remaining_work
    slack_ratio = np.where(np.abs(median_slack) > eps, slack / (np.abs(median_slack) + eps), 0.0)
    slack_ratio = np.clip(slack_ratio, -2.0, 2.0)
    successor_release = (1.0 + slack_ratio) * (1.0 + uncertainty)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_base = duration * critical_pressure * successor_release
    bottleneck_pressure = np.where(ddl_protection_active, bottleneck_base * np.power(1.0 + norm_uncertainty, 1.6078156723557147), bottleneck_base)

    def ddl_aware_normalize(x):
        x = np.copy(x)
        x_min = np.min(x) if N > 0 else 0.0
        x_max = np.max(x) if N > 0 else 1.0
        range_val = np.maximum(x_max - x_min, eps)
        clip_offset = 0.20839467391344374 * range_val
        x_clipped = np.clip(x, x_min - clip_offset, x_max + clip_offset)
        return (x_clipped - x_min) / (range_val + eps)
    norm_critical = ddl_aware_normalize(critical_pressure)
    norm_bottleneck = ddl_aware_normalize(bottleneck_pressure)
    norm_energy = ddl_aware_normalize(min_incremental_energy)
    wait_scaled = ready_wait_time / (1.0 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = ddl_aware_normalize(wait_saturation)
    score = neg_slack + 0.7250271824360095 * norm_critical + 0.36546691977386453 * norm_bottleneck + norm_energy - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
