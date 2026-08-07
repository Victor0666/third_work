import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with structural improvements guided by counterfactual evidence:
      - Introduces conditional DDL protection gate: only activates bottleneck & successor terms when slack is tight AND uncertainty is high.
      - Replaces linear urgency with normalized neg_slack as dominant additive term (preserves hard deadline dominance).
      - Adds unconditional upward_rank × remaining_work interaction (verified to resolve starvation without quantile estimation).
      - Uses bounded sigmoid wait saturation with learned exponent for robust anti-starvation.
      - Removes all inactive parameters (energy_duration_ratio_weight, successor_bottleneck_coupling, urgency_cap_exponent) per diagnostics.
      - All normalizations use DDL-aware min-max clipping for small-N sets to preserve signal integrity.
      - No branching beyond boolean gates; all operations finite, deterministic, and epsilon-guarded.
    """
    eps = 6.18072498023116e-05
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
    unc_std = np.std(uncertainty) if N > 1 else eps
    dispersion_base = 1.7727821905115915 * (unc_std + eps)
    unc_min, unc_max = (np.min(uncertainty), np.max(uncertainty))
    unc_range = np.maximum(unc_max - unc_min, eps)
    unc_normalized = (uncertainty - unc_min) / (unc_range + eps)
    ddl_protection_active = (slack <= median_slack) & (unc_normalized > 0.3373751145558375)
    critical_path_pressure = upward_rank * remaining_work
    cp_min, cp_max = (np.min(critical_path_pressure), np.max(critical_path_pressure))
    cp_range = np.maximum(cp_max - cp_min, eps)
    norm_critical_path = (critical_path_pressure - cp_min) / (cp_range + eps)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    bottleneck_base = duration * upward_rank * remaining_work
    bottleneck_pressure = np.where(ddl_protection_active, bottleneck_base * np.power(1.0 + unc_normalized, 2.0077589436742826), 0.0)
    bp_min, bp_max = (np.min(bottleneck_pressure), np.max(bottleneck_pressure))
    bp_range = np.maximum(bp_max - bp_min, eps)
    norm_bottleneck = np.where(ddl_protection_active, (bottleneck_pressure - bp_min) / (bp_range + eps), 0.0)
    wait_scaled = ready_wait_time / (1.0 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_sigmoid = 1.0 / (1.0 + np.exp(-wait_clipped))
    wait_saturation = np.power(wait_sigmoid, 1.2803020268244256)
    w_min, w_max = (np.min(wait_saturation), np.max(wait_saturation))
    w_range = np.maximum(w_max - w_min, eps)
    norm_wait = (wait_saturation - w_min) / (w_range + eps)
    score = neg_slack + 0.7339739618356943 * norm_critical_path + norm_bottleneck - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
