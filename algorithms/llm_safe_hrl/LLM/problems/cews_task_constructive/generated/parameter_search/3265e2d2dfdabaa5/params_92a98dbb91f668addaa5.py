import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's uncertainty-aware normalization and feasibility-preserving urgency cap
    with Parent 1's direct clipped negative slack penalty and explicit upward_rank × remaining_work bottleneck term.
    
    Key structural improvements:
      - Dual-risk signal: combines clipped negative slack (hard DDL violation) AND feasibility-capped urgency (soft deadline proximity)
      - Unified bottleneck term: additive composition of (upward_rank × remaining_work) + (duration × upward_rank × remaining_work × (1+urgency))
        to capture both static critical-path importance and dynamic release pressure
      - Uncertainty-weighted normalization applied consistently across all terms
      - Anti-starvation via wait_saturation (sigmoid) AND linear inversion of normalized wait time (Parent 1's inv_wait), weighted separately
      - All operations guarded against NaN/inf/zero; no unbounded logic or hidden state.
    """
    eps = 0.0029600893531380834
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
        center = np.median(x)
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 0.8679877684437746 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = dispersion if dispersion > eps else fallback_range
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    norm_neg_slack = adaptive_normalize(neg_slack)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.9251617436573574)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = adaptive_normalize(urgency_linear)
    static_bottleneck = upward_rank * remaining_work
    norm_static_bottleneck = adaptive_normalize(static_bottleneck)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    dynamic_bottleneck = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    unc_normalized = adaptive_normalize(uncertainty)
    dynamic_bottleneck = dynamic_bottleneck * np.power(1.0 + unc_normalized, 1.1394166030072506)
    norm_dynamic_bottleneck = adaptive_normalize(dynamic_bottleneck)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    wait_scaled = ready_wait_time / (8.960600536769242 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait_sat = adaptive_normalize(wait_saturation)
    norm_wait_linear = adaptive_normalize(ready_wait_time)
    inv_wait_linear = 1.0 - norm_wait_linear
    score = 1.8309912611606445 * norm_neg_slack + norm_urgency + 0.5092950821034345 * norm_static_bottleneck + 0.4933254113035469 * norm_dynamic_bottleneck + 1.8933358094703217 * norm_energy_eff - norm_wait_sat - inv_wait_linear
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
