import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's uncertainty-aware normalization and urgency capping
    with Parent 1's direct negative-slack penalty and upward-rank × remaining-work bottleneck coupling.
    
    Key structural improvements:
      - Dual-risk signal: explicit clipped negative slack penalty (Parent 1) *plus* feasibility-preserving urgency (Parent 2)
      - Bottleneck term now has two orthogonal components: (1) duration×upward_rank×remaining_work (Parent 2) AND (2) upward_rank×remaining_work (Parent 1),
        both amplified by uncertainty and urgency — enabling joint critical-path + successor-release pressure modeling
      - Uncertainty-weighted normalization applied uniformly across all terms (Parent 2), ensuring risk-context alignment
      - Anti-starvation via wait_saturation (Parent 2) *and* linear inverted wait (Parent 1) — combined as weighted sum for robust fairness
      - All operations bounded, finite, and deterministic; no branching beyond clipping/min/max.
    """
    eps = 0.02529050646959149
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
        dispersion = 0.6183566095859783 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = dispersion if dispersion > eps else fallback_range
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    norm_neg_slack = adaptive_normalize(neg_slack)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.535167374362189)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = adaptive_normalize(urgency_linear)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure_1 = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    bottleneck_pressure_2 = upward_rank * remaining_work
    unc_normalized = adaptive_normalize(uncertainty)
    amp_factor = np.power(1.0 + unc_normalized, 1.2713642019602853)
    bottleneck_pressure = (bottleneck_pressure_1 + bottleneck_pressure_2) * amp_factor
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (3.11003274969048 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_sigmoid = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait_sigmoid = adaptive_normalize(wait_sigmoid)
    norm_wait_linear = 1.0 - adaptive_normalize(ready_wait_time)
    norm_wait = 0.6939422028453301 * norm_wait_sigmoid + (1.0 - 0.6939422028453301) * norm_wait_linear
    score = 0.15905297503211654 * norm_neg_slack + norm_urgency + 0.7274833482506875 * norm_bottleneck + 0.8984013994837013 * norm_energy_eff - norm_wait + 1.0169777292420088 * adaptive_normalize(upward_rank * remaining_work)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
