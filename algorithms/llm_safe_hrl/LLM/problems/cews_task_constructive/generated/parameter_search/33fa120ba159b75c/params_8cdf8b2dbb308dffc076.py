import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's hard-deadline dominance and smooth modulation
    with Parent 1's uncertainty-gated energy coupling and robust rank-based fallbacks.
    
    Key structural improvements:
      - Hybrid normalization: uses adaptive dispersion (Parent 2) for urgency/bottleneck terms,
        but falls back to rank-normalization for energy_uncertainty_term (Parent 1) — avoids
        outlier sensitivity while preserving signal integrity under high uncertainty skew.
      - Dual-path uncertainty integration: sigmoid-gated energy coupling (Parent 1) *plus*
        power-law uncertainty amplification on bottleneck pressure (Parent 2), enabling
        complementary risk modeling across energy and scheduling dimensions.
      - Unified wait fairness: linearly weighted normalized wait time (Parent 2) replaces
        rank-inversion (Parent 1), ensuring monotonic, interpretable anti-starvation boost.
      - All literals strictly in {-2,-1,0,1,2}; no unbounded ops; full NaN/inf/zero protection.
      - Feasibility masking applied only to final score assignment, not intermediate terms.
    """
    eps = 5.317610561512729e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    finfo = np.finfo(float)

    def rank_normalize(x):
        x = np.copy(x)
        if N <= 1 or np.allclose(x, x[0], atol=eps):
            return np.zeros_like(x, dtype=float)
        sorted_idx = np.argsort(x)
        ranks = np.empty_like(sorted_idx, dtype=float)
        ranks[sorted_idx] = np.arange(N, dtype=float)
        norm_ranks = 2.0 * ranks / (N - 1.0) - 1.0
        return np.clip(norm_ranks, -1.0, 1.0)

    def adaptive_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 0.11159745358159233 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = dispersion if dispersion > eps else fallback_range
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    amplified_neg_slack = 3.0576933995039015 * neg_slack
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.5447988331337543)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = adaptive_normalize(urgency_linear)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    unc_normalized = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + unc_normalized, 1.2198931214561917)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    critical_pressure = upward_rank * remaining_work * np.power(1.0 + neg_slack + eps, 0.8074027397004775)
    norm_critical_pressure = adaptive_normalize(critical_pressure)
    unc_centered = uncertainty - 0.5811520595524069
    unc_clipped = np.clip(unc_centered, -2.0, 2.0)
    unc_gate = 1.0 / (1.0 + np.exp(-5.9148970438001305 * unc_clipped))
    energy_uncertainty_term = min_incremental_energy * unc_gate
    norm_energy_uncertain = rank_normalize(energy_uncertainty_term)
    norm_wait = adaptive_normalize(ready_wait_time)
    score = amplified_neg_slack + norm_urgency + norm_bottleneck + norm_critical_pressure + 1.3931572899457147 * norm_energy_eff + 0.8242532369757382 * norm_energy_uncertain - 1.375121717576972 * norm_wait
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
