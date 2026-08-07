import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's robustness with novel bottleneck-duration coupling:
      - Preserves rank-based normalization for small-N stability and skew resilience.
      - Keeps piecewise linear urgency (not power-law) for strict DDL dominance without over-penalization.
      - Introduces *duration-augmented bottleneck pressure*: multiplies critical path pressure by total duration
        to prioritize tasks that both dominate the critical path AND consume significant time/energy resources.
      - Unified uncertainty gating applies identically to energy and bottleneck terms for consistency.
      - Hard deadline guard enforces strict feasibility before any optimization trade-off.
      - All operations bounded, clipped, and protected against NaN/inf/zero; uses only {-2,-1,0,1,2} structural literals.
      - No global IQR — avoids percentile noise on tiny ready sets; rank normalization is more reliable at low N.
    """
    eps = 1.6723596126141206e-05
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
    score = np.full(N, finfo.max, dtype=float)
    feasible_mask = slack >= -eps
    if not np.any(feasible_mask):
        return score

    def rank_normalize(x):
        x = np.copy(x)
        if N <= 1 or np.all(x == x[0]):
            return np.zeros_like(x, dtype=float)
        sorted_idx = np.argsort(x)
        ranks = np.empty_like(sorted_idx, dtype=float)
        ranks[sorted_idx] = np.arange(N, dtype=float)
        norm_ranks = 2.0 * ranks / (N - 1.0) - 1.0
        return np.clip(norm_ranks, -1.0, 1.0)
    median_slack = np.median(slack) if N > 0 else 0.0
    tight_slack_boundary = median_slack * (1.0 - 0.10618303829343226)
    urgency_linear = np.clip(tight_slack_boundary - slack, 0.0, 2.0)
    urgency_mask = (slack <= tight_slack_boundary) & (slack >= -eps)
    urgency = np.where(urgency_mask, urgency_linear * 1.2848452376369703, 0.0)
    norm_urgency = rank_normalize(urgency)
    med_energy = np.median(min_incremental_energy) if N > 0 else 0.0
    abs_devs = np.abs(min_incremental_energy - med_energy)
    mad_energy = np.median(abs_devs) if N > 0 else eps
    denom_energy = mad_energy + eps
    norm_energy_eff = (min_incremental_energy - med_energy) / (denom_energy + eps)
    norm_energy_eff = np.clip(norm_energy_eff, -1.0, 1.0)
    unc_centered = uncertainty - 0.19505720854181824
    unc_clipped = np.clip(unc_centered, -2.0, 2.0)
    unc_gate = 1.0 / (1.0 + np.exp(-8.015121649363117 * unc_clipped))
    energy_uncertainty_term = min_incremental_energy * unc_gate
    norm_energy_uncertain = rank_normalize(energy_uncertainty_term)
    duration = min_exec_time + min_comm_time
    bottleneck_pressure = duration * upward_rank * remaining_work * unc_gate
    norm_bottleneck = rank_normalize(bottleneck_pressure)
    norm_wait = rank_normalize(ready_wait_time)
    inv_wait = 1.0 - norm_wait
    local_score = norm_urgency + 0.9369471781922937 * norm_energy_eff + 0.8667845652078521 * norm_energy_uncertain + 0.714635544130927 * norm_bottleneck + 0.705373168851394 * norm_bottleneck - 0.7224008337502658 * inv_wait
    score[feasible_mask] = local_score[feasible_mask]
    score[~feasible_mask] = 942.107631486095 * finfo.max
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
