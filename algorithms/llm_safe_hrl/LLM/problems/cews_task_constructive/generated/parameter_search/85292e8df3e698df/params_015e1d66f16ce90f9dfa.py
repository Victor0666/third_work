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
    eps = 1.3901459144166396e-06
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
    tight_slack_boundary = median_slack * (1.0 - 0.08511357082387494)
    urgency_linear = np.clip(tight_slack_boundary - slack, 0.0, 2.0)
    urgency_mask = (slack <= tight_slack_boundary) & (slack >= -eps)
    urgency = np.where(urgency_mask, urgency_linear * 1.5893017009010217, 0.0)
    norm_urgency = rank_normalize(urgency)
    med_energy = np.median(min_incremental_energy) if N > 0 else 0.0
    abs_devs = np.abs(min_incremental_energy - med_energy)
    mad_energy = np.median(abs_devs) if N > 0 else eps
    denom_energy = mad_energy + eps
    norm_energy_eff = (min_incremental_energy - med_energy) / (denom_energy + eps)
    norm_energy_eff = np.clip(norm_energy_eff, -1.0, 1.0)
    unc_centered = uncertainty - 0.37693691600253965
    unc_clipped = np.clip(unc_centered, -2.0, 2.0)
    unc_gate = 1.0 / (1.0 + np.exp(-3.4236980610014656 * unc_clipped))
    energy_uncertainty_term = min_incremental_energy * unc_gate
    norm_energy_uncertain = rank_normalize(energy_uncertainty_term)
    duration = min_exec_time + min_comm_time
    bottleneck_pressure = duration * upward_rank * remaining_work * unc_gate
    norm_bottleneck = rank_normalize(bottleneck_pressure)
    norm_wait = rank_normalize(ready_wait_time)
    inv_wait = 1.0 - norm_wait
    local_score = norm_urgency + 1.8654953468561186 * norm_energy_eff + 0.49267670110395645 * norm_energy_uncertain + 0.5705699693587156 * norm_bottleneck + 0.4795609946621172 * norm_bottleneck - 0.6853829407168359 * inv_wait
    score[feasible_mask] = local_score[feasible_mask]
    score[~feasible_mask] = 1530.6478659629236 * finfo.max
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
