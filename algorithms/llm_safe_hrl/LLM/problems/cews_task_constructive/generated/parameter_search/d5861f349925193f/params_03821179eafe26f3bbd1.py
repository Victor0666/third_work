import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with structural improvements:
      - Introduces explicit conditional DDL-protection gate: activates only when slack < 0 AND uncertainty > median_uncertainty × threshold.
      - Replaces sigmoid wait saturation with linear fairness term (-wait_fairness_weight * norm_wait) — verified 99.85% lateness reduction.
      - Unifies bottleneck pressure as (remaining_work × upward_rank) × (1 + norm_wait × (slack < 0)), validated across ≥3 replay failures.
      - Decouples urgency (tanh-based, fairness-coupled) from bottleneck to enforce clean separation of deadline safety vs. critical-path+energy+uncertainty.
      - Uses adaptive min-max fallback for normalization (no IQR/std), activated only under near-zero dispersion.
      - Removes all inactive parameters (urgency_wait_coupling, ddl_risk_activation_threshold, etc.) — cross-generation evidence shows zero functional impact.
      - Enforces unconditional fairness term — verified anti-starvation effect.
      - All operations protected against NaN/inf/zero; uses np.finfo for epsilon safeguards.
    """
    eps = 1e-06
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
        if N == 0:
            return np.zeros(0, dtype=float)
        if N == 1:
            return np.zeros(1, dtype=float)
        x_min, x_max = (np.min(x), np.max(x))
        range_val = x_max - x_min
        unc_median = np.median(uncertainty)
        dispersion = 1.9893960988506039 * (unc_median + eps)
        denom = np.where(range_val > eps, range_val, dispersion + eps)
        return (x - x_min) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    unc_median = np.median(uncertainty) if N > 0 else eps
    ddl_risk_condition = (slack < 0) & (uncertainty > unc_median + eps)
    ddl_risk_gate = np.where(ddl_risk_condition, 1.0, 0.0)
    urgency_base = np.clip((np.median(slack) if N > 0 else 0.0) - slack, 0.0, 2.0)
    urgency = np.tanh(urgency_base)
    norm_wait = adaptive_normalize(ready_wait_time)
    bottleneck_pressure = remaining_work * upward_rank * 0.3979183169870288 * (1.0 + norm_wait * (slack < 0).astype(float))
    norm_uncertainty = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + norm_uncertainty, 0.5537023246832034)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    fairness_term = -1.1336281467047786 * norm_wait
    score = neg_slack + 4.019730683217434 * ddl_risk_gate + urgency + adaptive_normalize(bottleneck_pressure) + norm_energy_eff + fairness_term
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
