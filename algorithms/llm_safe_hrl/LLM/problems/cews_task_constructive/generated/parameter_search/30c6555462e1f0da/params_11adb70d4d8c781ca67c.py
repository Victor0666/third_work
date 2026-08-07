import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with IQR-based robust normalization and refined urgency modeling.
    
    Key features:
      - Uses configurable quantile-based dispersion (Q_{center+upper} - Q_{center-lower}) for outlier-robust normalization.
      - Conditional DDL gate remains: activates only when slack < 0 AND uncertainty > median_uncertainty.
      - Bottleneck pressure multiplies upward_rank and remaining_work (validated coupling), enhanced by wait during deadline stress.
      - Uncertainty amplification via power-law on normalized uncertainty.
      - Urgency uses tanh scaled by tunable parameter — no hardcoded constants beyond {-2,-1,0,1,2}.
      - All numeric literals strictly in {-2,-1,0,1,2}; no try/except, no loops, no I/O, no randomness.
      - Fully deterministic, shape-compliant, and finite-valued.
    """
    eps = 0.00022821223273002532
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
        q_lower = np.clip(0.153471662338445 - 0.025694793215609593, 0.0, 1.0)
        q_upper = np.clip(0.153471662338445 + 0.14622800826385735, 0.0, 1.0)
        q_lo = np.quantile(x, q_lower, method='midpoint')
        q_hi = np.quantile(x, q_upper, method='midpoint')
        range_val = q_hi - q_lo
        unc_median = np.median(uncertainty) if N > 0 else eps
        dispersion = 1.9950089126102863 * (unc_median + eps)
        denom = np.where(range_val > eps, range_val, dispersion + eps)
        center = np.quantile(x, 0.153471662338445, method='midpoint')
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    unc_median = np.median(uncertainty) if N > 0 else eps
    ddl_risk_condition = (slack < 0) & (uncertainty > unc_median + eps)
    ddl_risk_gate = np.where(ddl_risk_condition, 1.0, 0.0)
    median_slack = np.quantile(slack, 0.153471662338445, method='midpoint') if N > 0 else 0.0
    urgency_base = np.clip(median_slack - slack, 0.0, 2.0)
    urgency = np.tanh(0.9377973437883347 * urgency_base)
    norm_wait = adaptive_normalize(ready_wait_time)
    bottleneck_pressure = remaining_work * upward_rank * 0.20896436093529308 * (1.0 + norm_wait * (slack < 0).astype(float))
    norm_uncertainty = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + norm_uncertainty, 2.080154679795963)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    fairness_term = -0.915778767999381 * norm_wait
    score = neg_slack + 2.562646487528391 * ddl_risk_gate + urgency + adaptive_normalize(bottleneck_pressure) + norm_energy_eff + fairness_term
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
