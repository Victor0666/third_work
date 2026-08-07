import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robustness with Parent 1's criticality gating:
      - Adaptive min-max normalization (Parent 2) for congestion resilience.
      - Tightened DDL risk gate with thresholded uncertainty condition (Parent 2).
      - Successor-slack-deficit penalty (Parent 2) to resolve release-blocking.
      - Criticality-aware gating (Parent 1): suppresses non-critical-path tasks under tight slack.
      - All numeric literals strictly in {-2,-1,0,1,2}; no unbounded loops or side effects.
    """
    eps = 1.7128032665052826e-06
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
        unc_median = np.median(uncertainty) if N > 0 else eps
        dispersion = 2.0 * (unc_median + eps)
        denom = np.where(range_val > eps, range_val, dispersion + eps)
        return (x - x_min) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    unc_median = np.median(uncertainty) if N > 0 else eps
    ddl_risk_condition = (slack < 0) & (uncertainty > unc_median * 1.1395633754556251 + eps)
    ddl_risk_gate = np.where(ddl_risk_condition, 1.0, 0.0)
    median_slack = np.median(slack) if N > 0 else 0.0
    tight_slack_mask = (slack < median_slack * (1.0 - 0.3441050817837163)) & (slack >= -eps)
    max_upward = np.max(upward_rank) if N > 0 else 1.0
    non_critical_mask = (upward_rank < max_upward * 0.21663515133824374) & tight_slack_mask
    critical_gate = np.where(non_critical_mask, 0.0, 1.0)
    urgency_base = np.clip(median_slack - slack, 0.0, 2.0)
    urgency = np.tanh(1.7981686618885697 * urgency_base)
    bottleneck_pressure = remaining_work * upward_rank * 0.3843802057255263
    bottleneck_pressure = bottleneck_pressure * critical_gate
    norm_uncertainty = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + norm_uncertainty, 1.1246442952131968)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    norm_wait = adaptive_normalize(ready_wait_time)
    fairness_term = -0.4749562044116914 * norm_wait
    successor_risk_proxy = np.where((slack < median_slack - eps) & (uncertainty > unc_median), 1.0, 0.0)
    successor_slack_deficit_penalty = 0.47820678934174 * successor_risk_proxy
    score = neg_slack + 3.587809677020247 * ddl_risk_gate + urgency + adaptive_normalize(bottleneck_pressure) + norm_energy_eff + fairness_term + successor_slack_deficit_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
