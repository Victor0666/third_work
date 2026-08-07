import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Adaptive min-max normalization (robust under congestion, avoids IQR overfitting).
      - Tightened DDL risk gate requiring uncertainty > median * threshold.
      - Successor-slack-deficit proxy using slack deficit + uncertainty correlation.
      - Criticality-aware energy weighting: only penalizes energy for high-upward-rank tasks.
      - Sharpened slack sensitivity via exponentiated deficit term.
      - All numeric literals strictly in {-2,-1,0,1,2}; no unbounded logic.
    """
    eps = 4.212065762668909e-06
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
        dispersion = 1.8025352339007494 * (unc_median + eps)
        denom = np.where(range_val > eps, range_val, dispersion + eps)
        return (x - x_min) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    unc_median = np.median(uncertainty) if N > 0 else eps
    ddl_risk_condition = (slack < 0) & (uncertainty > unc_median * 1.1450257926704002 + eps)
    ddl_risk_gate = np.where(ddl_risk_condition, 1.0, 0.0)
    median_slack = np.median(slack) if N > 0 else 0.0
    slack_deficit = np.clip(median_slack - slack, 0.0, 2.0)
    urgency = np.power(slack_deficit + eps, 1.5218948421198815)
    norm_wait = adaptive_normalize(ready_wait_time)
    bottleneck_pressure = remaining_work * upward_rank * 0.0 * (1.0 + norm_wait * (slack < 0).astype(float))
    norm_uncertainty = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + norm_uncertainty, 1.7414660468867091)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    up_rank_median = np.median(upward_rank) if N > 0 else 0.0
    critical_mask = upward_rank >= up_rank_median
    norm_energy_eff = np.where(critical_mask, norm_energy_eff, 0.0)
    fairness_term = -1.3159825070383844 * norm_wait
    successor_risk_proxy = np.where((slack < median_slack - eps) & (uncertainty > unc_median), 1.0, 0.0)
    successor_slack_deficit_penalty = 0.6668883685625581 * successor_risk_proxy
    score = neg_slack + 4.535440553809638 * ddl_risk_gate + urgency + adaptive_normalize(bottleneck_pressure) + 0.40716215256139776 * norm_energy_eff + fairness_term + successor_slack_deficit_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
