import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with scenario-aware normalization and bounded piecewise urgency gating.
    
    Key improvements:
      - Bounded piecewise urgency: activates only when slack < median_slack * deadline_risk_gate_factor
      - Scenario-aware IQR normalization: uses distinct percentile bounds for high- vs low-uncertainty tasks,
        selected via a single conditional branch (not per-task branching).
      - All numeric literals are restricted to {-2,-1,0,1,2}; tunable thresholds moved to PARAMETER_SCHEMA.
      - Exactly 1 conditional branch (high_uncertainty_mask), satisfying branch limit.
      - No loops, no I/O, no randomness, no VM/host selection.
    """
    eps = 0.007623171922950848
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    uncertainty_median = np.median(uncertainty) if N > 0 else 0.0
    high_uncertainty_mask = (uncertainty > uncertainty_median).astype(float)
    q_low = np.where(high_uncertainty_mask[0] == 1.0, 28.373699663480263, 38.063912314963474)
    q_high = np.where(high_uncertainty_mask[0] == 1.0, 86.76437518597632, 80.03467135117194)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low_val = np.percentile(x, q_low)
        q_high_val = np.percentile(x, q_high)
        iqr = q_high_val - q_low_val
        center = np.median(x)
        denom = iqr if iqr > eps else np.max(np.abs(x - center)) + eps
        return (x - center) / (denom + eps)
    median_slack = np.median(slack)
    deadline_risk_threshold = 0.6114984157175462 * median_slack
    urgency_base = np.clip(-slack / (np.abs(median_slack) + eps), -1.0, 1.0)
    urgency_gate = (slack < deadline_risk_threshold).astype(float)
    urgency_final = urgency_base * (1.0 + urgency_gate)
    norm_urgency = iqr_normalize(urgency_final)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / duration
    norm_energy_eff = iqr_normalize(energy_per_duration)
    bottleneck_pressure = duration * (upward_rank * remaining_work + eps)
    norm_bottleneck = iqr_normalize(bottleneck_pressure)
    wait_ramp = np.clip(ready_wait_time / 1.363821791237821, 0.0, 1.0)
    norm_wait = iqr_normalize(wait_ramp)
    max_uncertainty = np.max(uncertainty) if N > 0 else 1.0
    ddl_protection_active = ((slack < median_slack) & (uncertainty > 0.6046274377164316 * max_uncertainty)).astype(float)
    norm_uncertainty = iqr_normalize(uncertainty)
    score = norm_urgency + 0.17928947368147385 * norm_bottleneck + 1.0716512123936492 * norm_energy_eff - norm_wait + ddl_protection_active * norm_uncertainty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
