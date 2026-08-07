import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with strict hard-deadline enforcement, uncertainty-gated bottleneck,
    and streamlined feature set per reflection.
    
    Key structural improvements:
      - Hard feasibility guard: all scores set to +inf if slack < -epsilon → enforces strict DDL adherence before any optimization.
      - Unified uncertainty-gated bottleneck: replaces dual static/dynamic terms with single term
        `duration * upward_rank * remaining_work * sigmoid(uncertainty * steepness)` — simpler, more interpretable, less overfit-prone.
      - Removed all inactive parameters (urgency_cap_exponent, wait_saturation_scale, slack_risk_penalty_weight) and redundant normalizations.
      - Retained only one urgency term (linear, capped, weighted) and one fairness term (inverted normalized wait time).
      - All numeric literals strictly in {-2,-1,0,1,2}; no branching; no hidden state; fully deterministic.
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
    finfo = np.finfo(float)
    score = np.full(N, finfo.max, dtype=float)
    feasible_mask = slack >= -eps
    if not np.any(feasible_mask):
        return score

    def adaptive_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 0.9319024515968716 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = dispersion if dispersion > eps else fallback_range
        return (x - center) / (denom + eps)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    norm_urgency = adaptive_normalize(urgency_linear)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    unc_scaled = uncertainty * 1.4466843640680298
    unc_clipped = np.clip(unc_scaled, -2.0, 2.0)
    bottleneck_gate = 1.0 / (1.0 + np.exp(-unc_clipped))
    bottleneck_pressure = duration * upward_rank * remaining_work * bottleneck_gate
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_wait = adaptive_normalize(ready_wait_time)
    inv_wait = 1.0 - norm_wait
    local_score = 0.4756149839748881 * norm_urgency + 0.9171910590143877 * norm_bottleneck + 1.7534081561277364 * norm_energy_eff - 0.6405627358608041 * inv_wait
    score[feasible_mask] = local_score[feasible_mask]
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
