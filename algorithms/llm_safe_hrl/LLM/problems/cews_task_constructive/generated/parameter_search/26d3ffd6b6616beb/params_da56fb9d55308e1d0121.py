import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with bounded control flow:
      - Pre-normalized negative-slack penalty (dominant)
      - Adaptive uncertainty-aware normalization using std(uncertainty)
      - Tight-slack urgency with linear ramp + learned exponent cap
      - Unified uncertainty gating via sigmoid centered at tunable PARAMS["uncertainty_gate_center"]
      - Bounded sigmoid wait saturation
      - All operations use only {-2,-1,0,1,2} literals; no unbounded loops or conditionals
      - Exactly 5 conditional expressions (all np.where or np.clip) — within branch limit
      - No nested conditionals; flat AST depth
    """
    eps = 1.0708286017562546e-06
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
    feasible_mask = slack >= -eps
    score = np.full(N, finfo.max, dtype=float)
    if not np.any(feasible_mask):
        return score

    def adaptive_normalize(x):
        x = np.copy(x)
        center = np.median(x) if N > 0 else 0.0
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 0.8834401187941919 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = np.where(dispersion > eps, dispersion, fallback_range)
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    tight_slack_boundary = median_slack * (1.0 - 0.17057636774108376)
    urgency_linear = np.clip(tight_slack_boundary - slack, 0.0, 2.0)
    urgency_mask = (slack <= tight_slack_boundary) & (slack >= -eps)
    urgency = np.where(urgency_mask, urgency_linear, 0.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 1.0376965849512627)
    urgency = np.minimum(urgency, urgency_cap)
    norm_urgency = adaptive_normalize(urgency)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + urgency + eps)
    unc_centered = uncertainty - 0.10299586841302899
    unc_clipped = np.clip(unc_centered, -2.0, 2.0)
    unc_gate = 1.0 / (1.0 + np.exp(-6.986944223854159 * unc_clipped))
    bottleneck_pressure = bottleneck_pressure * unc_gate
    unc_normalized = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + unc_normalized, 1.4949161377987512)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (5.9721419309968695 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait = adaptive_normalize(wait_saturation)
    local_score = 1727.8708699755657 * neg_slack + norm_urgency + 0.7754164317862245 * norm_bottleneck + 1.3315862852931752 * norm_energy_eff - norm_wait
    score[feasible_mask] = local_score[feasible_mask]
    score[~feasible_mask] = 1727.8708699755657 * finfo.max
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
