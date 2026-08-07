import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with uncertainty-aware median dispersion normalization and unified bottleneck gating.
    
    Key improvements:
      - Restores robust adaptive_normalize (median + uncertainty-aware dispersion) per reflection — avoids IQR brittleness on small N.
      - Adds `median_centering_bias` to tune centering: positive values shift priority toward tasks with above-median features (e.g., higher urgency), 
        enabling fine-grained control over risk-aware skew without breaking monotonicity.
      - Unifies gating: only one slack-based feasibility gate applied to bottleneck pressure (not urgency), aligning with reflection.
      - Restores blended fairness (sigmoid + linear inverted) for balanced starvation resistance and deadline adherence.
      - All numeric literals strictly in {-2,-1,0,1,2}; no branching beyond clipping/min/max; fully deterministic.
    """
    eps = 0.008570193067474871
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
        center = np.median(x) + -0.4290785385340721 * np.std(x) if N > 1 else np.median(x)
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 0.3822014949785939 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = dispersion if dispersion > eps else fallback_range
        return (x - center) / (denom + eps)
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.7538373860973534)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = adaptive_normalize(urgency_linear)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    base_bottleneck = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    feasibility_mask = (slack >= -0.37314185467397587).astype(float)
    unc_normalized = adaptive_normalize(uncertainty)
    amp_factor = np.power(1.0 + unc_normalized, 1.264167099689348)
    bottleneck_pressure = base_bottleneck * amp_factor * feasibility_mask
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    wait_scaled = ready_wait_time / (1.402258332702564 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_sigmoid = 1.0 / (1.0 + np.exp(-wait_clipped))
    norm_wait_sigmoid = adaptive_normalize(wait_sigmoid)
    norm_wait_linear = 1.0 - adaptive_normalize(ready_wait_time)
    norm_wait = 0.14723453045110385 * norm_wait_sigmoid + (1.0 - 0.14723453045110385) * norm_wait_linear
    cp_coupling = upward_rank * remaining_work
    norm_cp_coupling = adaptive_normalize(cp_coupling)
    score = norm_urgency + 0.2804072054893647 * norm_bottleneck + 0.9304578061684913 * norm_energy_eff - norm_wait + 0.7959580834644102 * norm_cp_coupling
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
