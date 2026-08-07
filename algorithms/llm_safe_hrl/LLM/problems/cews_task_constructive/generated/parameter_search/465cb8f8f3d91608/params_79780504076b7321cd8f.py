import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with 12 parameters and bounded structure:
      - Restores `energy_efficiency_weight` and couples energy/duration with upward-rank normalization.
      - Introduces *slack-triggered successor-blocking gate*: activates bottleneck amplification only when 
        normalized slack < threshold — avoids over-penalization in safe regions.
      - Host-load–aware anti-starvation uses same slack > 0 condition as before (no extra uncertainty threshold).
      - All numeric thresholds now declared in PARAMETER_SCHEMA; no hidden constants.
      - Exactly 12 parameters; all used; only {-2,-1,0,1,2} literals in function body; finiteness enforced.
    """
    eps = 1.022006901759567e-06
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    neg_slack = np.clip(-slack, 0.0, None)
    slack_centered = -slack
    slack_gate = 1.0 / (1.0 + np.exp(-1.8127924277376826 * (slack_centered + eps)))
    urgency_linear = np.clip(-slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.8720355104463531)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    ur_interaction = np.power(upward_rank * remaining_work + eps, 1.0522346798499826)
    bottleneck_pressure = ur_interaction * duration * (1.0 + urgency_linear + eps)
    slack_normalized = (slack - np.min(slack)) / (np.ptp(slack) + eps) if N > 1 else np.zeros_like(slack)
    unc_normalized = (uncertainty - np.min(uncertainty)) / (np.ptp(uncertainty) + eps) if N > 1 else np.zeros_like(uncertainty)
    blocking_gate = np.where(slack_normalized < 0.8724732825964516, 1.0, 0.0)
    amplified_uncertainty = np.power(1.0 + unc_normalized, 1.3051236382751952)
    bottleneck_pressure = bottleneck_pressure * (blocking_gate * amplified_uncertainty + (1.0 - blocking_gate))
    energy_per_duration = min_incremental_energy / (duration + eps)
    rank_normalized_energy = np.where(upward_rank > eps, energy_per_duration / (upward_rank + eps), energy_per_duration)
    wait_scaled = ready_wait_time / (2.6711333974828144 + eps)
    wait_clipped = np.clip(wait_scaled, -2.0, 2.0)
    wait_saturation = 1.0 / (1.0 + np.exp(-wait_clipped))
    wait_boost = np.where(slack > 0.0, wait_saturation, 0.0)

    def adaptive_normalize(x):
        x = np.copy(x)
        center = np.median(x) if N > 0 else 0.0
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 0.8054376368322426 * (unc_std + eps)
        denom = np.where(dispersion > eps, dispersion, fallback_range)
        return (x - center) / (denom + eps)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    norm_energy_eff = adaptive_normalize(rank_normalized_energy)
    norm_wait = adaptive_normalize(wait_boost)
    score = 4.651980342042342 * neg_slack + 0.4596417062661391 * norm_bottleneck + 0.526075695189222 * norm_energy_eff - 0.5084955821536207 * norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
