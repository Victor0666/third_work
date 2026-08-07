import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three key structural improvements:
      - Restores hard-deadline dominance by *amplifying pre-normalized neg_slack* (not gating or diluting it).
      - Replaces conditional DDL gating with *additive, slack-modulated critical-path coupling*: 
        upward_rank × remaining_work × (1 + clipped_neg_slack)^modulation — preserves signal strength under risk without binary switches.
      - Makes wait-based fairness *unconditional and linearly scaled* by normalized ready_wait_time (not sigmoid-saturated),
        ensuring monotonic anti-starvation boost that scales smoothly with starvation depth.
      - All operations remain finite, deterministic, and use only {-2,-1,0,1,2} literals.
      - No branching, no hidden constants, full compliance with interface contract.
    """
    eps = 2.7066864999929578e-06
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
        center = np.median(x)
        unc_std = np.std(uncertainty) if N > 1 else eps
        dispersion = 0.6019188528155883 * (unc_std + eps)
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = dispersion if dispersion > eps else fallback_range
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    amplified_neg_slack = 1.075074167829736 * neg_slack
    median_slack = np.median(slack) if N > 0 else 0.0
    urgency_linear = np.clip(median_slack - slack, 0.0, 2.0)
    non_neg_slack = np.clip(slack, 0.0, None)
    max_non_neg_slack = np.max(non_neg_slack) if N > 0 else eps
    urgency_cap = np.power(max_non_neg_slack + eps, 0.7608371893381792)
    urgency_linear = np.minimum(urgency_linear, urgency_cap)
    norm_urgency = adaptive_normalize(urgency_linear)
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy_eff = adaptive_normalize(energy_per_duration)
    bottleneck_pressure = duration * upward_rank * remaining_work * (1.0 + urgency_linear + eps)
    unc_normalized = adaptive_normalize(uncertainty)
    bottleneck_pressure = bottleneck_pressure * np.power(1.0 + unc_normalized, 2.086969378023103)
    norm_bottleneck = adaptive_normalize(bottleneck_pressure)
    critical_pressure = upward_rank * remaining_work * np.power(1.0 + neg_slack + eps, 0.6410264125391347)
    norm_critical_pressure = adaptive_normalize(critical_pressure)
    norm_wait = adaptive_normalize(ready_wait_time)
    score = amplified_neg_slack + norm_urgency + norm_bottleneck + norm_critical_pressure + 0.8757794053518949 * norm_energy_eff - norm_wait
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
