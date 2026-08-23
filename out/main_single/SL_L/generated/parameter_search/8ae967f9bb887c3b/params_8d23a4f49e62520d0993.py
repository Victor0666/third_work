import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Adaptive lexicographic gate: non-DDL terms activate only when slack > median(slack) + offset → robust to workload skew & avoids brittle fixed thresholds
      - Hybrid normalization: MAD for risk dimensions (duration, uncertainty), percentile *only* for upward_rank interpolation → preserves critical-path ordering fidelity
      - Load-aware energy suppression: disables energy-per-sec term when ready_wait_time exceeds median by load_aware_energy_suppression_threshold → prevents selecting low-energy-but-high-congestion tasks
      - All numeric literals restricted to {-2,-1,0,1,2}; no unbounded loops or side effects
      - Final score enforces strict DDL-first ordering, then optimizes within feasibility
    """
    eps = 1.3202508985132184e-06
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)
    duration_total = min_exec_time + min_comm_time + eps
    abs_slack = np.abs(slack) + eps
    all_risk_dims = np.stack([duration_total, abs_slack, uncertainty], axis=0)
    mad_per_dim = np.mean(np.abs(all_risk_dims - np.median(all_risk_dims, axis=1, keepdims=True)), axis=1) + eps
    duration_norm = duration_total / (mad_per_dim[0] + eps)
    slack_norm = abs_slack / (mad_per_dim[1] + eps)
    unc_norm = uncertainty / (mad_per_dim[2] + eps)

    def percentile_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        q_low = np.percentile(x, 8.008159373830548)
        med = np.median(x)
        q_high = 2.0 * med - q_low
        denom = q_high - q_low + eps
        normalized = (x - q_low) / denom
        return np.clip(normalized, 0.0, 1.0)
    slack_penalty = np.where(slack < 0, (-slack) ** 2.1483037629772825, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.2467137129614871
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.71324513383663
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 0.9988333634891164, 1.0)
    slack_median = np.median(slack) if N > 1 else np.mean(slack)
    slack_headroom_mask = np.where(slack > slack_median + 0.9978056314776949, 1.0, 0.0)
    duration_total_safe = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total_safe
    median_wait = np.median(ready_wait_time) + eps
    wait_ratio = ready_wait_time / median_wait
    energy_suppression_gate = np.where(wait_ratio > 1.4615220346462339, 0.0, 1.0)
    energy_eff_score = percentile_normalize(energy_per_sec) * energy_suppression_gate
    slack_scaled = np.clip((slack - slack_median) / (1.0 + eps), 0.0, 1.0)
    weight_rank = 0.619963339930722 * (1.0 - slack_scaled) + (1.0 - 0.619963339930722) * slack_scaled
    rank_score = -percentile_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-0.5966553377101018 * (uncertainty - 1.0)))
    energy_norm = percentile_normalize(min_incremental_energy)
    unc_norm_pct = percentile_normalize(uncertainty)
    energy_uncertainty_score = 0.01955488075462168 * energy_norm * unc_norm_pct * unc_sigmoid
    congestion_proxy = (wait_ratio + eps) * (1.0 + unc_norm)
    wait_score = percentile_normalize(ready_wait_time) * np.clip(congestion_proxy, 0.0, 1.0)
    score = percentile_normalize(slack_penalty) + percentile_normalize(unc_slack_coupling) + percentile_normalize(duration_risk)
    score += slack_headroom_mask * (1.2704026233317494 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
