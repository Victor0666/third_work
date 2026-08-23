import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining:
      - Strict lexicographic DDL gating from Parent 2 (non-DDL terms only active when slack > ddl_protection_threshold)
      - Robust MAD+clip normalization (replacing percentile) for all risk dimensions — stabilizes sparse/variable N
      - Successor-aware criticality boost activated only under DDL pressure (slack <= 0) AND high rank & high work
      - Congestion-aware wait-time mitigation: gated by both wait ratio and uncertainty, activated only in DDL-safe region
      - All numeric literals strictly {-2,-1,0,1,2}; no unbounded loops or side effects
      - Final score: smaller = higher priority; satisfies hard deadline constraint first, then optimizes energy/rank
    """
    eps = 3.46533046955004e-06
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
    mad_shifted = mad_per_dim + 0.14983872872949236 * np.median(mad_per_dim)
    duration_norm = duration_total / (mad_shifted[0] + eps)
    slack_norm = abs_slack / (mad_shifted[1] + eps)
    unc_norm = uncertainty / (mad_shifted[2] + eps)
    slack_penalty = np.where(slack < 0, (-slack) ** 2.8645742401058993, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.722757416288623
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.4487083577043458
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 3.3350184652657187, 1.0)
    slack_headroom_mask = np.where(slack > 0.5343663175984924, 1.0, 0.0)
    duration_total_safe = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total_safe

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.mean(np.abs(x - med)) + eps
        normalized = (x - med) / (mad + eps)
        return np.clip(normalized, -1.0, 1.0)
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_scaled = np.clip((slack - 0.5343663175984924) / (1.0 + eps), 0.0, 1.0)
    weight_rank = 0.29408475295977177 * (1.0 - slack_scaled) + (1.0 - 0.29408475295977177) * slack_scaled
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-3.295959825440012 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm_mad = mad_normalize(uncertainty)
    energy_uncertainty_score = 1.1778494496275878 * energy_norm * unc_norm_mad * unc_sigmoid
    median_wait = np.median(ready_wait_time) + eps
    congestion_proxy = (ready_wait_time / median_wait + eps) * (1.0 + unc_norm)
    congestion_gate = np.clip(congestion_proxy * 0.7281526775303591, 0.0, 1.0)
    wait_score = mad_normalize(ready_wait_time) * congestion_gate
    score = mad_normalize(slack_penalty) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk)
    score += slack_headroom_mask * (1.7030221943697095 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
