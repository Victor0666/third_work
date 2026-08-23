import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Strict lexicographic DDL gating via hard `slack > 0` mask (simplified from prior threshold)
      - Critical path release prioritization: `upward_rank * remaining_work` interaction scaled by criticality_boost under DDL pressure
      - Power-law anti-starvation: `ready_wait_time ** wait_time_decay_exponent`, gated by slack headroom and normalized robustly
      - MAD-based normalization per dimension instead of percentile (more stable for small N)
      - Explicit slack sign-aware scaling: linear interpolation from 0 to 1 over [slack_min_bound, slack_max_bound] for rank balance
      - All divisions guarded by eps; all inf/nan replaced before computation; no in-place mutation
      - Final score structure: [DDL violation penalty] + [critical path urgency] + [gated non-DDL efficiency & fairness]
    """
    eps = 2.7741641146188655e-09
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

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        dev = np.abs(x - med)
        mad = np.median(dev) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 3.707427285235739, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.9454845248581238
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.8840348582134545
    is_ddl_constrained = slack <= 0.0
    successor_release_score = upward_rank * remaining_work * 1.9101623643557646
    successor_release_norm = -mad_normalize(successor_release_score)
    successor_release_contribution = np.where(is_ddl_constrained, successor_release_norm, 0.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -8.900308768532454
    slack_ub = 45.281618081770446
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.675689729210576 + (1.0 - 0.675689729210576) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-3.0997402894384476 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 1.2704834110664585 * energy_norm * unc_norm * unc_sigmoid
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 0.6910240191142454, 0.0)
    wait_score = mad_normalize(wait_power)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + successor_release_contribution
    score += slack_headroom_mask * (0.938297964158664 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
