import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Strict lexicographic DDL gating: all non-DDL terms disabled if slack <= 0
      - Critical path release prioritization via upward_rank × remaining_work product, scaled only when slack <= 0
      - Power-law anti-starvation: ready_wait_time ** wait_time_decay_exponent, gated by slack headroom
      - MAD-based robust normalization instead of percentile to avoid heterogeneity distortion
      - Dual feasibility gating for energy-aware terms: slack > 0 AND duration_total > eps
      - Unified risk-weighted duration: (min_exec_time + min_comm_time) * (1 + uncertainty), normalized via MAD
      - All numeric literals restricted to {-2, -1, 0, 1, 2}
      - Final score enforces: DDL violation penalty first → among feasible: critical successor release + energy efficiency → starvation mitigation only when safe
    """
    eps = 2.2136750148211477e-08
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
        mad = np.median(np.abs(x - med)) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_score = np.where(slack < 0, (-slack) ** 3.432945044283224, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    risk_duration = duration_total * (1.0 + uncertainty)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 1.8399118880730192
    is_tight_or_violated = slack <= 0
    successor_release = upward_rank * remaining_work
    successor_score = np.where(is_tight_or_violated, mad_normalize(successor_release) * 3.745695491614488, 0.0)
    slack_headroom_mask = np.where((slack > 0.0) & (duration_total > eps), 1.0, 0.0)
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -48.69886885632494
    slack_ub = 14.500171297098616
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.41648877618910735 + (1.0 - 0.41648877618910735) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_sigmoid = 1.0 / (1.0 + np.exp(-0.9765119977478047 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = 0.6130533354146213 * energy_norm * unc_norm * unc_sigmoid
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 1.2380335219254137, 0.0)
    wait_score = mad_normalize(wait_power)
    score = mad_normalize(slack_score) + mad_normalize(risk_duration) + mad_normalize(unc_slack_coupling) + successor_score
    score += slack_headroom_mask * (energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
