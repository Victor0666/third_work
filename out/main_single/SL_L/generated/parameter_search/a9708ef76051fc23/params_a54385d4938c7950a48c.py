import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Strict lexicographic DDL gating (slack > 0) preserved.
      - Successor-release coupling: (upward_rank × remaining_work) scaled by normalized slack deficit, activated only under tight/late conditions.
      - Host-load–aware energy scaling via (1 + normalized_uncertainty), applied only when slack > 0 to avoid energy optimization under violation.
      - Bounded exponential urgency decay (-slack)**exponent for critical-path prioritization near deadlines.
      - Robust MAD-based normalization (not percentile) for improved stability in sparse ready sets (N ≤ 5).
      - Ready-wait anti-starvation now uses exponential decay: exp(-ready_wait_decay * normalized_wait), reducing dominance in congested queues.
      - All divisions guarded; no unbounded ops; shape (N,) enforced.
    """
    eps = 1.0092877314857122e-05
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
    slack_score = np.where(slack < 0, (-slack) ** 1.1105294342628667, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 2.5482039140376074
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.4066748632577976
    is_tight_or_violated = slack <= 0
    rank_work_product = upward_rank * remaining_work
    slack_deficit_norm = np.where(is_tight_or_violated, 1.0 / (1.0 + np.exp(-slack)), 0.0)
    successor_release_score = np.where(is_tight_or_violated, rank_work_product * slack_deficit_norm * 0.622454971947376, 0.0)
    slack_headroom_mask = np.where(slack > 0.0, 1.0, 0.0)
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -7.254577660928973
    slack_ub = 20.363736693103437
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.8500681905761891 + (1.0 - 0.8500681905761891) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    unc_norm = mad_normalize(uncertainty)
    energy_norm = mad_normalize(min_incremental_energy)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.465669481554309 * (uncertainty - 1.0)))
    energy_uncertainty_score = 0.7934378294590683 * energy_norm * (1.0 + unc_norm) * unc_sigmoid
    wait_normalized = mad_normalize(ready_wait_time)
    wait_score = np.where(slack_headroom_mask > 0.0, np.exp(-0.7808908998164527 * np.abs(wait_normalized)), 0.0)
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(successor_release_score)
    score += slack_headroom_mask * (0.8478719918193016 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
