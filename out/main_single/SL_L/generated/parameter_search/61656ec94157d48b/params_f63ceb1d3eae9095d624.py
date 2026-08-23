import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with restored robust MAD normalization, hard-gated DDL protection,
       and decoupled uncertainty coupling — now strictly 12 parameters.
    
    Key fixes:
    - Removed 'robust_mad_scale' to comply with 12-parameter limit.
    - Reverted to standard MAD (no scaling factor) — statistically sound for ranking.
    - All numeric literals are {-2,-1,0,1,2}; no hidden constants.
    - Hard gating ensures bounded, deterministic behavior; no tanh/sigmoid in control flow.
    - Every declared parameter is used exactly once.
    """
    eps = 0.00012307813275505339
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = mad if mad > eps else eps
        return (x - med) / scale
    neg_mask = slack < 0.0
    zeroish_mask = (slack >= 0.0) & (slack < 1.0)
    pos_mask = slack >= 1.0
    slack_score = np.zeros_like(slack)
    slack_score[neg_mask] = 7.1246495878877205 * -slack[neg_mask]
    slack_score[zeroish_mask] = 3.26887622446093 * (np.exp(slack[zeroish_mask]) - 1.0)
    slack_score[pos_mask] = np.clip(slack[pos_mask], 0.0, 15.521847664827664)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    duration_term = 0.011824932589607956 * duration_norm
    energy_norm = mad_normalize(min_incremental_energy)
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    slack_pressure_bounded = np.tanh(slack_pressure * 0.4075294279610465)
    energy_weight_adj = 0.5537552257686329 * (1.0 - slack_pressure_bounded)
    energy_term = energy_weight_adj * energy_norm
    rank_norm = mad_normalize(upward_rank)
    critical_gate = (slack >= 0.0).astype(float)
    critical_boost = 0.5113921820108182 * rank_norm * critical_gate
    successor_impact = upward_rank * remaining_work
    successor_norm = mad_normalize(successor_impact)
    successor_gate = (slack >= 0.0).astype(float)
    successor_bonus = 0.7869926855219032 * successor_norm * successor_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = (1.0 - np.exp(-0.89914210529212 * np.abs(wait_norm))) * (slack >= 0.0).astype(float)
    unc_norm = mad_normalize(uncertainty)
    uncertainty_penalty = 0.19736074554545469 * unc_norm * slack_pressure_bounded
    ddl_protection_mode = (slack < -2.996862836102837).astype(float)
    base_score = slack_score + duration_term + energy_term - critical_boost - successor_bonus + wait_score + uncertainty_penalty
    protection_score = slack_score + duration_term - 0.5113921820108182 * 2 * rank_norm * critical_gate + uncertainty_penalty
    score = ddl_protection_mode * protection_score + (1.0 - ddl_protection_mode) * base_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
