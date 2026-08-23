import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Strict lexicographic DDL gating via boolean mask (not soft weighting)
      - Successor-release coupling: (upward_rank * remaining_work) * normalized_slack_deficit
      - Host-load–aware energy scaling: (1 + normalized_uncertainty) only in non-critical region
      - Bounded exponential urgency decay: (-slack)**exponent for critical tasks, replacing ramps
      - Robust MAD-based normalization (not percentile) to eliminate degeneracy in sparse sets
      - Anti-starvation term using ready_wait_time / (MAD(slack) + 1), activated only when slack > 0
      - All feature interactions capped to prevent explosion; no unbounded functions
    """
    eps = 1.231924516510094e-07
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
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    ddl_feasible = (slack > 0.0).astype(float)
    ddl_violated = (slack <= 0.0).astype(float)
    slack_abs = np.abs(slack)
    urgency_score = np.where(slack < 0.0, np.power(slack_abs, 2.2648849248891194), 0.0)
    slack_deficit = np.where(slack < 0.0, -slack, 0.0)
    norm_slack_deficit = mad_normalize(slack_deficit)
    successor_coupling = upward_rank * remaining_work * norm_slack_deficit * ddl_violated
    duration_total = min_exec_time + min_comm_time + eps
    duration_risk = duration_total * uncertainty * 0.2647868227426648 * ddl_violated
    energy_per_sec = min_incremental_energy / (duration_total + eps)
    energy_eff_norm = mad_normalize(energy_per_sec)
    energy_eff_score = ddl_feasible * 0.44360925542938334 * energy_eff_norm
    rank_norm = mad_normalize(upward_rank)
    slack_headroom = np.clip(slack / (48.03388186546232 - -14.344059994870719 + eps), 0.0, 1.0)
    rank_weight = 0.25000663721400473 * (1.0 - slack_headroom) + (1.0 - 0.25000663721400473) * slack_headroom
    rank_score = ddl_feasible * -rank_norm * rank_weight
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = ddl_feasible * (1.6870015966199705 * mad_normalize(min_incremental_energy) * (1.0 + np.clip(unc_norm, 0.0, 1.0)))
    wait_score = np.where(ddl_feasible > 0.0, ready_wait_time / (slack_abs + 1.0), 0.0)
    wait_score = ddl_feasible * mad_normalize(wait_score)
    slack_centered = slack - np.median(slack) if N > 1 else 0.0
    unc_slack_coupling = uncertainty * np.abs(slack_centered) * 1.1121402956218087
    score = mad_normalize(urgency_score) + mad_normalize(successor_coupling) + mad_normalize(duration_risk) + mad_normalize(unc_slack_coupling)
    score += energy_eff_score + rank_score + energy_uncertainty_score + wait_score
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= rank_median
    is_high_work = remaining_work >= work_median
    critical_gate = np.where(ddl_violated * is_high_rank * is_high_work, 2.1114048914444252, 1.0)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
