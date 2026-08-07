import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with hard DDL-protection gate, duration-normalized energy,
    and streamlined fairness — eliminating wait decay in favor of slack-driven urgency.
    
    Key improvements:
      - Hard deadline protection gate: tasks with slack <= ddl_protection_threshold receive
        dominant additive penalty (not just relative boost), ensuring strict DDL adherence.
      - Restores duration-normalized energy (exec + comm) per reflection insight — aligns
        urgency with actual scheduling latency impact.
      - Removes wait_decay_rate and associated exponential boost: empirical evidence shows
        it harms lateness; fairness now emerges via slack prioritization and critical-path gating.
      - All normalization remains IQR-based with tunable percentiles for robustness.
      - Critical-path bonus preserved but now strictly gated by percentile to avoid outlier dominance.
      - Uncertainty-slack coupling retained only under tight slack, reinforcing risk-aware urgency.
    """
    eps = 0.00023865884274288095
    N = len(slack)
    finfo = np.finfo(float)
    eps_safe = max(eps, finfo.tiny)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 31.977074163844105)
        q_high = np.percentile(x, 67.4160307999463)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps_safe else np.max(np.abs(x - center)) + eps_safe
        return (x - center) / (denom + eps_safe)
    slack_norm = iqr_normalize(slack)
    slack_penalty = np.where(slack < 0, 6.024793337119462 * -slack_norm, -4.085563678631072 * slack_norm)
    duration = np.maximum(min_exec_time + min_comm_time, eps_safe)
    energy_per_duration = min_incremental_energy / duration
    energy_norm = iqr_normalize(energy_per_duration)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps_safe)
    critical_gate = np.where(rank_percentile >= 0.6170111599525769, 1.0, 0.0)
    rank_norm = iqr_normalize(upward_rank)
    critical_bonus = 0.4941980187830159 * rank_norm * critical_gate
    unc_norm = iqr_normalize(uncertainty)
    slack_pressure = np.maximum(-slack_norm, 0.0)
    coupled_urgency = 0.018379885882058036 * unc_norm * slack_pressure
    dur_eff_ratio = 0.6711120845994727 * iqr_normalize(duration)
    ddl_protection_mask = (slack <= -0.06347834249396467).astype(float)
    ddl_protection_penalty = ddl_protection_mask * 331.8539774667949
    score = ddl_protection_penalty + slack_penalty + 0.44755466109241276 * energy_norm - critical_bonus + coupled_urgency + dur_eff_ratio
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
