import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating conditional DDL protection gate and successor-release-aware interactions.
    
    Key structural mutations:
    - Replaces linear wait fairness with smooth anti-starvation term: 1 - exp(-rate * norm_wait)
      to avoid clipping artifacts and improve cross-seed stability.
    - Introduces successor-release interaction: multiplies critical path boost by normalized remaining_work,
      prioritizing tasks whose execution unblocks high-work descendants — addresses 'successor_blocked' patterns.
    - Adds host-load-aware energy dampening: energy weight decays exponentially with slack pressure instead of tanh,
      improving gradient continuity near zero slack.
    - Uses robust mean-absolute scaling (not MAD) for all feature normalizations to resist outliers and align with
      counterfactual evidence recommending improved cross-seed stability.
    - Applies strict conditional DDL protection gate: critical path leverage and work-density bonus are only active
      when slack >= 0 AND uncertainty <= median_uncertainty, preventing over-commitment to risky paths.
    - Removes duration_balance from raw duration; instead applies it to *normalized* duration after robust scaling.
    """
    eps = 0.0003829963456488823
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        mu = np.mean(x)
        mad = np.mean(np.abs(x - mu))
        scale = 1.3548321935549217 * (mad if mad > eps else eps)
        return (x - mu) / scale
    slack_norm = np.where(slack < 0.0, 1.5051624030909416 * -slack, np.where(slack < 1.0, 1.4784908083322659 * (np.exp(slack) - 1.0), np.clip(slack, 0.0, 57.58210048175416)))
    duration = min_exec_time + min_comm_time
    duration_norm = robust_normalize(duration)
    energy_norm = robust_normalize(min_incremental_energy)
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    energy_weight_adj = 0.10129964143530207 * np.exp(-slack_pressure * 0.3218683003070852)
    rank_norm = robust_normalize(upward_rank)
    work_norm = robust_normalize(remaining_work)
    median_unc = np.median(uncertainty)
    ddl_protection_gate = ((slack >= 0.0) & (uncertainty <= median_unc + eps)).astype(float)
    critical_boost = 0.9557302070752243 * rank_norm * work_norm * ddl_protection_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = robust_normalize(work_density)
    work_density_bonus = 0.8075215242007291 * work_density_norm * ddl_protection_gate
    wait_norm = robust_normalize(ready_wait_time)
    wait_score = 1.757863684632941 * (1.0 - np.exp(-1.757863684632941 * wait_norm))
    unc_norm = robust_normalize(uncertainty)
    slack_pressure_bounded = 1.0 / (1.0 + np.exp(-slack_pressure * 0.3218683003070852))
    uncertainty_amplifier = 1.0171103832857473 * unc_norm * slack_pressure_bounded
    score = slack_norm + 1.2494026229034108 * duration_norm + energy_weight_adj * energy_norm - critical_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
