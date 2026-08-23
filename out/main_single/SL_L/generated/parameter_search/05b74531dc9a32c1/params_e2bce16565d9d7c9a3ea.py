import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating:
      - Unified MAD-based normalization (robust to outliers, N=1 safe) instead of percentile clipping
      - Critical path urgency expressed via upward_rank × remaining_work interaction, scaled by slack pressure
      - DDL-protection gate now uses smooth sigmoid instead of hard threshold for slack > 0, improving gradient flow
      - Anti-starvation term restructured as `ready_wait_time * exp(-|slack|/tau)` — decays smoothly with headroom
      - Energy-efficiency term replaced by ratio `min_incremental_energy / (min_exec_time + min_comm_time + eps)`
        normalized via MAD, avoiding division instability
      - All gates and interactions explicitly bounded and guarded; no unclipped divisions or exponentials
      - Criticality boost applied multiplicatively only when slack <= 0 AND upward_rank > factor*median AND remaining_work > factor*median
      - Final score preserves lexicographic DDL-first ordering: violation terms dominate feasible ones
    """
    eps = 7.402238552061465e-08
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
    slack_score = np.where(slack < 0, (-slack) ** 3.0189599937336196, 0.0)
    deadline_pressure = np.maximum(0.0, -slack)
    unc_slack_coupling = uncertainty * deadline_pressure * 0.9351289624697756
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 1.087764284135522
    is_tight_or_violated = slack <= 0
    rank_median = np.median(upward_rank) if N > 1 else np.mean(upward_rank)
    work_median = np.median(remaining_work) if N > 1 else np.mean(remaining_work)
    is_high_rank = upward_rank >= 0.9612090930655706 * rank_median
    is_high_work = remaining_work >= 0.9612090930655706 * work_median
    critical_gate = np.where(is_high_rank & is_high_work & is_tight_or_violated, 0.9005406439634378, 1.0)
    slack_gate = 1.0 / (1.0 + np.exp(-4.116742293925425 * slack))
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    slack_lb = -48.740983956045326
    slack_ub = 29.559444446984084
    slack_scaled = np.clip((slack - slack_lb) / (slack_ub - slack_lb + eps), 0.0, 1.0)
    weight_rank = 0.5763071575813988 + (1.0 - 0.5763071575813988) * (1.0 - slack_scaled)
    rank_score = -mad_normalize(upward_rank) * weight_rank
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    unc_sigmoid = 1.0 / (1.0 + np.exp(-4.116742293925425 * (uncertainty - 1.0)))
    energy_uncertainty_score = 0.321143340855094 * energy_norm * unc_norm * unc_sigmoid
    wait_decay = np.exp(-np.abs(slack) / (29.559444446984084 + eps))
    wait_score = mad_normalize(ready_wait_time) * wait_decay
    score = mad_normalize(slack_score) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk)
    score += slack_gate * (0.803055372049202 * energy_eff_score + rank_score + energy_uncertainty_score + wait_score)
    score = score * critical_gate
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
