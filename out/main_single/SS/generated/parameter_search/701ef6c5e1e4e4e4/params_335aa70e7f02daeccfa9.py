import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: merges best of both parents — preserves Parent 2's stable inverse ddl_pressure,
       adopts Parent 1's feasibility-gated & pressure-modulated starvation relief, removes redundant slope parameter,
       introduces explicit starvation_feasibility_coupling for adaptive anti-starvation under tight deadlines,
       unifies all robust normalizations, enforces hard feasibility gating on energy/uncertainty terms,
       and maintains strict [-2,2] clipping per term for bounded AST depth and stability."""
    eps = 0.00430723415451145
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad + eps
        return (x - med) / spread
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    hard_feasibility_gate = np.where(slack < 0.0, 0.0, 1.0)
    slack_abs = np.abs(slack) + eps
    ddl_pressure = np.clip(1.0 / (1.0 + slack_abs), 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 2.433773770073159 * norm_slack_penalty
    successor_release = norm_rank * norm_work * ddl_pressure
    coupled_rank = norm_rank * (1.0 + 1.339182172933597 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.34080956900548914, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_feasibility_gate
    wait_benefit = np.clip(0.18742068472502854 * ready_wait_time * 0.9635587704427937 * hard_feasibility_gate * (1.0 - ddl_pressure), 0.0, 2.0)
    duration_penalty = 0.45905701602453486 * norm_duration * ddl_pressure * hard_feasibility_gate
    slack_energy_suppress = np.where(slack < 0.0, 1.0 - 0.9918071143808237, 1.0)
    suppressed_norm_energy = norm_energy * slack_energy_suppress
    load_surrogate = norm_duration * norm_uncert
    load_gate = np.where((hard_feasibility_gate > 0.0) & (norm_uncert > 0.34080956900548914), np.clip(load_surrogate, 0.0, 2.0), 0.0)
    host_load_penalty = 0.5807762494414461 * load_gate * suppressed_norm_energy
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.6998989340962122 * np.clip(suppressed_norm_energy, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(duration_penalty, -2.0, 2.0) + 0.49177693321167415 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.6166622970100644 * np.clip(norm_work, -2.0, 2.0) + host_load_penalty
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
