import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Hybrid priority rule combining Parent 2's exponential urgency & gated criticality with Parent 1's
    work-density bonus and starvation-uncertainty modulation. Introduces novel starvation modulation:
    wait boost scaled inversely by uncertainty to avoid over-committing to old tasks in volatile environments.
    All normalizations use mean-abs + epsilon for stability; no unbounded ops or loops.
    """
    eps = 2.7245972982050994e-09
    finfo = np.finfo(np.float64)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def robust_norm(x):
        denom = np.mean(np.abs(x)) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    duration_raw = min_exec_time + min_comm_time
    norm_duration = robust_norm(duration_raw)
    norm_rank = robust_norm(upward_rank)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    work_density = np.divide(remaining_work, duration_raw + eps)
    norm_work_density = robust_norm(work_density)
    slack_sign_mask = (slack < 0).astype(float)
    urgency_base = -norm_slack * slack_sign_mask
    urgency_clipped = np.clip(urgency_base, 0.0, 3.2361000252966896)
    urgency_penalty = np.exp(2.9359339945429572 * urgency_clipped)
    rank_median = np.median(norm_rank)
    critical_mask = (norm_rank >= rank_median).astype(float)
    critical_boost = 3.0122649120045506 * norm_rank * critical_mask
    slack_pressure = np.maximum(0.0, -norm_slack)
    uncertainty_coupled_pressure = 1.294418731791309 * slack_pressure * norm_uncert
    wait_boost_base = 1.0 - np.exp(-0.08913450138737335 * norm_wait)
    starvation_mod = np.clip(1.0 - 0.857697395242498 * norm_uncert, 0.0, 1.0)
    wait_boost = wait_boost_base * starvation_mod
    rank_energy_penalty = 1.0113847033789618 * norm_energy * norm_rank
    work_density_bonus = 0.025151562166491434 * norm_work_density
    score = urgency_penalty + uncertainty_coupled_pressure + rank_energy_penalty + 1.3339332523924352 * norm_energy + 1.7186481497020092 * norm_duration - critical_boost - wait_boost - work_density_bonus
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=-finfo.max)
    return score
