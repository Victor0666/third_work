import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Hybrid priority rule combining Parent 2's exponential urgency & gated criticality with Parent 1's
    work-density bonus and starvation-uncertainty modulation. Introduces novel starvation modulation:
    wait boost scaled inversely by uncertainty to avoid over-committing to old tasks in volatile environments.
    All normalizations use mean-abs + epsilon for stability; no unbounded ops or loops.
    """
    eps = 1.065679683037072e-08
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
    urgency_clipped = np.clip(urgency_base, 0.0, 5.587137871779711)
    urgency_penalty = np.exp(1.8725977896774608 * urgency_clipped)
    rank_median = np.median(norm_rank)
    critical_mask = (norm_rank >= rank_median).astype(float)
    critical_boost = 2.643181486039544 * norm_rank * critical_mask
    slack_pressure = np.maximum(0.0, -norm_slack)
    uncertainty_coupled_pressure = 1.3128526463222645 * slack_pressure * norm_uncert
    wait_boost_base = 1.0 - np.exp(-0.3479378943539286 * norm_wait)
    starvation_mod = np.clip(1.0 - 0.30004888046368877 * norm_uncert, 0.0, 1.0)
    wait_boost = wait_boost_base * starvation_mod
    rank_energy_penalty = 0.79431815806623 * norm_energy * norm_rank
    work_density_bonus = 0.2791445518681457 * norm_work_density
    score = urgency_penalty + uncertainty_coupled_pressure + rank_energy_penalty + 2.3156606731110467 * norm_energy + 1.284785776645525 * norm_duration - critical_boost - wait_boost - work_density_bonus
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=-finfo.max)
    return score
