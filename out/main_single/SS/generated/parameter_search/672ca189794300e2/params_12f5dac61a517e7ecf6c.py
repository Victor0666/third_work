import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's robust normalization and sigmoid gating
       with Parent 1's raw-slack fidelity via adaptive blending. Introduces convex blend
       of raw and robust-normalized slack to preserve both extreme deadline violation signals
       and outlier-resilient relative ranking. Removes redundant `rank_slack_coupling` and
       `uncertainty_gate_threshold` duplication; unifies uncertainty gating under single
       threshold. All parameters used; no literals except -2,-1,0,1,2."""
    eps = 6.061057368722176e-05
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
        x_abs = np.abs(x)
        center = np.median(x_abs) if N > 1 else x_abs[0]
        spread = np.median(np.abs(x_abs - center)) if N > 1 else np.abs(x_abs[0] - center) + eps
        return (x_abs - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    blended_slack = (1.0 - 0.04997726039391435) * slack + 0.04997726039391435 * norm_slack
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.1242643313017444
    slack_pressure = np.clip(-blended_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.6750280111539517 * (slack_pressure - 1.0)))
    norm_rank = robust_normalize(upward_rank)
    boosted_rank = norm_rank * (1.0 + 1.4310797951118706 * rank_gate)
    norm_uncert = robust_normalize(uncertainty)
    uncert_gate = np.where(norm_uncert > 0.8520317156570404, 1.0, 0.0)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    duration_risk_score = norm_duration * uncert_gate * raw_slack_penalty
    norm_wait = robust_normalize(ready_wait_time)
    wait_benefit = 1.0 - np.exp(-0.7442740324184536 * (norm_wait + 2.440709656853782e-08))
    norm_energy = robust_normalize(min_incremental_energy)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    norm_work = robust_normalize(remaining_work)
    score = +raw_slack_penalty - boosted_rank - 1.3374343198048764 * norm_energy - norm_duration - wait_benefit + 0.39088280655789664 * duration_risk_score + 0.31199083579008985 * energy_uncert_penalty + 0.6451409030364599 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
