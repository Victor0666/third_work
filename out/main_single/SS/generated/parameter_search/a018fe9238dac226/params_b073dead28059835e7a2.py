import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: removes unused 'rank_slack_coupling'; retains robust starvation relief, adaptive gating, and novel energy-slack interaction."""
    eps = 0.0047693068255523
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
        center = np.median(x_abs)
        scale = np.median(np.abs(x_abs - center)) + eps
        return (x_abs - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_pressure = np.power(np.clip(-norm_slack, 0.0, None), 2.1559930381447305)
    rank_gate = np.clip(1.0 - norm_slack / (1.0867542372696049 + eps), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.20779349063731 * rank_gate)
    uncert_gate = ((norm_uncert > 0.4719685319071323) & (slack_pressure > 0)).astype(float)
    duration_risk_interaction = norm_duration * uncert_gate * 0.5619244343425069
    wait_benefit = 1.0 - np.exp(-0.35801425158724365 * (norm_wait + eps))
    energy_slack_penalty = norm_energy * (1.0 + 0.24419866998272166 * slack_pressure)
    score = +slack_pressure + energy_slack_penalty + 0.40905002477704444 * norm_energy - boosted_rank - wait_benefit + duration_risk_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
