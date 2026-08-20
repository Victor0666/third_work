import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces robust_normalize with mean-abs scaling for stability;
       introduces conditional DDL-protection gate; replaces exponential wait relief with tanh-based anti-starvation;
       adds successor-release interaction via remaining_work * slack coupling under deadline pressure."""
    eps = 0.09300707783396855
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def stable_normalize(x):
        x_abs = np.abs(x)
        scale = np.mean(x_abs) + eps
        return x / (scale + eps)
    norm_slack = stable_normalize(slack)
    norm_energy = stable_normalize(min_incremental_energy)
    norm_duration = stable_normalize(min_exec_time + min_comm_time)
    norm_rank = stable_normalize(upward_rank)
    norm_work = stable_normalize(remaining_work)
    norm_wait = stable_normalize(ready_wait_time)
    norm_uncert = stable_normalize(uncertainty)
    slack_pressure = np.tanh(-norm_slack * 2.198488229761734)
    ddl_protection_gate = np.clip(slack_pressure, 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.584328447232266 * ddl_protection_gate)
    successor_release_bonus = norm_work * ddl_protection_gate
    wait_benefit = np.tanh(0.5199618665019747 * (norm_wait + 5.900366148027074e-06))
    uncert_gate = (norm_uncert > 0.880934692521571) & (ddl_protection_gate > 0.7411412558744553)
    duration_risk_score = norm_duration * uncert_gate.astype(float) * ddl_protection_gate
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_protection_gate * uncert_gate.astype(float)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.2036847032759335
    norm_slack_penalty = stable_normalize(raw_slack_penalty)
    score = +norm_slack_penalty - boosted_rank - 0.6867569142540494 * norm_energy - norm_duration - wait_benefit + 0.1173610362516398 * duration_risk_score + 0.19401521270770267 * energy_uncert_penalty - 0.8073713165039077 * successor_release_bonus
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
