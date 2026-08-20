import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: removes unused 'slack_pressure_gate_steepness'; uses binary DDL protection gate,
       successor-release interaction, and simplified robust normalization. All declared parameters are used."""
    eps = 0.09552153320999246
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
        center = np.median(x) if N > 1 else x[0]
        spread = np.median(np.abs(x - center)) if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    ddl_urgent = np.where((slack <= 0.0) | (slack <= 0.09552153320999246) & (uncertainty >= 0.4252627255798446), 1.0, 0.0)
    protected_rank = norm_rank * (1.0 + 0.5723619700682162 * ddl_urgent)
    successor_release_score = norm_work * norm_rank
    energy_uncert_penalty = norm_energy * norm_uncert * np.where((norm_energy > 0.0) & (norm_uncert > 0.4252627255798446), 1.0, 0.0)
    wait_benefit = np.clip(0.04247080842516807 * (ready_wait_time + 1.0217498653197922e-09), 0.0, 1.0)
    duration_risk_score = norm_duration * np.where((norm_uncert > 0.4252627255798446) & (slack <= 0.0), 1.0, 0.0)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.3013476587348185
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    score = +norm_slack_penalty - protected_rank - successor_release_score - 1.0610857517191026 * norm_energy - norm_duration + 0.7822209495596004 * duration_risk_score + 0.7428151187281832 * energy_uncert_penalty - wait_benefit + 1.8911806740374573 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
