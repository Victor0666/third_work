import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule:
       - Replaces unstable median-MAD with robust min-max normalization using explicit zero-slack guard.
       - Replaces power-law successor-release with bounded linear interaction: min_exec_time * norm_rank * min(slack_urgency_cap, max(0, -slack + eps)).
       - Removes inactive ddl_pressure_gate_threshold and redundant epsilon-guarded divisions.
       - Introduces slack_urgency_cap to prevent urgency overamplification near deadline breach.
       - All features normalized via [x - min(x)] / (max(x) - min(x) + eps) with clipping to [-2,2] for stability.
       - Preserves starvation relief, host-load surrogate, and energy-uncertainty coupling under unified feasibility gating.
    """
    eps = 3.560572414197043e-06
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_minmax_normalize(x):
        x = np.copy(x)
        if N == 1:
            return np.zeros_like(x)
        x_min = np.min(x)
        x_max = np.max(x)
        spread = x_max - x_min
        denom = spread if spread > eps else eps
        return (x - x_min) / denom
    norm_slack = robust_minmax_normalize(slack)
    norm_energy = robust_minmax_normalize(min_incremental_energy)
    norm_duration = robust_minmax_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_minmax_normalize(upward_rank)
    norm_work = robust_minmax_normalize(remaining_work)
    norm_wait = robust_minmax_normalize(ready_wait_time)
    norm_uncert = robust_minmax_normalize(uncertainty)
    slack_urgency_magnitude = np.maximum(0.0, -slack + eps)
    bounded_slack_urgency = np.clip(slack_urgency_magnitude, 0.0, 0.19061694111868122)
    norm_slack_urgency = robust_minmax_normalize(bounded_slack_urgency)
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_urgency = norm_rank * norm_work * ddl_breach
    successor_release = min_exec_time * norm_rank * ddl_breach * norm_slack_urgency
    ddl_pressure = norm_slack_urgency
    load_gate = (norm_uncert >= 0.029565808239932913).astype(float) * ddl_pressure
    host_load_surrogate = norm_duration * norm_uncert * load_gate
    energy_uncert_gate = (norm_uncert >= 0.029565808239932913).astype(float) * ddl_pressure
    energy_uncert_penalty = norm_energy * norm_uncert * energy_uncert_gate
    wait_benefit = np.exp(-0.014597007859794406 * ready_wait_time) * (1.0 - ddl_pressure)
    score = +np.clip(norm_slack_urgency, -2.0, 2.0) - np.clip(critical_path_urgency, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - 0.11392659762465768 * np.clip(norm_energy * ddl_pressure, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(host_load_surrogate, -2.0, 2.0) + 0.819212086937045 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.1315131453865916 * np.clip(norm_work, -2.0, 2.0) + 1.1571758489024515 * np.clip(norm_rank * ddl_pressure, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
