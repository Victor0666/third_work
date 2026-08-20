import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces sigmoid gates with bounded linear interpolation for numerical stability;
       introduces successor-release interaction via norm_work * norm_rank under DDL pressure;
       removes redundant duration_robustness and wait_saturation_offset per evidence of inactivity;
       uses clipped linear slack scaling instead of power law to avoid overflow and preserve monotonicity;
       applies strict DDL-protection gating to all energy/uncertainty terms, validated by counterfactuals."""
    eps = 0.00011572746284352276
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_gate = np.clip(slack, 0.0, 1.0)
    ddl_gate = np.where(slack < 0.0, 0.0, ddl_gate)
    ddl_pressure = np.clip(-norm_slack, 0.0, 1.0)
    successor_release = norm_rank * norm_work * ddl_pressure
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_rank = norm_rank * (1.0 + 1.308912902585741 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.14532522212094617, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_benefit = np.clip(0.5198868995440543 * ready_wait_time, 0.0, 1.0)
    exec_penalty = norm_duration * ddl_pressure * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.37961706406461465 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + 0.44456968694635834 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.09584204022337862 * np.clip(norm_work, -2.0, 2.0) + np.clip(ddl_pressure * 3.7067496836393174, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
