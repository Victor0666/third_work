import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: removes unused parameters; uses robust range normalization;
       applies unified congestion signal (ready_wait_time + uncertainty) gated by ddl_gate;
       replaces power-law slack penalty with clipped linear pressure for stability;
       enforces critical-path leverage only under slack <= 0 via ddl_breach;
       suppresses energy/uncertainty terms exclusively under ddl_gate (slack >= 0);
       retains sign-preserving structure, finite output, and strict (N,) shape."""
    eps = 0.0036988404159094507
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_range_normalize(x):
        x = np.copy(x)
        if N == 1:
            return np.zeros_like(x)
        x_min = np.min(x)
        x_max = np.max(x)
        denom = x_max - x_min + eps
        return (x - x_min) / denom
    norm_slack = robust_range_normalize(slack)
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_duration = robust_range_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_range_normalize(upward_rank)
    norm_work = robust_range_normalize(remaining_work)
    norm_wait = robust_range_normalize(ready_wait_time)
    norm_uncert = robust_range_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-4.357644596569325 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    slack_pressure = np.clip(-norm_slack, 0.0, 1.0)
    congestion = (ready_wait_time + uncertainty) * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.755690514364135 * ready_wait_time)
    energy_uncert_penalty = norm_energy * norm_uncert * ddl_gate
    score = +1.407133570514632 * slack_pressure - 1.0659209255338247 * critical_path_leverage - 1.2550614103373228 * norm_work * ddl_gate - 1.32582343807129 * norm_energy * ddl_gate - wait_benefit + 0.09302595461386806 * energy_uncert_penalty + congestion + norm_rank * (1.0 - ddl_gate)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
