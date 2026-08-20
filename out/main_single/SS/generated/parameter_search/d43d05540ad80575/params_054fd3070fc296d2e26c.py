import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: merges Parent 2's numerical stability with Parent 1's robust successor-release modeling;
       introduces exponentiated successor-release interaction (validated by DDL diagnostics) for stronger critical-path unblocking;
       replaces linear DDL gate with centered linear ramp for sharper, tunable feasibility enforcement;
       retains clipped linear slack penalty and median-MAD normalization for monotonicity and outlier resilience."""
    eps = 4.85879664613842e-05
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
    gate_lower = 0.5336411391122725 - 0.17222600485496198
    gate_upper = 0.5336411391122725 + 0.17222600485496198
    ddl_pressure = np.clip((norm_slack - gate_lower) / (gate_upper - gate_lower + eps), 0.0, 1.0)
    ddl_pressure = np.where(norm_slack <= gate_lower, 0.0, ddl_pressure)
    ddl_pressure = np.where(norm_slack >= gate_upper, 1.0, ddl_pressure)
    successor_release = (norm_rank * norm_work) ** 0.8994247083971698 * ddl_pressure
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_rank = norm_rank * (1.0 + 0.5033333758775659 * ddl_pressure)
    uncert_gate = np.where(norm_uncert > 0.3193109796226499, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_pressure
    wait_benefit = np.clip(0.0005362246944969433 * ready_wait_time, 0.0, 1.0)
    exec_penalty = norm_duration * ddl_pressure
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.10092084921939554 * np.clip(norm_energy * ddl_pressure, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + 0.42160031949429466 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.7920692117154462 * np.clip(norm_work, -2.0, 2.0) + np.clip(ddl_pressure * 6.373660231208943, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
