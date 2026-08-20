import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule: merges Parent 2's robust median-MAD normalization and feasibility-gated critical-path booster with Parent 1's explicit exec/comm separation;
       introduces *comm_priority_under_low_uncertainty* to promote early I/O placement when bandwidth is predictable — avoids over-deferral of comm-bound tasks on edge;
       replaces unstable `norm_duration` with independent, DDL-gated `norm_exec` and `norm_comm`, each with adaptive sensitivity;
       retains all proven gates (ddl_gate, slack_pressure_gate, uncert_gate) but reassigns them to decoupled signals for finer control;
       enforces monotonicity via bounded clipping [-2,2] and uses np.nan_to_num for guaranteed finiteness."""
    eps = 0.042702856631325066
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
    norm_exec = median_mad_normalize(min_exec_time)
    norm_comm = median_mad_normalize(min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_gate = 1.0 / (1.0 + np.exp(-5.67985621030666 * slack))
    raw_slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    slack_pressure_gate = 1.0 / (1.0 + np.exp(-4.08281110745341 * (raw_slack_pressure - 1.0)))
    critical_path_booster = norm_rank * norm_work * ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.9917803983328746
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    rank_amplifier = 1.0 + 1.863856800295904 * slack_pressure_gate
    rank_amplifier = np.clip(rank_amplifier, 1.0, 2.0)
    boosted_rank = norm_rank * rank_amplifier
    uncert_gate = 1.0 / (1.0 + np.exp(-5.67985621030666 * (norm_uncert - 0.003599062598647247)))
    wait_benefit = 1.0 - np.exp(-0.09103162089181885 * (ready_wait_time + 1.2740848570398206e-07))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.9917803983328746 * raw_slack_pressure) * ddl_gate
    exec_penalty = norm_exec * slack_pressure_gate * ddl_gate
    comm_priority_factor = np.where(norm_uncert < 0.003599062598647247, 0.8335877479894749, 0.0)
    comm_penalty = norm_comm * comm_priority_factor * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_booster, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 0.6032333296453712 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.7764378968373601 * np.clip(energy_uncert_penalty, -2.0, 2.0) + np.clip(energy_slack_penalty, -2.0, 2.0) + 0.7440485577280525 * np.clip(norm_work, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + np.clip(comm_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
