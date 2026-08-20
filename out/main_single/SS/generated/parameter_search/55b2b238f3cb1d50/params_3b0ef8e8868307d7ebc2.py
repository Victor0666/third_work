import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces global starvation relief with *conditional* anti-starvation gate — activated only when a task has waited longer than threshold_ratio × median_wait, preventing overcorrection;
       retains decoupled exec/comm signals and median-MAD normalization;
       introduces wait_benefit_steepness for sharper, more controllable saturation;
       removes redundant wait_decay and comm_priority_under_low_uncertainty (merged into existing logic via norm_comm + ddl_gate);
       all intermediate terms clipped to [-2,2] for bounded AST depth and stability."""
    eps = 0.0030784055710262694
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
    ddl_gate = 1.0 / (1.0 + np.exp(-9.03321913728303 * slack))
    raw_slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    slack_pressure_gate = 1.0 / (1.0 + np.exp(-1.6769755264382469 * (raw_slack_pressure - 1.0)))
    critical_path_booster = norm_rank * norm_work * ddl_gate
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.941388803265038
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    rank_amplifier = 1.0 + 2.2939896894580447 * slack_pressure_gate
    rank_amplifier = np.clip(rank_amplifier, 1.0, 2.0)
    boosted_rank = norm_rank * rank_amplifier
    uncert_gate = 1.0 / (1.0 + np.exp(-9.03321913728303 * (norm_uncert - 0.6383136121535714)))
    wait_median = np.median(ready_wait_time) if N > 1 else ready_wait_time[0]
    wait_threshold = 0.6173736245966306 * (wait_median + 0.00020690928139996224)
    wait_activation = 1.0 / (1.0 + np.exp(-3.8693364427111554 * (ready_wait_time - wait_threshold)))
    wait_benefit = wait_activation * (1.0 - np.exp(-1.0 * (ready_wait_time - wait_threshold + 0.00020690928139996224)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.941388803265038 * raw_slack_pressure) * ddl_gate
    exec_penalty = norm_exec * slack_pressure_gate * ddl_gate
    comm_penalty = norm_comm * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_booster, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.1240755734265733 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.9484592742579132 * np.clip(energy_uncert_penalty, -2.0, 2.0) + np.clip(energy_slack_penalty, -2.0, 2.0) + 0.24489449878800995 * np.clip(norm_work, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + np.clip(comm_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
