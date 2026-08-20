import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule: all declared parameters are used; no unused or missing references;
       uses robust median-MAD normalization; applies DDL-protection gate to energy/uncertainty;
       includes successor-release interaction (min_exec_time * upward_rank * (slack <= 0));
       enforces strict feasibility gating; clips urgency signal to [-2,2]; uses np.finfo for epsilon."""
    eps = np.finfo(float).tiny
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
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    urgency = np.clip(norm_slack, -2.0, 2.0)
    ddl_gate = 1.0 / (1.0 + np.exp(-3.5530022433158255 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_leverage = norm_rank * norm_work * ddl_breach
    successor_release = min_exec_time * upward_rank * ddl_breach
    norm_successor = median_mad_normalize(successor_release)
    energy_penalty = norm_energy * ddl_gate
    energy_uncert_penalty = energy_penalty * norm_uncert * 0.3935765524036671
    wait_benefit = 1.0 - np.exp(-0.29618696487401885 * ready_wait_time)
    work_bias = norm_work * ddl_gate
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    pressure_gate = np.tanh(3.193273068861949 * (slack_pressure - 1.0))
    rank_coupling = norm_rank * (1.0 + 2.357181365490325 * pressure_gate)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.3378254709948958
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    score = +urgency - critical_leverage - norm_successor - 0.13995328518205885 * energy_penalty - wait_benefit + 0.3648463932457418 * work_bias + energy_uncert_penalty - rank_coupling + np.clip(norm_slack_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
