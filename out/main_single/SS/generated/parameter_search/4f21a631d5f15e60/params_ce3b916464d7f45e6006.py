import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's logistic starvation relief and dual-pressure gating with Parent 1's validated critical-path leverage term;
       replaces redundant norm_duration term with explicit critical-path coupling; uses median-MAD normalization for outlier resilience.
    """
    eps = 0.002174382637430605
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.2423660330351067
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.0421770525655476 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.724816395838225 * rank_gate)
    critical_path_leverage = norm_rank * norm_work * rank_gate
    uncert_gate = np.where(norm_uncert > 0.17910821063193907, 1.0, 0.0)
    raw_duration = min_exec_time + min_comm_time
    norm_raw_duration = median_mad_normalize(raw_duration)
    duration_risk_score = norm_raw_duration * uncert_gate * slack_pressure
    wait_benefit = 1.0 - np.exp(-0.1804819552195633 * (ready_wait_time + 4.913567366653445e-06))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 0.543413442161452 * critical_path_leverage - 0.6051146440324967 * norm_energy - wait_benefit + 0.0017870889128101645 * duration_risk_score + 0.31523544131151066 * energy_uncert_penalty + 0.6296480760509298 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
