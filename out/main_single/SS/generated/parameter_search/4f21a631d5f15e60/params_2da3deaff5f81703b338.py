import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's logistic starvation relief and dual-pressure gating with Parent 1's validated critical-path leverage term;
       replaces redundant norm_duration term with explicit critical-path coupling; uses median-MAD normalization for outlier resilience.
    """
    eps = 0.008213291712924732
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
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.8427936852194187
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-4.917066337701783 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.5611418724030726 * rank_gate)
    critical_path_leverage = norm_rank * norm_work * rank_gate
    uncert_gate = np.where(norm_uncert > 0.5209001126673893, 1.0, 0.0)
    raw_duration = min_exec_time + min_comm_time
    norm_raw_duration = median_mad_normalize(raw_duration)
    duration_risk_score = norm_raw_duration * uncert_gate * slack_pressure
    wait_benefit = 1.0 - np.exp(-0.0038191624718717765 * (ready_wait_time + 3.473139148569958e-05))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 0.24256073190313066 * critical_path_leverage - 1.348978366639832 * norm_energy - wait_benefit + 0.35938805343393376 * duration_risk_score + 0.1402585390399712 * energy_uncert_penalty + 0.9477947465451761 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
