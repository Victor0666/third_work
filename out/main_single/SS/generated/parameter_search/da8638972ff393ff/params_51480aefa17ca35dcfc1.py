import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: introduces conditional successor-release coupling gated by slack_pressure only;
       uses norm_duration for scale-consistent risk scoring; enforces hard DDL protection via +inf clamping;
       removes rank quantile gating to comply with 12-parameter limit while preserving structural intent.
    """
    eps = 0.0003715580794134149
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
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.6200269317895497
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-4.578571721924512 * (slack_pressure - 1.0)))
    successor_release_active = np.where(slack_pressure > 0.3693485132310034, 1.0, 0.0)
    critical_path_leverage = norm_rank * norm_work * successor_release_active
    boosted_rank = norm_rank * (1.0 + 1.9040051260388333 * rank_gate)
    uncert_gate = np.where(norm_uncert > 0.29692243219918557, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure
    wait_benefit = 1.0 - np.exp(-0.582546874186082 * (ready_wait_time + 3.157728099273058e-07))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - boosted_rank - 1.9040051260388333 * critical_path_leverage - 1.452841383490469 * norm_energy - wait_benefit + 1.2002530922345058 * duration_risk_score + 0.007266371149514126 * energy_uncert_penalty + 1.4858885185863748 * norm_work
    deadline_violated = slack < -eps
    score = np.where(deadline_violated, np.finfo(float).max, score)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
