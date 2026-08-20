import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: replaces linear starve-relief with bounded waiting bonus;
       adds successor-release-aware term via remaining_work * upward_rank interaction;
       replaces global robust_normalize with median-MAD per-feature; uses conditional DDL protection gate.
    """
    eps = 0.016611172642283553
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
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.310551026292675
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-1.217922413861043 * (slack_pressure - 1.0)))
    boosted_rank = norm_rank * (1.0 + 2.2007217306146756 * rank_gate)
    uncert_gate = (norm_uncert > 0.9091130799649432).astype(float)
    duration_risk_score = norm_duration * uncert_gate * ddl_breach
    wait_bonus = np.clip(0.18398347439516585 * (ready_wait_time + 1.4014546148326253e-08), 0.0, 1.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    score = +norm_slack_penalty - critical_path_leverage - boosted_rank - 1.5340027073066267 * norm_energy - wait_bonus + 0.0012162769263098568 * duration_risk_score + 0.8132878872803659 * energy_uncert_penalty + 0.41799034104277 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
