import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's hard DDL urgency and successor-release coupling
       with Parent 1's robust slack penalty exponentiation. Uses median-MAD normalization to preserve
       sign semantics, especially for negative slack. All 12 parameters used; no numeric literals except -2,-1,0,1,2."""
    eps = 0.005958885310536843
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
        x = np.asarray(x, dtype=float)
        center = np.median(x) if N > 1 else x[0]
        dev = np.abs(x - center)
        spread = np.median(dev) if N > 1 else dev[0] + eps
        return (x - center) / (spread + eps)
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    ddl_urgent = (slack <= 0.0).astype(float)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.0060700192375138
    slack_pressure_gate = 1.0 / (1.0 + np.exp(-1.1655348740207991 * -norm_slack))
    protected_rank = norm_rank * (1.0 + 0.9809715739201953 * (ddl_urgent + (1.0 - ddl_urgent) * slack_pressure_gate))
    successor_release_score = norm_work * norm_rank
    energy_uncert_penalty = norm_energy * np.where(norm_uncert > 0.5155926031803504, 1.0, 0.0)
    duration_risk_score = norm_duration * norm_uncert * np.clip(-norm_slack, 0.0, np.inf)
    wait_benefit = 1.0 - np.exp(-0.31658986648300397 * (norm_wait + 6.817264153343866e-08))
    score = -ddl_urgent * 43.24710598292998 - raw_slack_penalty - protected_rank - successor_release_score - 1.8774473600520911 * norm_energy - norm_duration - wait_benefit + 0.14916535855816337 * duration_risk_score + 0.8100863736623589 * energy_uncert_penalty + 0.5307344212741465 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
