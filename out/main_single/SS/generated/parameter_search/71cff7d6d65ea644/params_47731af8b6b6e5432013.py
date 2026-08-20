import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces fragile piecewise starvation with bounded arctan relief,
       removes inactive parameters, simplifies rank-slack coupling, and enforces strict DDL-first logic.
       Critical path amplification now exclusively gated by slack urgency — no double-counting.
       All normalization uses median-MAD; all operations guarded against NaN/inf/zero."""
    eps = 0.012790261218111087
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        center = np.median(x) if N > 1 else x[0]
        spread = np.median(np.abs(x - center)) if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_margin_ratio = np.abs(slack) / (min_exec_time + min_comm_time + eps)
    ddl_urgent = np.where((slack <= 0.0) | (slack_margin_ratio < 0.04045196471951486), 1.0, 0.0)
    successor_release_impact = norm_work * norm_rank
    energy_modulation_gate = np.where((norm_uncert > 0.8225885449733568) & (ddl_urgent == 1.0), 1.0, 0.0)
    slack_pressure_norm = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-3.136123231520789 * (slack_pressure_norm - 1.0)))
    wait_benefit = 2.0 / np.pi * np.arctan(1.517009704576135 * norm_wait)
    duration_risk_score = norm_duration * norm_uncert * ddl_urgent
    energy_uncert_penalty = norm_energy * norm_uncert * energy_modulation_gate
    score = +norm_slack + norm_slack ** 1.5156739128815486 * ddl_urgent - norm_rank * (1.0 + 1.5958706731517578 * rank_gate * ddl_urgent) - successor_release_impact - 1.8441695457728153 * norm_energy * energy_modulation_gate - wait_benefit + 1.0245356791302234 * duration_risk_score + 0.6625631710658771 * energy_uncert_penalty + 0.8167302344976035 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
