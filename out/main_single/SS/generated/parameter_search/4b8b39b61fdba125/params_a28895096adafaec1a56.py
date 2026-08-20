import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: introduces a hard-deadline protection gate that *dominates* the score when slack < 0,
       using a steep sigmoid to trigger additive priority bias — ensuring strict DDL satisfaction over energy minimization.
       Replaces rank-slack coupling with unified DDL gate; uses raw slack for fidelity *and* bounded normalized pressure
       for robust gating. All operations are finite, deterministic, and shape-compliant."""
    eps = 0.03515960100464881
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
        x_abs = np.abs(x)
        center = np.median(x_abs) if N > 1 else x_abs[0]
        spread = np.median(np.abs(x_abs - center)) if N > 1 else np.abs(x_abs[0] - center) + eps
        return (x_abs - center) / (spread + eps)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.1563058245584363
    slack_pressure_raw = np.clip(-slack, 0.0, None)
    slack_pressure_norm = np.zeros_like(slack_pressure_raw)
    if N > 1:
        denom = np.max(slack_pressure_raw) - np.min(slack_pressure_raw) + eps
        slack_pressure_norm = np.clip((slack_pressure_raw - np.min(slack_pressure_raw)) / denom, 0.0, 2.0)
    else:
        slack_pressure_norm[0] = 0.0 if slack[0] >= 0 else 2.0
    ddl_gate = 1.0 / (1.0 + np.exp(-5.159166229994343 * (slack_pressure_norm - 1.0)))
    boosted_rank = norm_rank * (1.0 + 2.3861403709828295 * ddl_gate)
    uncert_gate = np.where(norm_uncert > 0.0014510777035129897, 1.0, 0.0)
    duration_risk_score = norm_duration * uncert_gate * slack_pressure_norm
    wait_benefit = 1.0 - np.exp(-0.4862993931320022 * (norm_wait + 3.206044788843911e-07))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate
    base_score = +raw_slack_penalty - boosted_rank - 0.9516068792731931 * norm_energy - norm_duration - wait_benefit + 0.02485236664058933 * duration_risk_score + 0.3813992043877641 * energy_uncert_penalty + 0.6896955792961765 * norm_work
    ddl_override = 21.070748894873596 * ddl_gate
    score = base_score + ddl_override
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
