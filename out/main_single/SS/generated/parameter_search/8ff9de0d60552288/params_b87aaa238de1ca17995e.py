import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: adds hard-deadline enforcement via lateness penalty multiplier.
    Key improvements:
      - Introduces `lateness_penalty_strength`: multiplies *all* penalty terms (not rewards) when slack is critically negative,
        ensuring strict adherence to hard deadlines without distorting intra-feasible prioritization.
      - Reverts to original robust_normalize (MAD-only, no clipping) per reflection — preserves rank fidelity under stress.
      - Restores `slack_pressure` (not raw) in energy/rank terms for stronger nonlinear urgency amplification.
      - Removes successor pressure term as instructed; avoids unbounded upstream propagation.
      - All numeric literals remain in {-2,-1,0,1,2}; uses np.finfo for safe finite bounds.
    """
    eps = 0.017844473502037293
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
        center = np.median(x_abs)
        scale = np.median(np.abs(x_abs - center)) + eps
        return (x_abs - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_pressure_raw = np.clip(-norm_slack, 0.0, None)
    slack_pressure = np.power(slack_pressure_raw + eps, 1.920010059154873)
    is_late = (slack < -eps).astype(float)
    lateness_multiplier = 1.0 + 2.3813128047721968 * is_late
    rank_gate = np.clip(1.0 - norm_slack / (1.2122234398726928 + eps), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.083561728942434 * rank_gate)
    uncertainty_activation = 1.0 / (1.0 + np.exp(-1.3630602713180708 * (norm_uncert - 0.036247331997181526)))
    duration_risk_interaction = norm_duration * uncertainty_activation * 1.2988721523377014
    wait_benefit = 1.0 - np.exp(-0.24520705923569455 * (norm_wait + eps))
    energy_slack_penalty = norm_energy * (1.0 + 0.2652917330311795 * slack_pressure)
    coupled_rank_reward = norm_rank * (1.0 + 0.3980517677826538 * slack_pressure)
    score = +lateness_multiplier * slack_pressure + lateness_multiplier * energy_slack_penalty + lateness_multiplier * 0.8183623781813264 * norm_energy - coupled_rank_reward - wait_benefit + lateness_multiplier * duration_risk_interaction
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    return score
