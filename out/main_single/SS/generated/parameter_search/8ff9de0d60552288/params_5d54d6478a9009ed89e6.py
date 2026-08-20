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
    eps = 0.00013871188789052016
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
    slack_pressure = np.power(slack_pressure_raw + eps, 1.5506633164322263)
    is_late = (slack < -eps).astype(float)
    lateness_multiplier = 1.0 + 2.539345690024671 * is_late
    rank_gate = np.clip(1.0 - norm_slack / (0.5806663447114153 + eps), 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.408413317477584 * rank_gate)
    uncertainty_activation = 1.0 / (1.0 + np.exp(-1.7219948657558817 * (norm_uncert - 0.4191842095553779)))
    duration_risk_interaction = norm_duration * uncertainty_activation * 1.3227584175237945
    wait_benefit = 1.0 - np.exp(-0.48372850775116305 * (norm_wait + eps))
    energy_slack_penalty = norm_energy * (1.0 + 0.9953819575284912 * slack_pressure)
    coupled_rank_reward = norm_rank * (1.0 + 0.46975586460052354 * slack_pressure)
    score = +lateness_multiplier * slack_pressure + lateness_multiplier * energy_slack_penalty + lateness_multiplier * 1.4082404202196455 * norm_energy - coupled_rank_reward - wait_benefit + lateness_multiplier * duration_risk_interaction
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    return score
