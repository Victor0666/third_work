import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: replaces steepened softplus starvation relief with *bounded linear ramp* for numerical safety, 
       monotonic discrimination, and zero-overhead at zero wait; reinstates clipped linear hinge for rank-slack coupling with tunable offset 
       to activate urgency earlier under mild deadline pressure; removes unused parameters (slack_pressure_gate_steepness, wait_decay) per validation;
       retains joint feasibility gate and median-MAD normalization; enforces strict [-2,2] clipping on all terms."""
    eps = 0.0002742005317960177
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
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    joint_gate = 1.0 / (1.0 + np.exp(-4.429768772051932 * slack)) * 1.0 / (1.0 + np.exp(4.429768772051932 * (norm_uncert - 0.44578801623736813)))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 3.1946563781123785
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    slack_hinge = np.clip(-norm_slack - 0.9852522875752747, 0.0, 1.0)
    boosted_rank = norm_rank * (1.0 + 1.6596884239200755 * slack_hinge)
    starvation_ramp = np.clip(1.2622332714285986 * (ready_wait_time + eps), 0.0, 2.0)
    wait_benefit = starvation_ramp
    energy_uncert_penalty = norm_energy * norm_uncert * joint_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - 1.3498243076821814 * np.clip(norm_energy * joint_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.2467985133373569 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.06825189624785466 * np.clip(norm_work * joint_gate, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
