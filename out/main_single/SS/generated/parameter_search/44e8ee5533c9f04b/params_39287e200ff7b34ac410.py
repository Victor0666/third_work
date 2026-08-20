import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: restores `ddl_protection_gate_slope` and introduces `load_aware_suppression_strength`
       to dynamically suppress energy preference on high-duration tasks — directly targeting 'avoidable_marginal_energy_or_load_cost'.
       Removes unstable `slack_alignment_exponent` per reflection, reverting to robust linear slack penalty + gated critical-path amplification.
       Introduces duration-load gate: `np.clip((min_exec_time + min_comm_time) / (median_duration + eps), 0.0, 2.0)` to identify congested candidates.
       All energy-related contributions now modulated by both hard feasibility AND duration-load gate — preventing over-aggressive low-energy scheduling on already-loaded VMs.
       Maintains hard slack-driven energy suppression and unified robust normalization."""
    eps = 0.00016748261904045344
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
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad + eps
        return (x - med) / spread
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    duration = min_exec_time + min_comm_time
    if N == 1:
        median_duration = duration[0]
    else:
        median_duration = np.median(duration)
    duration_load_signal = np.clip(duration / (median_duration + eps), 0.0, 2.0)
    hard_feasibility_gate = np.where(slack < 0.0, 0.0, 1.0)
    ddl_pressure = np.clip(-norm_slack, 0.0, 1.0)
    raw_slack_penalty = np.maximum(-slack, 0.0)
    norm_slack_penalty = robust_normalize(raw_slack_penalty)
    slack_penalty = 1.70515500820565 * norm_slack_penalty
    successor_release = norm_rank * norm_work * ddl_pressure
    ddl_gate = 1.0 / (1.0 + np.exp(-4.260174616541142 * ddl_pressure))
    coupled_rank = norm_rank * (1.0 + 0.8701326796500954 * ddl_gate)
    uncert_gate = np.where(norm_uncert > 0.4335720640432025, 1.0, 0.0)
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * hard_feasibility_gate
    wait_benefit = np.clip(0.31141977266509485 * ready_wait_time, 0.0, 2.0)
    load_suppress = 1.0 - 0.40895803633389605 * (duration_load_signal - 1.0)
    load_suppress = np.clip(load_suppress, 0.0, 1.0)
    slack_energy_suppress = np.where(slack < 0.0, 1.0 - 0.5513925418628466, 1.0)
    suppressed_norm_energy = norm_energy * slack_energy_suppress * load_suppress * hard_feasibility_gate
    score = +np.clip(slack_penalty, -2.0, 2.0) - np.clip(successor_release, -2.0, 2.0) - np.clip(coupled_rank, -2.0, 2.0) - 0.9949168533682213 * np.clip(suppressed_norm_energy, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.6898589463535498 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.8391070307132446 * np.clip(norm_work, -2.0, 2.0)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.reshape(-1)
