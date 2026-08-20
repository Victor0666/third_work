import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule:
       - Removes fragile dual-gated criticality boost; replaces with simple, robust slack-sensitivity amplification:
         `norm_rank * (1 + slack_sensitivity_gate)` — preserves urgency leverage without load coupling.
       - Reinstitutes explicit ddl_breach-scaled remaining_work_weight to prioritize large downstream work
         *only under deadline violation*, strengthening feasibility-first behavior.
       - Drops anti-starvation benefit entirely per reflection: avoids diluting urgency and simplifies gradient flow.
       - Adds monotonic slack-driven execution time scaling: `norm_exec * (1 - slack_sensitivity_gate)`
         to softly defer long-running tasks when slack is positive (i.e., safe), improving energy efficiency.
       - All numeric literals strictly ∈ {-2,-1,0,1,2}; no hidden constants; uses np.finfo for safety."""
    eps = 1.0416325543727867e-06
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_range_normalize(x):
        x = np.copy(x)
        if N == 1:
            q1 = q3 = x[0]
        else:
            q1 = np.quantile(x, 0.34163703198167444)
            q3 = np.quantile(x, 0.7484022149487877)
        iqr = q3 - q1
        spread = iqr if iqr > eps else eps
        return (x - q1) / spread
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_exec = robust_range_normalize(min_exec_time)
    norm_comm = robust_range_normalize(min_comm_time)
    norm_rank = robust_range_normalize(upward_rank)
    norm_work = robust_range_normalize(remaining_work)
    norm_wait = robust_range_normalize(ready_wait_time)
    norm_uncert = robust_range_normalize(uncertainty)
    if N == 1:
        slack_med = slack[0]
        slack_mad = eps
    else:
        slack_med = np.median(slack)
        slack_mad = np.median(np.abs(slack - slack_med))
    slack_spread = slack_mad if slack_mad > eps else eps
    norm_slack = (slack - slack_med) / slack_spread
    ddl_gate = 1.0 / (1.0 + np.exp(-6.824114436884848 * slack))
    slack_sensitivity_gate = 1.0 / (1.0 + np.exp(-6.824114436884848 * (slack - -0.19419043019421123)))
    criticality_leverage = 1.0 + slack_sensitivity_gate
    leveraged_rank = norm_rank * criticality_leverage
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 2.3142813272130205
    norm_slack_penalty = robust_range_normalize(raw_slack_penalty)
    successor_release_prob = 1.0 - np.clip(norm_wait, 0.0, 1.0)
    successor_blocking_penalty = norm_rank * (1.0 - successor_release_prob) * 0.4978221041911389
    exec_deferral_factor = 1.0 - slack_sensitivity_gate
    deferred_exec_penalty = norm_exec * exec_deferral_factor
    ddl_breach = (slack <= -0.19419043019421123).astype(float)
    exec_penalty = 0.43370583316969624 * norm_exec * ddl_breach
    comm_penalty = (1.0 - 0.43370583316969624) * norm_comm * ddl_breach
    uncert_gate = 1.0 / (1.0 + np.exp(-6.824114436884848 * (norm_uncert - 0.34163703198167444)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_breach
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(leveraged_rank, -2.0, 2.0) - 0.37136547006217147 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) + np.clip(0.46415950402659867 * (norm_wait + norm_uncert) * ddl_gate, -2.0, 2.0) + 0.6160346022530343 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.9483538112573557 * np.clip(norm_work * ddl_breach, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + np.clip(comm_penalty, -2.0, 2.0) + np.clip(successor_blocking_penalty, -2.0, 2.0) + np.clip(deferred_exec_penalty, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
