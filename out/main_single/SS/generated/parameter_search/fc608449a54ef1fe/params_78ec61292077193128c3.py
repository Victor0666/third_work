import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: combines Parent 2's robust quantile normalization and sign-preserving MAD for slack,
       with Parent 1's exec/comm separation and wait saturation control; removes unused criticality_scale;
       introduces unified 'urgency surface' combining raw_slack_pressure, norm_rank, and norm_work via bounded product;
       uses only one criticality signal instead of duplicated gates; ensures all feature interactions are clipped and bounded."""
    eps = 0.0004464725586773364
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
            q1 = np.quantile(x, 0.3554716489104056)
            q3 = np.quantile(x, 0.7473538759912033)
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
    ddl_gate = 1.0 / (1.0 + np.exp(-3.8619553309036645 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_pressure = np.clip(-slack, 0.0, 1.0)
    urgency_surface = np.clip(norm_rank * norm_work * raw_slack_pressure, -2.0, 2.0)
    exec_penalty = 0.5563969209052969 * norm_exec * raw_slack_pressure * ddl_gate
    comm_penalty = (1.0 - 0.5563969209052969) * norm_comm * raw_slack_pressure * ddl_gate
    uncert_gate = 1.0 / (1.0 + np.exp(-3.8619553309036645 * (norm_uncert - 0.43194176311360233)))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    wait_normalized = norm_wait
    wait_benefit = np.clip(wait_normalized / (1.2340800485994685 + eps), 0.0, 1.0)
    energy_slack_penalty = norm_energy * raw_slack_pressure * ddl_gate
    score = +np.clip(norm_slack, -2.0, 2.0) + np.clip(raw_slack_pressure ** 1.4459704633992299, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(urgency_surface, -2.0, 2.0) - 0.44253817641365734 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.6148724138995983 * np.clip(energy_uncert_penalty, -2.0, 2.0) + np.clip(exec_penalty, -2.0, 2.0) + np.clip(comm_penalty, -2.0, 2.0) + 0.8110575268836768 * np.clip(norm_work * (1.0 - ddl_gate), -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
