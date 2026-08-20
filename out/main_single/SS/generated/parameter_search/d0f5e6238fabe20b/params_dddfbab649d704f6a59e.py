import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule:
       - Replaces adaptive power-law slack penalty with *bounded linear urgency ramp* under dual gate for numerical stability and interpretability.
       - Introduces explicit *congestion-thresholded starvation relief*: wait_benefit scaled by (1 - risk_gate) to prevent interference with deadline pressure.
       - Removes redundant rank_boost_upper_bound; uses dual-gated rank scaling + lower-bound clipping only.
       - All nonlinearities are bounded, all divisions guarded, all outputs finite and shape-(N,)."""
    eps = 0.00012388960961325782
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
            xmin = xmax = x[0]
        else:
            xmin = np.min(x)
            xmax = np.max(x)
        spread = xmax - xmin if xmax - xmin > eps else eps
        return (x - xmin) / spread
    norm_slack = robust_range_normalize(slack)
    norm_energy = robust_range_normalize(min_incremental_energy)
    norm_rank = robust_range_normalize(upward_rank)
    norm_work = robust_range_normalize(remaining_work)
    norm_wait = robust_range_normalize(ready_wait_time)
    norm_uncert = robust_range_normalize(uncertainty)
    congestion = ready_wait_time + uncertainty
    norm_congestion = robust_range_normalize(congestion)
    ddl_gate = (slack <= 0.0).astype(float)
    risk_gate = 1.0 / (1.0 + np.exp(-3.9728920333787068 * (norm_congestion - 0.30615147561159584)))
    dual_gate = ddl_gate * risk_gate
    critical_path_leverage = norm_rank * norm_work * dual_gate
    urgency_ramp = np.clip(3.6478078517531514 * (1.0 - norm_slack), 0.0, 1.0) * ddl_gate
    norm_urgency_ramp = robust_range_normalize(urgency_ramp)
    wait_benefit = 1.0 - np.exp(-0.36342171868675505 * ready_wait_time)
    congestion_suppressed_wait = wait_benefit * (1.0 - risk_gate)
    energy_uncert_penalty = norm_energy * norm_uncert * dual_gate
    duration = min_exec_time + min_comm_time
    norm_duration = robust_range_normalize(duration)
    duration_uncert_penalty = norm_duration * norm_uncert * dual_gate
    rank_boost = 1.0 + 1.7795304296122958 * urgency_ramp
    rank_boost = np.clip(rank_boost, 0.3959073776797434, 2.0)
    boosted_rank = norm_rank * rank_boost
    score = +np.clip(norm_urgency_ramp, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(boosted_rank, -2.0, 2.0) - np.clip(congestion_suppressed_wait, -2.0, 2.0) - 0.508400619285946 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) + 0.46537646963260587 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 0.32872792261898726 * np.clip(duration_uncert_penalty, -2.0, 2.0) + 0.36778284934183597 * np.clip(norm_work * ddl_gate, -2.0, 2.0) + np.clip(norm_congestion * ddl_gate, -2.0, 2.0) + 0.14762435223973264 * np.clip(-congestion_suppressed_wait, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
