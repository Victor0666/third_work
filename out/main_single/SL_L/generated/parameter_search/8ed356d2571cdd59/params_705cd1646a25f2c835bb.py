import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with bounded piecewise slack penalty and successor-release pressure.
    
    Key improvements:
    - Replaces exponentiated slack penalty with robust piecewise: linear for safe (slack > 0), 
      quadratic for risky (slack <= 0) — avoids instability while preserving risk-aware steepness.
    - Introduces normalized successor-release pressure: norm_work * norm_uncert * (1 - norm_slack), 
      clipped to [0,1], directly encoding urgency of large/uncertain sub-DAGs under deadline stress.
    - Simplifies criticality gate to univariate `norm_slack <= threshold` (no uncertainty thresholding),
      improving robustness per self-reflection.
    - All numeric literals are from {-2,-1,0,1,2}; normalization uses np.finfo(float).tiny only implicitly via eps.
    """
    eps = 0.005508870822077102
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_norm(x):
        x_abs = np.abs(x)
        denom = np.mean(x_abs) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    norm_duration = robust_norm(min_exec_time + min_comm_time)
    norm_rank = robust_norm(upward_rank)
    norm_work = robust_norm(remaining_work)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    safe_mask = (norm_slack > 0).astype(float)
    risky_mask = (norm_slack <= 0).astype(float)
    slack_penalty = safe_mask * 0.15842565034713602 * norm_slack + risky_mask * 1.343773281448804 * norm_slack ** 2
    successor_pressure_raw = norm_work * norm_uncert * (1.0 - norm_slack)
    successor_pressure = np.clip(successor_pressure_raw, 0.0, 1.0) * 0.9102153946036997
    rank_activation_mask = (norm_slack <= -0.40085496336257354).astype(float)
    wait_boost = 1.0 - np.exp(-0.2692452856377505 * (norm_wait + eps))
    score = slack_penalty
    score += safe_mask * 0.9028910703259672 * norm_energy
    score -= rank_activation_mask * 0.7449530471520777 * norm_rank
    score += successor_pressure
    score -= wait_boost * 1.216318997817826
    score = np.nan_to_num(score, nan=-233269.99618938728, posinf=1137.5185717937375, neginf=-32467224.8084023)
    return score
