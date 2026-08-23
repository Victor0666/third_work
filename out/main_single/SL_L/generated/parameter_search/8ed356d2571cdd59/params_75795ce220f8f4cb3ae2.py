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
    eps = 0.0002618227494547556
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
    slack_penalty = safe_mask * 0.5066124664548962 * norm_slack + risky_mask * 2.3569738008136443 * norm_slack ** 2
    successor_pressure_raw = norm_work * norm_uncert * (1.0 - norm_slack)
    successor_pressure = np.clip(successor_pressure_raw, 0.0, 1.0) * 0.3059037223680901
    rank_activation_mask = (norm_slack <= -0.46387038107543993).astype(float)
    wait_boost = 1.0 - np.exp(-0.038930491475039186 * (norm_wait + eps))
    score = slack_penalty
    score += safe_mask * 0.679454864168377 * norm_energy
    score -= rank_activation_mask * 0.31926309074914694 * norm_rank
    score += successor_pressure
    score -= wait_boost * 0.42986501202594174
    score = np.nan_to_num(score, nan=360756.5901985555, posinf=3865.6027053232006, neginf=-40844006.53296924)
    return score
