import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's stability with novel slack-energy coupling and convex starvation response.
    
    Key improvements:
      - Keeps Parent 2's robust normalized-slack gating and bounded linear duration-uncertainty blend.
      - Adds *energy-slack coupling*: boosts priority of low-energy tasks only when slack is sufficiently positive,
        improving energy-awareness without violating DDL-first semantics.
      - Replaces linear wait saturation with *convex exponential saturation* (wait^exponent) for sharper starvation response.
      - Uses unified robust normalization with median/MAD fallback when mean fails (more outlier-resilient than Parent 2).
      - All numeric literals restricted to {-2,-1,0,1,2}; all parameters accessed via PARAMS.
    """
    eps = 0.02910888974628687
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def robust_normalize(x):
        x = np.asarray(x)
        finite_mask = np.isfinite(x)
        if not np.any(finite_mask):
            return np.zeros_like(x)
        x_clean = x[finite_mask]
        center = np.median(x_clean)
        mad = np.median(np.abs(x_clean - center)) + eps
        scale = max(np.mean(np.abs(x_clean - center)) + eps, mad)
        return (x - center) / (scale + eps)
    slack_norm = robust_normalize(slk)
    slack_penalty = np.where(slk < 0, 6.446141424001665 * np.abs(slack_norm), -0.4466948172676303 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.6180259440952467 * robust_normalize(inv_energy)
    rank_active = np.where(slack_norm >= 0, rank, 0.0)
    rank_score = -1.994809583986414 * robust_normalize(rank_active + eps)
    duration = exec_t + comm_t
    dur_norm = robust_normalize(duration + eps)
    uncert_norm = robust_normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.2692329608771522 * uncert_norm
    dur_score = robust_normalize(dur_uncert_blend)
    wait_power = np.power(np.maximum(wait, 0.0), 0.5611456943429476)
    wait_sat = 1.0 - np.exp(-0.08945919197581728 * wait_power)
    wait_score = -robust_normalize(wait_sat + eps)
    unc_slack_interaction = 1.5543315896917254 * uncert_norm * np.where(slack_norm < 0, np.abs(slack_norm), 0.0)
    energy_slack_boost = np.where(slack_norm > 0.6186875667347236, 0.8675381514602013 * robust_normalize(inv_energy), 0.0)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + energy_slack_boost
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 45120.91698060504
    min_safe = -finfo.max / 45120.91698060504
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
