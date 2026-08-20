import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining DDL-first safety, successor-aware duration,
    inverse-energy efficiency, and robust median-based normalization.
    
    Key improvements:
      - Robust MAD-normalization using np.median(abs(x - median)) with tunable scale
      - Successor-release-aware duration: exec + comm + successor_work_coupling * remaining_work
      - Slack-gated criticality: upward_rank contributes only if slack >= -epsilon
      - Quadratic slack penalty for violation risk; linear gain for positive slack
      - Exponential wait saturation prevents runaway priority drift
      - Uncertainty-slack interaction only under deadline stress (slk < 0)
      - All operations guarded against NaN/inf/zero; deterministic & finite output
    """
    eps = 0.00011196820309771408
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def robust_normalize(x):
        x = np.asarray(x)
        med = np.median(x)
        dev = x - med
        mad = np.median(np.abs(dev))
        scale = 0.39564981821268275 * (mad + eps)
        return dev / scale
    duration = exec_t + comm_t + 0.16778702997467676 * work
    norm_duration = robust_normalize(duration + eps)
    inv_energy = 1.0 / (energy + eps)
    norm_energy = robust_normalize(inv_energy)
    energy_score = -0.05426688931993454 * norm_energy
    norm_slack = robust_normalize(slk)
    slack_penalty = np.where(slk < 0, 2.3636794466095887 * norm_slack ** 2, -1.2995731673631208 * np.abs(norm_slack))
    rank_active = np.where(slk >= -eps, rank, 0.0)
    norm_rank = robust_normalize(rank_active + eps)
    rank_score = -1.231252403053928 * norm_rank
    norm_uncert = robust_normalize(uncert + eps)
    dur_uncert_blend = (1.0 + 0.765527599486253 * norm_uncert) / (np.abs(norm_duration) + eps + 0.765527599486253 * norm_uncert + eps)
    dur_score = robust_normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.2830334472208821 * wait)
    norm_wait = robust_normalize(wait_sat + eps)
    wait_score = -norm_wait
    slack_stress = np.where(slk < 0, np.abs(norm_slack), 0.0)
    unc_slack_interaction = 2.425258508680642 * norm_uncert * slack_stress
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction
    score = np.nan_to_num(score, nan=np.finfo(float).smallest_subnormal, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
