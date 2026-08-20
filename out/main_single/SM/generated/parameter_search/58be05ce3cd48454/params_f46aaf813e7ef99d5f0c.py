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
    eps = 1.3772558523924931e-06
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
        scale = 1.2034155419290111 * (mad + eps)
        return dev / scale
    duration = exec_t + comm_t + 0.20003590997044726 * work
    norm_duration = robust_normalize(duration + eps)
    inv_energy = 1.0 / (energy + eps)
    norm_energy = robust_normalize(inv_energy)
    energy_score = -0.7353279296826007 * norm_energy
    norm_slack = robust_normalize(slk)
    slack_penalty = np.where(slk < 0, 4.683055151200116 * norm_slack ** 2, -2.478421106109485 * np.abs(norm_slack))
    rank_active = np.where(slk >= -eps, rank, 0.0)
    norm_rank = robust_normalize(rank_active + eps)
    rank_score = -0.9352959323732495 * norm_rank
    norm_uncert = robust_normalize(uncert + eps)
    dur_uncert_blend = (1.0 + 0.9005049477948179 * norm_uncert) / (np.abs(norm_duration) + eps + 0.9005049477948179 * norm_uncert + eps)
    dur_score = robust_normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.44493712606771557 * wait)
    norm_wait = robust_normalize(wait_sat + eps)
    wait_score = -norm_wait
    slack_stress = np.where(slk < 0, np.abs(norm_slack), 0.0)
    unc_slack_interaction = 4.093330650154359 * norm_uncert * slack_stress
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction
    score = np.nan_to_num(score, nan=np.finfo(float).smallest_subnormal, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
