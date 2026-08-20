import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining DDL-hard constraint enforcement, robust criticality gating,
    inverse energy efficiency, exponential wait saturation, and MAD-based normalization.
    
    Key improvements over parents:
      - Uses median absolute deviation (MAD) with tunable scale for robust normalization,
        more stable than mean-abs under skewed ready-task distributions.
      - Criticality strictly gated: upward_rank contributes *only* if slack > -epsilon (hard DDL feasibility).
      - Slack penalty: quadratic for negative slack, linear gain for positive — matches urgency semantics.
      - Wait bias uses exponential saturation (1 - exp(-rate * wait)) to bound starvation correction.
      - Uncertainty-slack interaction is multiplicative and only active under deadline stress.
      - All operations safeguarded against NaN/inf using np.finfo and finite fallbacks.
    """
    eps = 1.9915615170542622e-07
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
        mad = np.median(np.abs(x - med))
        scale = 2.116819269781651 * (mad + eps)
        return (x - med) / scale
    duration = exec_t + comm_t
    slack_norm = robust_normalize(slk)
    slack_penalty = np.where(slk < 0, 5.238643273149016 * slack_norm ** 2, -5.583010123071031 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.4130045706219643 * robust_normalize(inv_energy)
    rank_active = np.where(slk > -eps, rank, 0.0)
    rank_score = -1.4903298996668202 * robust_normalize(rank_active + eps)
    dur_norm = robust_normalize(duration + eps)
    uncert_norm = robust_normalize(uncert + eps)
    dur_uncert_blend = (1.0 + 0.9607162531770951 * uncert_norm) / (dur_norm + eps + 0.9607162531770951 * uncert_norm + eps)
    dur_score = robust_normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.23743474413716747 * wait)
    wait_score = -robust_normalize(wait_sat + eps)
    slack_stress = np.where(slk < 0, np.abs(slack_norm), 0.0)
    unc_slack_interaction = 0.30353631533592274 * uncert_norm * slack_stress
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
