import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's sharp deadline risk handling and inverse-energy scoring
    with Parent 1's robust fairness term and anti-starvation wait bias. Key improvements:
      - Quadratic slack penalty (Parent 2) + soft linear gain for positive slack
      - Inverse-energy scoring (1/(energy+eps)) for sharper low-energy preference
      - Criticality gated by slack >= -epsilon (not just > 0), improving boundary stability
      - Uncertainty-slack interaction applied *only* under negative slack (tighter coupling)
      - Duration fairness via tanh-scaled normalized duration (Parent 1) to avoid load skew
      - Wait-time exponential saturation (Parent 2) + linear fallback near zero for stability
      - All normalizations use mean-abs scale + epsilon for stability and boundedness
    Smaller score = higher priority.
    """
    eps = 6.567211224034214e-09
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        scale = np.mean(np.abs(x)) + eps
        return x / scale
    duration = exec_t + comm_t
    norm_duration = normalize(duration + eps)
    slack_norm = normalize(slk + eps)
    slack_penalty = np.where(slk < 0, 1.2072134336730551 * slack_norm ** 2, -0.5431258773888772 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.7823347672028083 * normalize(inv_energy)
    rank_active = np.where(slk >= -eps, rank, 0.0)
    rank_score = -0.5752076482456611 * normalize(rank_active + eps)
    slack_stress_mask = (slk < 0).astype(float)
    norm_uncert = normalize(uncert + eps)
    unc_slack_interaction = 2.34907963267983 * norm_uncert * slack_stress_mask
    wait_linear_fallback = 2.857740002926246 * wait
    wait_sat = np.where(wait < eps, wait_linear_fallback, 1.0 - np.exp(-0.1955118454961887 * wait))
    wait_score = -normalize(wait_sat + eps)
    fairness_term = np.tanh(norm_duration) * norm_duration
    score = slack_penalty + energy_score + rank_score + unc_slack_interaction + wait_score + 0.5614021550435491 * fairness_term
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 1.141183066281122
    min_safe = finfo.min / 1.141183066281122
    score = np.nan_to_num(score, nan=np.median(score) if np.any(np.isfinite(score)) else 0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values remain after sanitization'
    return score
