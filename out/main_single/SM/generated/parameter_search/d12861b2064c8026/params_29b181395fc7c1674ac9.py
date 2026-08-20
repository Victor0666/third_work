import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule synthesizing best practices:
    - DDL compliance first: piecewise quadratic slack penalty + strict criticality gating.
    - Energy prioritization via inverse scaling with robust clipping.
    - Uncertainty-slack interaction only under deadline stress.
    - Exponential wait decay for anti-starvation with bounded saturation.
    - Adaptive fairness: power-law penalty on duration deviation to avoid bias toward extremes.
    - All operations guarded against NaN/inf/zero; deterministic and finite-output guaranteed.
    Smaller score = higher priority.
    """
    eps = 0.00026466467255859623
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
        center = np.median(x)
        dev = np.abs(x - center)
        scale = np.median(dev) if len(dev) > 0 else 1.0
        scale = max(scale, eps)
        return (x - center) / scale
    duration = exec_t + comm_t
    dur_norm = robust_normalize(duration + eps)
    uncert_norm = robust_normalize(uncert + eps)
    dur_uncert_blend = (1.0 + 1.0425184325681642 * uncert_norm) / (np.abs(dur_norm) + eps + 1.0425184325681642 * np.abs(uncert_norm) + eps)
    fairness_term = np.sign(dur_norm) * np.abs(dur_norm) ** 0.9932488923672866
    slack_norm = robust_normalize(slk + eps)
    slack_penalty = np.where(slk < 0, 7.022469324712838 * np.abs(slack_norm) ** 2, -1.1657322959772969 * np.abs(slack_norm))
    inv_energy = 1.0 / (np.clip(energy, eps, None) + eps)
    energy_score = -1.4870620944026434 * robust_normalize(inv_energy)
    rank_active = np.where(slk > -eps, rank, 0.0)
    rank_score = -0.7543651730834711 * robust_normalize(rank_active + eps)
    wait_sat = 1.0 - np.exp(-0.15711559080753348 * wait)
    wait_score = -robust_normalize(wait_sat + eps)
    slack_stress = np.where(slk < 0, np.abs(slack_norm), 0.0)
    unc_slack_interaction = 2.5046459293302354 * uncert_norm * slack_stress
    score = slack_penalty + energy_score + rank_score + robust_normalize(fairness_term) + wait_score + unc_slack_interaction
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=np.median(score) if np.isfinite(np.median(score)) else 0.0, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values in final priority score'
    return score
