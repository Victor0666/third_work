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
    eps = 7.39997717567352e-07
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
    dur_uncert_blend = (1.0 + 0.8199843583370616 * uncert_norm) / (np.abs(dur_norm) + eps + 0.8199843583370616 * np.abs(uncert_norm) + eps)
    fairness_term = np.sign(dur_norm) * np.abs(dur_norm) ** 1.8804011484350753
    slack_norm = robust_normalize(slk + eps)
    slack_penalty = np.where(slk < 0, 2.817390527943639 * np.abs(slack_norm) ** 2, -4.0209009167783805 * np.abs(slack_norm))
    inv_energy = 1.0 / (np.clip(energy, eps, None) + eps)
    energy_score = -2.2636885516227974 * robust_normalize(inv_energy)
    rank_active = np.where(slk > -eps, rank, 0.0)
    rank_score = -0.6315987498113117 * robust_normalize(rank_active + eps)
    wait_sat = 1.0 - np.exp(-0.01631300457168519 * wait)
    wait_score = -robust_normalize(wait_sat + eps)
    slack_stress = np.where(slk < 0, np.abs(slack_norm), 0.0)
    unc_slack_interaction = 3.773475378321869 * uncert_norm * slack_stress
    score = slack_penalty + energy_score + rank_score + robust_normalize(fairness_term) + wait_score + unc_slack_interaction
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=np.median(score) if np.isfinite(np.median(score)) else 0.0, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values in final priority score'
    return score
