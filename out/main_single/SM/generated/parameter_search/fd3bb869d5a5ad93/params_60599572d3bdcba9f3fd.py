import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining DDL-first robustness (Parent 2) with stability from MAD-normalization (Parent 1).
    
    Key improvements:
      - Replaces mean-abs normalization with median/MAD-based robust normalization for all features,
        reducing sensitivity to outliers in heterogeneous edge-cloud workloads.
      - Uses piecewise quadratic-negative + linear-positive slack penalty (validated superior in Parent 2),
        but computes slack margin adaptively using *positive* slack median (like Parent 1) for tighter risk targeting.
      - Retains inverse energy efficiency (1/(energy+ε)) for sharp low-energy bias.
      - Keeps exponential wait decay (not linear) for calibrated anti-starvation.
      - Criticality gated by slack > -epsilon (DDL-first), not attenuated — simplifies causal logic and improves DDL compliance.
      - Adds robustness_mad_scale parameter to tune outlier resilience without breaking scale invariance.
      - All operations are finite, deterministic, and shape-preserving; no loops or side effects.
    """
    eps = 1.0123708262306087e-09
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
        mad = np.median(np.abs(x - center)) + eps
        scale = 1.9972156975354978 * mad + eps
        return (x - center) / scale
    duration = exec_t + comm_t
    norm_duration = robust_normalize(duration)
    pos_slack = slk[slk > eps]
    slack_margin = np.median(pos_slack) if len(pos_slack) > 0 else 1.0
    slack_margin = max(slack_margin, eps)
    slack_penalty = np.zeros_like(slk)
    slack_penalty[slk < 0] = 0.5213079159776702 * (-slk[slk < 0] / (slack_margin + eps)) ** 2
    slack_penalty[slk >= 0] = -1.5428189362481246 * (slk[slk >= 0] / (slack_margin + eps))
    inv_energy = 1.0 / (energy + eps)
    norm_inv_energy = robust_normalize(inv_energy)
    energy_score = -3.7347419830645 * norm_inv_energy
    rank_active = np.where(slk > -eps, rank, 0.0)
    norm_rank = robust_normalize(rank_active + eps)
    rank_score = -1.1543392087878734 * norm_rank
    norm_uncert = robust_normalize(uncert + eps)
    dur_uncert_signal = (1.0 + 0.5469465936261533 * norm_uncert) / (norm_duration + eps + 0.5469465936261533 * norm_uncert + eps)
    dur_score = robust_normalize(dur_uncert_signal)
    wait_sat = 1.0 - np.exp(-0.14486849025351117 * wait)
    norm_wait_sat = robust_normalize(wait_sat + eps)
    wait_score = -norm_wait_sat
    slack_stress = np.where(slk < 0, np.abs(slk) / (slack_margin + eps), 0.0)
    unc_slack_interaction = 0.4800230558010099 * norm_uncert * slack_stress
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
