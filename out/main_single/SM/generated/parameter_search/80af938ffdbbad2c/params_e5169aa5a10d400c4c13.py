import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with DDL-protection gating, bounded linear starvation correction, and concave duration-uncertainty blending.
    
    Key structural improvements:
      - Introduces *DDL-protection margin*: criticality activates only when slack_norm < margin (not >= 0), preventing premature de-prioritization
        of high-rank tasks under mild latency pressure — enforces hard-deadline semantics first.
      - Replaces convex wait-saturation with *bounded linear starvation correction*: avoids distortion from old-task dominance and stabilizes ordering.
      - Upgrades duration-uncertainty coupling to *concave power blend* (dur^α + ratio * uncert^α) for better separation in high-uncertainty regimes.
      - Removes fragile energy-slack coupling per reflection; retains strict DDL-first feasibility assurance.
      - All normalizations use median/MAD-based robust scaling; all outputs guaranteed finite and shape-(N,).
    """
    eps = 1.0648896942909377e-05
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
    slack_penalty = np.where(slk < 0, 0.9773243299428311 * np.abs(slack_norm), -0.1293749813873726 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.489584441595591 * robust_normalize(inv_energy)
    rank_active = np.where(slack_norm < 0.8015217911328437, rank, 0.0)
    rank_score = -1.2525893237822752 * robust_normalize(rank_active + eps)
    duration = exec_t + comm_t
    dur_norm = robust_normalize(duration + eps)
    uncert_norm = robust_normalize(uncert + eps)
    alpha = 0.28564768876862223
    dur_pow = np.power(np.abs(dur_norm) + eps, alpha)
    uncert_pow = np.power(np.abs(uncert_norm) + eps, alpha)
    dur_uncert_blend = dur_pow + 0.9290061702666053 * uncert_pow
    dur_score = robust_normalize(dur_uncert_blend)
    wait_linear = 0.060575552919036685 * wait
    wait_score = -robust_normalize(wait_linear + eps)
    unc_slack_interaction = 1.253740167714532 * uncert_norm * np.where(slack_norm < 0, np.abs(slack_norm), 0.0)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 5391110.6844408605
    min_safe = -finfo.max / 5391110.6844408605
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
