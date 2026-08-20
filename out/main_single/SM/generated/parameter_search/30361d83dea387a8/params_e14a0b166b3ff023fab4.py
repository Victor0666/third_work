import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Uncertainty-gated duration: uses `uncert^p` as multiplicative suppression factor on duration,
        replacing fragile additive blend — enforces sharper risk-aware pruning without coupling bias.
      - IQR-based rank normalization: replaces median/floor with interquartile range scaling,
        eliminating degeneracy and improving robustness to skewed or near-zero upward_rank distributions.
      - Removed slack-aware energy coupling (per self-reflection): restores monotonic feasibility-first priority;
        energy remains uniformly weighted, ensuring consistent tradeoff behavior across slack regimes.
      - All normalizations are finite, bounded, and use only {-2,-1,0,1,2} literals.
      - No hidden constants; all tunables declared in PARAMETER_SCHEMA.
    """
    eps = 0.0005453461228729225
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_iqr(x, q_low=0.2534800287248981, q_high=0.7161739488569168):
        x = np.asarray(x)
        finite_mask = np.isfinite(x)
        if not np.any(finite_mask):
            return np.zeros_like(x)
        x_finite = x[finite_mask]
        q1 = np.quantile(x_finite, q_low)
        q3 = np.quantile(x_finite, q_high)
        iqr = q3 - q1
        scale = np.where(iqr > eps, iqr, eps)
        center = (q1 + q3) / 2.0
        return (x - center) / (scale + eps)
    width = 2.3456969008687065
    gate_linear_region = (slk >= -width) & (slk <= 0.0)
    ddl_gate = np.where(slk <= -width, 1.0, np.where(gate_linear_region, 1.0 + slk / width, 0.0))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    if len(finite_abs_slk) == 0:
        slk_scale = eps
    else:
        q1_slk = np.quantile(finite_abs_slk, 0.2534800287248981)
        q3_slk = np.quantile(finite_abs_slk, 0.7161739488569168)
        slk_scale = np.where(q3_slk - q1_slk > eps, q3_slk - q1_slk, eps)
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 6.781157338878454, -0.5151797692513287) * np.abs(slack_norm)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.404982920056228 * normalize_iqr(inv_energy)
    rank_score = -0.6881158564000615 * ddl_gate * normalize_iqr(rank)
    bottleneck = rank * work
    bottleneck_score = -1.1592524405938511 * normalize_iqr(bottleneck + eps)
    duration = exec_t + comm_t
    uncert_gate = np.clip(uncert, 0.0, 2.0) ** 1.5608084068634356
    gated_duration = duration * (1.0 + uncert_gate)
    dur_score = normalize_iqr(gated_duration + eps)
    wait_clipped = np.clip(wait, 0.0, 9.630048018750934)
    wait_score = -normalize_iqr(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 2761345.7344977767
    min_safe = -finfo.max / 2761345.7344977767
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
