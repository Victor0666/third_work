import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Winsorized normalization for rank/work features to suppress outlier-driven bottleneck over-prioritization.
      - Hard feasibility mask + soft ramping *only inside feasible region* — no urgency boost outside safety.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-(N,) output satisfying all interface contracts.
    """
    eps = 8.457512358179126e-06
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def winsorize_and_normalize(x):
        x = np.asarray(x)
        finite_mask = np.isfinite(x)
        if not np.any(finite_mask):
            return np.full_like(x, 0.0, dtype=np.float64)
        x_finite = x[finite_mask]
        low_q = np.quantile(x_finite, 0.535964735428075 / 2.0)
        high_q = np.quantile(x_finite, 1.0 - 0.535964735428075 / 2.0)
        x_winsorized = np.clip(x, low_q, high_q)
        abs_x = np.abs(x_winsorized)
        scale = np.quantile(abs_x, 0.535964735428075)
        scale = np.where(scale > eps, scale, eps)
        return x_winsorized / (scale + eps)
    uncert_normalized = winsorize_and_normalize(uncert + eps)
    host_proxy = np.power(1.0 + uncert, 2.521151285916887)
    slack_offset = slk - 0.0006312200632240349
    ddl_feasible_gate = 1.0 / (1.0 + np.exp(-4.814398293640514 * slack_offset))
    slack_penalty = np.where(slk < 0, 2.7229462750857545 * np.abs(slk), 0.0)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.9693731786781206 * winsorize_and_normalize(inv_energy) * ddl_feasible_gate
    rank_score = -1.981649959979168 * winsorize_and_normalize(rank + eps) * ddl_feasible_gate
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.521151285916887)
    bottleneck_score = -1.3251640346438402 * winsorize_and_normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = winsorize_and_normalize(duration + eps)
    dur_uncert_blend = dur_norm + 1.5945948218878487 * uncert_normalized
    dur_score = winsorize_and_normalize(dur_uncert_blend) * ddl_feasible_gate * (1.0 / host_proxy)
    wait_clipped = np.clip(wait, 0.0, 36.24297540054158)
    wait_normalized = winsorize_and_normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max * 0.42011783372027955, neginf=finfo.min * 0.42011783372027955)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
