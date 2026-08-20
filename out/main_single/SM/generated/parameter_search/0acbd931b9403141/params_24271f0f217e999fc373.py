import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Smooth sigmoid DDL gate (replacing piecewise linear) for differentiability and stronger monotonic feasibility.
      - Preserves sharpened power-law starvation mitigation and robust median normalization.
      - Removes load_interaction_weight (reducing parameter count to 12) while retaining all core structural improvements.
      - All operations guarded against NaN/inf/zero; no hidden constants beyond {-2,-1,0,1,2}.
      - Final clamping uses tunable finfo_max_scale for safety and portability.
    """
    eps = 0.000656060843097884
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x, floor=eps):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            med = np.median(abs_x[finite_mask])
            scale = np.maximum(med, floor)
        else:
            scale = floor
        return x / (scale + eps)
    steep = 2.6104416456391673
    ddl_gate = 1.0 / (1.0 + np.exp(-steep * slk))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    slk_scale = np.median(finite_abs_slk) if len(finite_abs_slk) > 0 else eps
    slk_scale = np.where(slk_scale > eps, slk_scale, eps)
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 0.7249430144544833, -1.1369158174929175) * np.abs(slack_norm)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.3300540421004812 * normalize(inv_energy)
    rank_stable = rank + 0.022595698585093613
    rank_score = -0.8472754842918714 * ddl_gate * normalize(rank_stable)
    bottleneck = rank * work
    bottleneck_score = -0.5562572918614712 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.1165841809082502 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 11.290517878239019)
    wait_sharpened = np.power(wait_clipped + eps, 1.4723194394245365)
    wait_score = -normalize(wait_sharpened)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 194760891.09130266
    min_safe = -finfo.max / 194760891.09130266
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
