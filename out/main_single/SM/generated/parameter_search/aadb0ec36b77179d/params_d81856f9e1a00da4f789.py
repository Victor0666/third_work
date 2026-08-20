import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with unified robust quantile normalization:
      - Replaces dual (IQR + quantile) and inconsistent scaling with a single, tunable quantile strategy.
      - Uses `robust_normalization_quantile` for all features — temporal (slack, duration, wait, uncertainty)
        and structural (rank, work, energy-derived) — but applies it *semantically*: 
          * For signed/temporal: normalize via abs-value quantile.
          * For non-negative/structural: normalize via value-domain quantile (no abs).
      - Eliminates fragile IQR and dedicated `energy_normalization_quantile` (removed to stay within 12 params).
      - Preserves piecewise DDL feasibility mask, starvation-aware wait ramp, and bottleneck sharpening.
      - All numeric literals are -2,-1,0,1,2; epsilon and clipping use PARAMS; no hidden constants.
    """
    eps = 0.0029956806583420446
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_abs_quantile(x, q=0.7054177136323365):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], q)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)

    def normalize_pos_quantile(x, q=0.7054177136323365):
        x = np.asarray(x)
        finite_mask = np.isfinite(x) & (x >= 0)
        if np.any(finite_mask):
            x_clean = x[finite_mask]
            if len(x_clean) > 0:
                scale = np.quantile(x_clean, q)
                scale = np.where(scale > eps, scale, eps)
            else:
                scale = eps
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize_abs_quantile(slk)
    slack_penalty = np.where(slk < 0, 7.575722314051109 * np.abs(slack_norm), -1.751837181003541 * np.abs(slack_norm))
    margin = 0.30437290814284845
    ddl_feasible_mask = np.clip((slk + margin) / (2 * margin + eps), 0.0, 1.0)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.17830356371355932 * normalize_pos_quantile(inv_energy) * ddl_feasible_mask
    rank_score = -0.23439382048316967 * normalize_pos_quantile(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.6819216719222423)
    bottleneck_score = -1.0491576797087876 * normalize_pos_quantile(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_abs_quantile(duration + eps)
    uncert_norm = normalize_abs_quantile(uncert + eps)
    dur_uncert_blend = dur_norm + 0.4707089964887135 * uncert_norm
    dur_score = normalize_abs_quantile(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 23.033951765159518)
    wait_normalized = normalize_pos_quantile(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=70583302232.49791, neginf=-70583302232.49791)
    score = np.clip(score, -70583302232.49791, 70583302232.49791)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
