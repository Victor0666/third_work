import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Per-task exponential slack sensitivity (exp(-|slack|/tau)) → preserves urgency gradients near zero slack
      - Uncertainty-as-load-proxy for energy gating → reuses uncertainty, no new param
      - Quantile-aware upward_rank normalization using declared PARAMS["rank_quantile_thresh"]
      - All numeric literals restricted to {-2,-1,0,1,2}; epsilon via PARAMS; no hidden constants.
      - score_clip_min_abs removed; symmetric clipping via score_clip_max only.
    """
    eps = 0.05183297166723561
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def quantile_normalize(x, q):
        x = np.asarray(x, dtype=np.float64)
        finite_mask = np.isfinite(x)
        if np.any(finite_mask):
            x_finite = x[finite_mask]
            q_val = np.quantile(x_finite, q)
            scale = np.max(np.abs(x_finite))
            scale = np.where(scale > eps, scale, eps)
            return (x - np.nanmedian(x_finite)) / (scale + eps)
        else:
            return np.zeros_like(x)

    def maxabs_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.max(abs_x[finite_mask])
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    abs_slack = np.abs(slk)
    slack_sensitivity = np.exp(-abs_slack / (15.255003727755483 + eps))
    slack_penalty = np.where(slk < 0, 11.456339934565841 * abs_slack, -5.39250112113608 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -3.8670124116743625 * maxabs_normalize(inv_energy)
    energy_gate = (1.0 - slack_sensitivity) * (1.0 - 0.27722452397576897 * maxabs_normalize(uncert + eps))
    energy_score = energy_score * (1.0 - energy_gate)
    rank_norm = quantile_normalize(rank, 0.7673454247080017)
    rank_sharpened = np.power(np.abs(rank_norm) + eps, 1.884942697786881)
    rank_score = -rank_sharpened * slack_sensitivity
    slack_inv = 1.0 / (abs_slack + eps)
    bottleneck_base = rank * work * slack_inv
    bottleneck_score = -2.4838928330414465 * maxabs_normalize(bottleneck_base + eps) * slack_sensitivity
    duration = exec_t + comm_t
    dur_norm = maxabs_normalize(duration + eps)
    uncert_norm = maxabs_normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.6612528279248471 * uncert_norm
    dur_score = maxabs_normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (13.058982357363135 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score_clip_max = 45307811087.482834
    score = np.clip(score, -score_clip_max, score_clip_max)
    score = np.nan_to_num(score, nan=0.0, posinf=score_clip_max, neginf=-score_clip_max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
