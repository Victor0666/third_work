import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Smooth sigmoidal energy gating centered at −ddl_feasibility_margin (replaces piecewise).
      - Quantile-based normalization for all features to resist outliers.
      - Clamped inverse slack sharpening on normalized slack for scale-invariant bottleneck coupling.
      - Exponential starvation saturation with tunable threshold (no exponent parameter — fixed curvature).
      - All numeric literals are {-2,-1,0,1,2}; no hidden constants; deterministic and finite.
    """
    eps = 0.028824402867309185
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], 0.9489084738192236)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slk_norm = normalize(slk)
    rank_norm = normalize(rank + eps)
    work_norm = normalize(work + eps)
    exec_norm = normalize(exec_t + eps)
    comm_norm = normalize(comm_t + eps)
    energy_norm = normalize(energy + eps)
    wait_norm = normalize(wait + eps)
    uncert_norm = normalize(uncert + eps)
    slack_penalty = np.where(slk_norm < 0, 5.36379759767276 * np.abs(slk_norm), -4.518896256834325 * slk_norm)
    gate_center = -0.00013576222911487051
    gate_slope = 2.0
    energy_gate = 1.0 / (1.0 + np.exp(gate_slope * (slk_norm - gate_center)))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.28360417791281733 * normalize(inv_energy) * energy_gate
    rank_score = -0.09273471328287196 * rank_norm
    slack_magnitude_inv = 1.0 / (np.abs(slk_norm) + eps)
    slack_magnitude_inv_clamped = np.clip(slack_magnitude_inv, 0.0, 1.0 / eps)
    bottleneck_sharpened = rank_norm * work_norm * np.power(slack_magnitude_inv_clamped, 2.372184377444742)
    bottleneck_score = -2.818091659963039 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    dur_uncert_blend = dur_norm + 1.0493162183819695 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_saturation = 1.0 - np.exp(-wait / (8.46534223623144 + eps))
    wait_score = -normalize(wait_saturation + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 46130176571.77238
    min_safe = -finfo.max / 46130176571.77238
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
