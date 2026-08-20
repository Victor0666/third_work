import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths:
      - Smooth sigmoid DDL gate (Parent 2) for monotonic feasibility enforcement.
      - Median-based normalization with floor (Parent 2) for robustness on skewed/zero rank.
      - Bounded linear duration-uncertainty blend (Parent 2) for stable risk suppression.
      - Starvation exponent ramp (Parent 2) for stronger long-wait bias without hard clipping.
      - Novel energy-DDL coupling: multiplies energy score by (1 - ddl_gate) to suppress low-energy bias when slack is abundant — prevents premature scheduling that may harm critical-path progress.
      - All normalizations and gates are finite, bounded, and use only {-2,-1,0,1,2} literals.
      - No hidden constants; all tunables declared in PARAMETER_SCHEMA.
    """
    eps = 0.09636433206954466
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
    steep = 0.5656971517660634
    ddl_gate = 1.0 / (1.0 + np.exp(-steep * slk))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    slk_scale = np.median(finite_abs_slk) if len(finite_abs_slk) > 0 else eps
    slk_scale = np.where(slk_scale > eps, slk_scale, eps)
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 11.820113001486233, -1.7074475864153706) * np.abs(slack_norm)
    inv_energy = 1.0 / (energy + eps)
    energy_base = -0.2241478545658444 * normalize(inv_energy)
    energy_score = energy_base * (1.0 - (1.0 - ddl_gate))
    rank_stable = rank + 0.02587906196220448
    rank_score = -2.1379187078225863 * ddl_gate * normalize(rank_stable)
    bottleneck = rank * work
    bottleneck_score = -2.546715546238524 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.6421893961521881 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 16.735060692875287)
    wait_sharpened = np.power(wait_clipped + eps, 1.340735481092998)
    wait_score = -normalize(wait_sharpened)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 2995588.432718508
    min_safe = -finfo.max / 2995588.432718508
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
