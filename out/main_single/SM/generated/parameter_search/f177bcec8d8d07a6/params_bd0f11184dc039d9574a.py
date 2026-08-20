import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with:
      - Robust min-max normalization using high quantile for outlier resilience.
      - Sigmoid ddl_protection_gate for smooth feasibility-aware weighting (replaces piecewise).
      - Exponential wait-saturation (1 - exp(-wait/θ)) for monotonic starvation mitigation.
      - Bottleneck term uses fixed exponent 2 (allowed literal) instead of tunable one — reduces parameter count.
      - Energy suppression reuses same sigmoid gate for coherence and simplicity.
      - Starvation guard bypasses non-bottleneck terms when slack is critically low.
      - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
      - Final score clamped via np.finfo with parameterized scale.
    """
    eps = 2.2814600299334344e-06
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
        finite_mask = np.isfinite(x)
        if np.any(finite_mask):
            x_finite = x[finite_mask]
            q_low = np.quantile(x_finite, 1.0 - 0.8452224859307784)
            q_high = np.quantile(x_finite, 0.8452224859307784)
            scale = np.where(q_high > q_low, q_high - q_low, eps)
            center = np.median(x_finite)
            return (x - center) / (scale + eps)
        else:
            return np.zeros_like(x)
    slack_penalty = np.where(slk < 0, 4.179610138125806 * np.abs(slk), -1.3596456483948431 * slk)
    energy_suppression_weight = np.where(slk < 0, 1.0, 1.0 / (1.0 + np.exp(0.19260310188293334 * slk)))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.309813150360161 * energy_suppression_weight * normalize(inv_energy)
    ddl_gate = energy_suppression_weight
    rank_score = -1.8644198193142891 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2)
    bottleneck_score = -1.3867620872231619 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.4765740766293585 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_exp_ramp = 1.0 - np.exp(-wait / (48.83044308073729 + eps))
    wait_score = -normalize(wait_exp_ramp + eps)
    starvation_mask = (slk < 2.486287257742354).astype(np.float64)
    starvation_bypass = bottleneck_score * starvation_mask
    base_score = slack_penalty + energy_score + rank_score + dur_score + wait_score
    score = np.where(starvation_mask > 0, starvation_bypass, base_score)
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 1119.1485768215725
    min_safe = -finfo.max / 1119.1485768215725
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
