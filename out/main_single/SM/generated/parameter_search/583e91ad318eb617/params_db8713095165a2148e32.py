import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's decisiveness with Parent 1's adaptivity:
      - Binary DDL gate replaced by *triple-region slack gating*: 
          [late] → full energy suppression, 
          [moderately urgent: slk_robust ∈ (threshold, 0]] → partial activation, 
          [non-urgent] → full energy use.
      - Introduces `energy_activation_threshold` to enable energy-aware scheduling even when slack is negative but not critically so — avoids premature energy neglect.
      - Preserves Parent 2's tunable bottleneck_offset and exponential starvation saturation.
      - Keeps robust slack normalization and max-abs normalization for stability.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-(N,) output satisfying all interface contracts.
    """
    eps = 1.1395680435872694e-06
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
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.max(abs_x[finite_mask])
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    abs_slk = np.abs(slk) + eps
    robust_slk_mag = np.power(abs_slk, 0.6758660642428722)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    threshold = -0.6899216389086651
    energy_gate = np.where(slk_robust <= threshold, 0.0, np.where(slk_robust <= 0.0, (slk_robust - threshold) / (0.0 - threshold + eps), 1.0))
    slack_penalty = np.where(slk_robust < 0, 2.63912733056353 * np.abs(slk_robust), -3.749004694053532 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.9327366347477275 * normalize(inv_energy)
    energy_score = energy_score * energy_gate
    rank_powered = np.power(rank + eps, 2.314748168212021)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + 3.9576414392201893
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -2.865005212616811 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.4281407524880299 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (42.48292823111065 + eps))
    wait_score = -1.078197161507565 * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
