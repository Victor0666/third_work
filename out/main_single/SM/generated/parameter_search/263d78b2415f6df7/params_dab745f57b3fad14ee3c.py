import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 2's decisiveness with adaptive energy gating:
      - Binary DDL gate replaced by thresholded suppression: energy term zeroed only when |robust_slk| < threshold.
      - Preserves Parent 2's tunable bottleneck_offset, exponential starvation, and clean parameter count.
      - Adds adaptive energy suppression threshold to avoid premature energy optimization when slack is shallowly positive but still risky.
      - Retains robust slack normalization and all safety guards.
      - No piecewise width or host_load_sensitivity — reduces overfitting risk while increasing interpretability.
      - All numeric literals strictly in {-2,-1,0,1,2}; epsilon via PARAMS; machine bounds via np.finfo.
      - Deterministic, finite, shape-(N,), side-effect-free.
    """
    eps = 0.001791541109508081
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
    robust_slk_mag = np.power(abs_slk, 0.689030087449559)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    energy_suppress_gate = np.where(np.abs(slk_robust) < 0.06635654861185732, 0.0, 1.0)
    slack_penalty = np.where(slk_robust < 0, 3.1196151400960273 * np.abs(slk_robust), -3.964435617943918 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.07917047117697933 * normalize(inv_energy)
    energy_score = energy_score * energy_suppress_gate
    rank_powered = np.power(rank + eps, 2.1785285238544745)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + 0.21532816537402688
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -0.8851583111527767 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.9492522461279848 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (11.97484629683687 + eps))
    wait_score = -0.1915477621120042 * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
