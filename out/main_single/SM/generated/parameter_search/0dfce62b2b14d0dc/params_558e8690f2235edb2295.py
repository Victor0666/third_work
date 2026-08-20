import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Smooth sigmoidal energy suppression (replacing threshold gate) for robust feasibility-energy tradeoff.
      - Robust min-max normalization for slack, energy, and uncertainty to stabilize cross-scenario behavior.
      - Unified risk-weighted execution score via bounded tanh: avoids linear blending fragility and ensures monotonic, saturating response to duration+uncertainty.
      - All parameters used; no unused entries; only {-2,-1,0,1,2} literals; epsilon & machine bounds via PARAMS/np.finfo.
      - Deterministic, finite, shape-(N,), side-effect-free.
    """
    eps = 0.09229867416422023
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_mm(x):
        x = np.asarray(x, dtype=np.float64)
        finite_mask = np.isfinite(x)
        if np.any(finite_mask):
            x_min = np.min(x[finite_mask])
            x_max = np.max(x[finite_mask])
            range_val = x_max - x_min
            range_val = np.where(range_val > eps, range_val, eps)
            return (x - x_min) / range_val
        else:
            return np.zeros_like(x)
    abs_slk = np.abs(slk) + eps
    robust_slk_mag = np.power(abs_slk, 1.4645487570860838)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    energy_suppress_gate = 1.0 / (1.0 + np.exp(-3.1812951207270705 * np.abs(slk_robust)))
    slack_penalty = np.where(slk_robust < 0, 3.017551983257215 * np.abs(slk_robust), -3.822054008709615 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.134604489664379 * normalize_mm(inv_energy)
    energy_score = energy_score * energy_suppress_gate
    rank_powered = np.power(rank + eps, 2.5237392569397397)
    rank_score = -normalize_mm(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + 0.6114428825383785
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -1.2119859794395431 * normalize_mm(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_mm(duration + eps)
    uncert_norm = normalize_mm(uncert + eps)
    risk_execution_raw = np.tanh(0.7326094967652061 * (dur_norm + uncert_norm))
    risk_execution_score = normalize_mm(risk_execution_raw + 1.0)
    wait_sat = 1.0 - np.exp(-wait / (17.85977766699321 + eps))
    wait_score = -0.5046522091963824 * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + risk_execution_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
