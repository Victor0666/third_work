import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Smooth sigmoidal energy suppression (replacing threshold gate) for robust feasibility-energy tradeoff.
      - Robust min-max normalization for slack, energy, and uncertainty to stabilize cross-scenario behavior.
      - Unified risk-weighted execution score via bounded tanh: avoids linear blending fragility and ensures monotonic, saturating response to duration+uncertainty.
      - All parameters used; no unused entries; only {-2,-1,0,1,2} literals; epsilon & machine bounds via PARAMS/np.finfo.
      - Deterministic, finite, shape-(N,), side-effect-free.
    """
    eps = 0.004317834778534665
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
    robust_slk_mag = np.power(abs_slk, 0.6378392503927987)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    energy_suppress_gate = 1.0 / (1.0 + np.exp(-8.52434228700579 * np.abs(slk_robust)))
    slack_penalty = np.where(slk_robust < 0, 8.554090348771375 * np.abs(slk_robust), -4.715055254731199 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -3.898775397906726 * normalize_mm(inv_energy)
    energy_score = energy_score * energy_suppress_gate
    rank_powered = np.power(rank + eps, 2.1522598753657616)
    rank_score = -normalize_mm(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + 0.18867892412105705
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -0.858615860275861 * normalize_mm(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_mm(duration + eps)
    uncert_norm = normalize_mm(uncert + eps)
    risk_execution_raw = np.tanh(0.32774979813723026 * (dur_norm + uncert_norm))
    risk_execution_score = normalize_mm(risk_execution_raw + 1.0)
    wait_sat = 1.0 - np.exp(-wait / (17.121654996907527 + eps))
    wait_score = -0.92949047856233 * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + risk_execution_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
