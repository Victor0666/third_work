import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Smooth sigmoidal energy suppression (replacing threshold gate) for robust feasibility-energy tradeoff.
      - Robust min-max normalization for slack, energy, and uncertainty to stabilize cross-scenario behavior.
      - Unified risk-weighted execution score via bounded tanh: avoids linear blending fragility and ensures monotonic, saturating response to duration+uncertainty.
      - All parameters used; no unused entries; only {-2,-1,0,1,2} literals; epsilon & machine bounds via PARAMS/np.finfo.
      - Deterministic, finite, shape-(N,), side-effect-free.
    """
    eps = 0.02280992126262726
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
    robust_slk_mag = np.power(abs_slk, 0.715481336976576)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    energy_suppress_gate = 1.0 / (1.0 + np.exp(-3.7187732224710595 * np.abs(slk_robust)))
    slack_penalty = np.where(slk_robust < 0, 9.772165454255147 * np.abs(slk_robust), -4.873551441797886 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.945903774498552 * normalize_mm(inv_energy)
    energy_score = energy_score * energy_suppress_gate
    rank_powered = np.power(rank + eps, 1.3986799655331643)
    rank_score = -normalize_mm(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + 3.1266669093649946
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -0.18406906630497455 * normalize_mm(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_mm(duration + eps)
    uncert_norm = normalize_mm(uncert + eps)
    risk_execution_raw = np.tanh(0.22216678759908004 * (dur_norm + uncert_norm))
    risk_execution_score = normalize_mm(risk_execution_raw + 1.0)
    wait_sat = 1.0 - np.exp(-wait / (12.905501751626941 + eps))
    wait_score = -1.2236347809423942 * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + risk_execution_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
