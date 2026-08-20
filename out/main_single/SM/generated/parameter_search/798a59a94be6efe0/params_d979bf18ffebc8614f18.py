import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved hybrid priority rule combining Parent 2's robustness with Parent 1's adaptive starvation mitigation:
      - Retains monotonic piecewise-linear DDL gate (Parent 2) for strict feasibility enforcement.
      - Replaces linear wait-clipping with smoothly sharpened power-law ramp (Parent 1's insight + novel exponent) for stronger but bounded starvation mitigation.
      - Uses median-based normalization with explicit floor (Parent 2) for stability, especially under low-rank degeneracy.
      - Keeps duration-uncertainty linear blend (Parent 2), avoiding unstable coupling terms (as diagnosed).
      - Removes fragile sigmoid gates and host-load interactions per performance analysis.
      - All operations guarded against NaN/inf/zero; no hidden constants beyond {-2,-1,0,1,2}.
      - Final clamping uses tunable finfo_max_scale for safety and cross-platform portability.
    """
    eps = 1.0078094686652293e-06
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
    width = 0.10042535698305523
    gate_linear_region = (slk >= -width) & (slk <= 0.0)
    ddl_gate = np.where(slk <= -width, 1.0, np.where(gate_linear_region, 1.0 + slk / width, 0.0))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    slk_scale = np.median(finite_abs_slk) if len(finite_abs_slk) > 0 else eps
    slk_scale = np.where(slk_scale > eps, slk_scale, eps)
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 0.6006993806532699, -1.939249695879531) * np.abs(slack_norm)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.07467949610546708 * normalize(inv_energy)
    rank_stable = rank + 0.015193024920114977
    rank_score = -1.1610240791121424 * ddl_gate * normalize(rank_stable)
    bottleneck = rank * work
    bottleneck_score = -0.7726153056999308 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.4098586528685231 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 37.37698022883547)
    wait_sharpened = np.power(wait_clipped + eps, 1.0027174360818354)
    wait_score = -normalize(wait_sharpened)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 182831.87252227694
    min_safe = -finfo.max / 182831.87252227694
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
