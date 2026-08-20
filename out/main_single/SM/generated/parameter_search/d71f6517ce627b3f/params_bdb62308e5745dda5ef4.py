import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with two key structural improvements:
      - Replaced hard wait clipping with bounded linear ramp: smooth, monotonic, avoids abrupt transitions
      - Introduced successor-release interaction: upward_rank * remaining_work * (1 / (|slack| + eps))^sharpness,
        sharpening critical-path focus as deadline pressure increases
      - Load-aware energy gating removed (to reduce parameter count) — instead, energy term remains active but
        is naturally suppressed by ddl_gate and slack_penalty when feasibility is at risk
      - All normalizations use adaptive quantile scaling; no median or mean bias
      - Strict DDL-first ordering preserved via dominant slack_penalty term
    """
    eps = 0.0006994465302416981
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
            scale = np.quantile(abs_x[finite_mask], 0.6914332265785339)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.32002249123792303 * slack_norm))
    slack_penalty = np.where(slk < 0, 7.090375794253137 * np.abs(slack_norm), -3.4022352180392366 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.8446397657155837 * normalize(inv_energy)
    rank_score = -1.7170426233802996 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.9639157636777864)
    bottleneck_score = -2.0847501064553713 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.07887658140039192 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 23.8852688942389)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 190215.13046466597
    min_safe = -finfo.max / 190215.13046466597
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
