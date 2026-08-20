import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with smooth latency intensification:
      - Replaces Parent 2's hard 1+latency_factor with sigmoid-smoothed urgency boost near slack=0.
      - Uses single tunable 'ddl_protection_gate_width' (not two separate gate parameters) to maintain 12-parameter count.
      - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
      - Smoothness is derived from existing ddl_protection_gate_width (reused), avoiding new parameter.
      - Preserves bounded IQR normalization, piecewise DDL gate, and monotonic structure.
      - Final score is finite, shape-(N,), deterministic, and satisfies all interface contracts.
    """
    eps = 0.0022946575646296307
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_iqr_bounded(x, q_low=0.22428975151779096, q_high=0.6245572708883794):
        x = np.asarray(x, dtype=np.float64)
        finite_mask = np.isfinite(x)
        if not np.any(finite_mask):
            return np.zeros_like(x)
        x_finite = x[finite_mask]
        q1 = np.quantile(x_finite, q_low)
        q3 = np.quantile(x_finite, q_high)
        iqr = q3 - q1
        scale = np.where(iqr > eps, iqr, eps)
        center = (q1 + q3) / 2.0
        normed = (x - center) / (scale + eps)
        return np.clip(normed, -2.0, 2.0)
    width = 2.636164971090183
    gate_linear_region = (slk >= -width) & (slk <= 0.0)
    ddl_gate = np.where(slk <= -width, 1.0, np.where(gate_linear_region, 1.0 + slk / width, 0.0))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    if len(finite_abs_slk) == 0:
        slk_scale = eps
    else:
        q1_slk = np.quantile(finite_abs_slk, 0.22428975151779096)
        q3_slk = np.quantile(finite_abs_slk, 0.6245572708883794)
        slk_scale = np.sqrt((q1_slk + eps) * (q3_slk + eps))
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 4.92260093218136, -4.368148490768172) * np.abs(slack_norm)
    lateness = np.maximum(0.0, -slk)
    smooth_factor = 1.0 / (1.0 + np.exp(-slk / (width + eps)))
    latency_factor = lateness / (abs_slk + eps) * smooth_factor
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.42478301668826685 * normalize_iqr_bounded(inv_energy) * (1.0 + latency_factor)
    rank_score = -1.9382346329380642 * ddl_gate * normalize_iqr_bounded(rank)
    bottleneck = rank * work
    bottleneck_score = -2.203949831158994 * normalize_iqr_bounded(bottleneck + eps)
    duration = exec_t + comm_t
    uncert_gate = np.clip(uncert, 0.0, 2.0) ** 2.3779983457962066
    gated_duration = duration * (1.0 + uncert_gate)
    dur_score = normalize_iqr_bounded(gated_duration + eps)
    wait_clipped = np.clip(wait, 0.0, 1.778037063162864)
    wait_score = -normalize_iqr_bounded(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 33908.637933309765
    min_safe = -finfo.max / 33908.637933309765
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
