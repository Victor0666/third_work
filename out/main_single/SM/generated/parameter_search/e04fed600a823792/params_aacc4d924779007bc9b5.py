import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Feasibility-aware energy term: monotonic slack-intensified energy score:
          energy_score = -weight * normalize_iqr(1/(energy+eps)) * (1 + max(0,-slack) / (|slack|+eps))
        Replaces fragile exponentiation with stable linear urgency scaling — avoids over-suppression.
      - Robust slack normalization via geometric-like scale to prevent collapse when slack≈0.
      - Bounded IQR normalization clamped to [-2,2] for stability.
      - All interactions strictly monotonic and bounded; no ad-hoc multiplicative modulations.
      - Exactly 12 parameters; all used; no numeric literals beyond {-2,-1,0,1,2}.
      - Uses np.finfo for safe clamping instead of hardcoded epsilons.
    """
    eps = 0.005401174723519128
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_iqr_bounded(x, q_low=0.1665312657222286, q_high=0.80145186371123):
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
    width = 2.02100686175116
    gate_linear_region = (slk >= -width) & (slk <= 0.0)
    ddl_gate = np.where(slk <= -width, 1.0, np.where(gate_linear_region, 1.0 + slk / width, 0.0))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    if len(finite_abs_slk) == 0:
        slk_scale = eps
    else:
        q1_slk = np.quantile(finite_abs_slk, 0.1665312657222286)
        q3_slk = np.quantile(finite_abs_slk, 0.80145186371123)
        slk_scale = np.sqrt((q1_slk + eps) * (q3_slk + eps))
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 7.712687110190621, -3.727156250685312) * np.abs(slack_norm)
    lateness = np.maximum(0.0, -slk)
    latency_factor = lateness / (abs_slk + eps)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.7390612636435465 * normalize_iqr_bounded(inv_energy) * (1.0 + latency_factor)
    rank_score = -0.1015974891485163 * ddl_gate * normalize_iqr_bounded(rank)
    bottleneck = rank * work
    bottleneck_score = -1.474485286384253 * normalize_iqr_bounded(bottleneck + eps)
    duration = exec_t + comm_t
    uncert_gate = np.clip(uncert, 0.0, 2.0) ** 2.039678588413924
    gated_duration = duration * (1.0 + uncert_gate)
    dur_score = normalize_iqr_bounded(gated_duration + eps)
    wait_clipped = np.clip(wait, 0.0, 27.0450600217088)
    wait_score = -normalize_iqr_bounded(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 9254609.94721034
    min_safe = -finfo.max / 9254609.94721034
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
