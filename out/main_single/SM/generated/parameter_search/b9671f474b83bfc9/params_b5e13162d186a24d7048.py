import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three structural improvements:
      - Replaces linear `slack_urgency_gain` with bounded inverse-slack exponentiation for stable urgency scaling.
      - Introduces robust min-max normalization for rank/work (instead of quantile) to improve cross-seed stability.
      - Adds soft feasibility gate: sigmoid(slack) applied to *all non-slack contributions*, gently suppressing them under extreme deadline pressure—preserving DDL dominance without hard gating.
      - Removed `soft_feasibility_gate_slope` to comply with 12-parameter limit; reused `ddl_protection_sigmoid_slope` for soft gate to preserve intent and reduce count.
    """
    eps = np.finfo(np.float64).tiny
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_minmax_pos(x, eps=eps):
        x = np.asarray(x)
        finite_mask = np.isfinite(x) & (x >= 0)
        if np.any(finite_mask):
            x_clean = x[finite_mask]
            x_min = np.min(x_clean)
            x_max = np.max(x_clean)
            range_val = x_max - x_min + eps
        else:
            x_min, range_val = (0.0, eps)
        return (x - x_min) / range_val

    def normalize_abs_quantile(x, q=0.88499266308194):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], q)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize_abs_quantile(slk)
    slack_penalty = np.where(slk < 0, 4.516905085748975 * np.abs(slack_norm), -np.power(np.abs(slk) + eps, -0.7950027856022305))
    sigmoid_gate = np.where(slk < 0, 1.0 / (1.0 + np.exp(-3.1067379695909416 * slk)), 0.0)
    soft_gate = 1.0 / (1.0 + np.exp(-3.1067379695909416 * slk))
    slack_magnitude = np.abs(slk) + eps
    slack_coupling_factor = np.power(slack_magnitude, -2.2127500023356026)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.05262505976016686 * normalize_abs_quantile(inv_energy) * sigmoid_gate * slack_coupling_factor * soft_gate
    rank_sharpened = np.power(rank + eps, 0.6691319773673781)
    rank_norm = normalize_minmax_pos(rank_sharpened)
    rank_score = -1.3144741365349626 * rank_norm * sigmoid_gate * soft_gate
    bottleneck_base = rank * work
    bottleneck_norm = normalize_minmax_pos(bottleneck_base + eps)
    bottleneck_score = -2.9939387403203157 * bottleneck_norm * soft_gate
    duration = exec_t + comm_t
    dur_norm = normalize_abs_quantile(duration + eps)
    uncert_norm = normalize_abs_quantile(uncert + eps)
    dur_uncert_blend = dur_norm + 1.6209082877495282 * uncert_norm
    dur_score = normalize_abs_quantile(dur_uncert_blend) * soft_gate
    wait_clipped = np.clip(wait, 0.0, 15.219129865632244)
    wait_normalized = normalize_abs_quantile(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp * soft_gate
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=41980836270.97368, neginf=-41980836270.97368)
    score = np.clip(score, -41980836270.97368, 41980836270.97368)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
