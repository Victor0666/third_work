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

    def normalize_abs_quantile(x, q=0.9241194889634408):
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
    slack_penalty = np.where(slk < 0, 11.917894994730354 * np.abs(slack_norm), -np.power(np.abs(slk) + eps, -1.1297781394869308))
    sigmoid_gate = np.where(slk < 0, 1.0 / (1.0 + np.exp(-0.5234192259486572 * slk)), 0.0)
    soft_gate = 1.0 / (1.0 + np.exp(-0.5234192259486572 * slk))
    slack_magnitude = np.abs(slk) + eps
    slack_coupling_factor = np.power(slack_magnitude, -1.2671362726506012)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.246635096106632 * normalize_abs_quantile(inv_energy) * sigmoid_gate * slack_coupling_factor * soft_gate
    rank_sharpened = np.power(rank + eps, 0.8198630685652948)
    rank_norm = normalize_minmax_pos(rank_sharpened)
    rank_score = -1.438003520099016 * rank_norm * sigmoid_gate * soft_gate
    bottleneck_base = rank * work
    bottleneck_norm = normalize_minmax_pos(bottleneck_base + eps)
    bottleneck_score = -2.270068020295588 * bottleneck_norm * soft_gate
    duration = exec_t + comm_t
    dur_norm = normalize_abs_quantile(duration + eps)
    uncert_norm = normalize_abs_quantile(uncert + eps)
    dur_uncert_blend = dur_norm + 1.0339685322630825 * uncert_norm
    dur_score = normalize_abs_quantile(dur_uncert_blend) * soft_gate
    wait_clipped = np.clip(wait, 0.0, 19.489466864025644)
    wait_normalized = normalize_abs_quantile(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp * soft_gate
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=23093118792562.03, neginf=-23093118792562.03)
    score = np.clip(score, -23093118792562.03, 23093118792562.03)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
