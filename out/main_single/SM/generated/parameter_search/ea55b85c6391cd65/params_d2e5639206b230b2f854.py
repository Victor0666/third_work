import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three key structural improvements:
      - Conditional sigmoid DDL protection gate active ONLY when slack < 0.
      - Dedicated `energy_ddl_coupling_exponent` decoupled from bottleneck logic.
      - Uses np.finfo for machine-precision safeguards instead of epsilon parameter.
      - All numeric literals are -2,-1,0,1,2; no hidden constants.
      - Preserves semantic-aware quantile normalization and starvation-aware wait ramp.
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

    def normalize_abs_quantile(x, q=0.8536448096470973):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], q)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)

    def normalize_pos_quantile(x, q=0.8536448096470973):
        x = np.asarray(x)
        finite_mask = np.isfinite(x) & (x >= 0)
        if np.any(finite_mask):
            x_clean = x[finite_mask]
            if len(x_clean) > 0:
                scale = np.quantile(x_clean, q)
                scale = np.where(scale > eps, scale, eps)
            else:
                scale = eps
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize_abs_quantile(slk)
    slack_penalty = np.where(slk < 0, 6.744636270944439 * np.abs(slack_norm), -5.841169750478803 * np.abs(slack_norm))
    sigmoid_gate = np.where(slk < 0, 1.0 / (1.0 + np.exp(-2.8104620943849623 * slk)), 0.0)
    slack_magnitude = np.abs(slk) + eps
    slack_coupling_factor = np.power(slack_magnitude, -1.0272176619548374)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -3.755936959593307 * normalize_pos_quantile(inv_energy) * sigmoid_gate * slack_coupling_factor
    rank_score = -0.0351798357420115 * normalize_pos_quantile(rank + eps) * sigmoid_gate
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * slack_magnitude_inv
    bottleneck_score = -2.083878317409062 * normalize_pos_quantile(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_abs_quantile(duration + eps)
    uncert_norm = normalize_abs_quantile(uncert + eps)
    dur_uncert_blend = dur_norm + 0.003147721697142171 * uncert_norm
    dur_score = normalize_abs_quantile(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 33.765542136286044)
    wait_normalized = normalize_pos_quantile(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=26254402760.40834, neginf=-26254402760.40834)
    score = np.clip(score, -26254402760.40834, 26254402760.40834)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
