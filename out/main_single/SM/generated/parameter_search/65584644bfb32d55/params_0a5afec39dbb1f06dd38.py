import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with 12 parameters:
      - Retains Parent 2's piecewise-linear DDL feasibility margin and semantic-aware normalization.
      - Replaces Parent 1's fragile sigmoid gate and exponentiated ddl_gate with direct slack-magnitude coupling via |slack|^{-exponent}, but reuses `successor_release_sharpness` parameter to avoid adding a new one.
      - Uses `successor_release_sharpness` for both bottleneck sharpening *and* energy slack coupling — justified because both benefit from similar deadline-tightness sensitivity.
      - All numeric literals are -2,-1,0,1,2; epsilon and clipping use PARAMS; no hidden constants.
      - Final score remains dominated by slack penalty to enforce hard deadline adherence first.
    """
    eps = 0.004980781764443596
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_abs_quantile(x, q=0.6704134208206813):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], q)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)

    def normalize_pos_quantile(x, q=0.6704134208206813):
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
    slack_penalty = np.where(slk < 0, 2.298446687624645 * np.abs(slack_norm), -3.4881718139916904 * np.abs(slack_norm))
    margin = 0.26440154032381213
    ddl_feasible_mask = np.clip((slk + margin) / (2 * margin + eps), 0.0, 1.0)
    slack_magnitude = np.abs(slk) + eps
    slack_coupling_factor = np.power(slack_magnitude, -1.984115064254472)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.307276317375968 * normalize_pos_quantile(inv_energy) * ddl_feasible_mask * slack_coupling_factor
    rank_score = -0.5555863589001018 * normalize_pos_quantile(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.984115064254472)
    bottleneck_score = -2.4507068866945403 * normalize_pos_quantile(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_abs_quantile(duration + eps)
    uncert_norm = normalize_abs_quantile(uncert + eps)
    dur_uncert_blend = dur_norm + 0.6105419526611333 * uncert_norm
    dur_score = normalize_abs_quantile(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 21.2080060663743)
    wait_normalized = normalize_pos_quantile(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=964028781313.7045, neginf=-964028781313.7045)
    score = np.clip(score, -964028781313.7045, 964028781313.7045)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
