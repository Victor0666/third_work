import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with key structural changes:
      - Replaced sigmoid DDL gate with piecewise-linear feasibility margin for smoother cross-seed consistency
      - Added explicit ddl_feasible_mask to conditionally activate energy scoring only when slack >= margin
      - Sharpened successor-release interaction using absolute slack magnitude with clamped exponentiation
      - Used bounded exponential saturation for wait-time (1 - exp(-wait/θ)) instead of linear ramp — eliminates flip behavior
      - All normalizations use robust quantile scaling; no mean/median bias
      - Strict DDL-first ordering preserved via dominant slack_penalty and conditional energy activation
    """
    eps = 0.06816830503897688
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
            scale = np.quantile(abs_x[finite_mask], 0.9203543317991258)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_feasibility_margin = 0.40164316952450085
    ddl_feasible_mask = np.where(slk >= -ddl_feasibility_margin, 1.0, 0.0)
    slack_penalty = np.where(slk < 0, 1.3598667684969652 * np.abs(slack_norm), -4.558470733772349 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.5671918663563473 * normalize(inv_energy) * ddl_feasible_mask
    rank_score = -0.6185794357194411 * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    slack_magnitude_inv_clamped = np.clip(slack_magnitude_inv, 0.0, 1.0 / eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv_clamped, 2.716100640522017)
    bottleneck_score = -1.7121800449263842 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.1629474188250362 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_saturation = 1.0 - np.exp(-wait / (3.280510033050595 + eps))
    wait_normalized = normalize(wait_saturation + eps)
    wait_score = -wait_normalized
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 34926064718.696106
    min_safe = -finfo.max / 34926064718.696106
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
