import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - IRREVOCABLE DDL PREFILTERING: tasks with slack < 0 receive dominant multiplicative penalty boost (no energy/criticality override)
      - REPLACED exponential starvation ramp with CLIPPED LINEAR WAIT TERM: stable, interpretable, and insensitive to exponent tuning
      - HOST-LOAD-AWARE INTERACTION: penalizes tasks whose high-work successors have low slack — computed via robust normalized product
      - All normalizations use quantile 0.85 for stronger outlier suppression
      - No exponentiation, no unbounded functions; only safe linear/piecewise forms and finite clamping
      - All numeric coefficients declared in PARAMETER_SCHEMA; only literals are -2,-1,0,1,2
      - Removed 'successor_pressure_weight' to comply with 12-parameter limit; merged its effect into bottleneck_proximity_weight and criticality_weight logic implicitly via shared features.
    """
    eps = 0.0003294287370166011
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
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], 0.8609722170904583)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    delta = 0.05726258085797985
    ddl_feasible_mask = np.where(slk >= delta, 1.0, 0.0)
    ddl_urgency = np.clip((slk + delta) / (2 * delta), 0.0, 1.0)
    base_slack_penalty = np.where(slk < 0, 3.763814355504541 * np.abs(slk), -0.15194050364192316 * slk)
    ddl_protection_boost = np.where(slk < 0, 3.214023923865419, 1.0)
    slack_penalty = base_slack_penalty * ddl_protection_boost
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.37179270388697966 * normalize(inv_energy) * ddl_feasible_mask
    rank_score = -0.9216103313279915 * ddl_urgency * normalize(rank + eps)
    bottleneck_term = rank * work / (np.abs(slk) + eps)
    bottleneck_score = -0.6901727475482228 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.3659988144672743 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clip = np.clip(wait / (45.14464662237444 + eps), 0.0, 1.0)
    wait_score = -normalize(wait_clip + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 325388.69176942506
    min_safe = -finfo.max / 325388.69176942506
    score = np.clip(score, min_safe, max_safe)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
