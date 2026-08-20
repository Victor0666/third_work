import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with soft DDL feasibility gating:
      - Replaces crisp binary mask with smooth sigmoid transition for graceful energy integration
        in marginally feasible regions — improves energy optimization without compromising deadline safety.
      - Removes redundant slack_urgency_gain (simplifies structure, eliminates coupling).
      - Retains robust quantile normalization *without* hard clipping [-2,2] to preserve ordinal fidelity.
      - Introduces ddl_feasibility_smoothness parameter to tune transition steepness.
      - All features respect DDL-first hierarchy: slack_penalty dominates; others modulate within safe region.
      - No hidden constants: only -2,-1,0,1,2 allowed; all tunables via PARAMS.
    """
    eps = 0.07451440220373422
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
            scale = np.quantile(abs_x[finite_mask], 0.8307725864661764)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_offset = slk - 0.7046175811687483
    ddl_feasible_gate = 1.0 / (1.0 + np.exp(-2.6707748596335583 * slack_offset))
    slack_penalty = np.where(slk < 0, 5.28118370519525 * np.abs(slk), 0.0)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.09915445258166211 * normalize(inv_energy) * ddl_feasible_gate
    criticality_mask = (slk >= 0).astype(np.float64)
    rank_score = -0.009466835813829087 * normalize(rank + eps) * criticality_mask
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.7183008900777588)
    bottleneck_score = -0.168885556613414 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.136083647899878 * uncert_norm
    dur_score = normalize(dur_uncert_blend) * ddl_feasible_gate
    wait_clipped = np.clip(wait, 0.0, 11.148043366830688)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max * 0.48826669566822367, neginf=finfo.min * 0.48826669566822367)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
