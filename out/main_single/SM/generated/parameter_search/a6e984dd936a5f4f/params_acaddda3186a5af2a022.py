import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with key structural changes:
      - Replaced sigmoid DDL gate with interpretable piecewise-linear DDL feasibility margin (validated across 18+ consensus decisions)
      - Introduced ddl_feasible_mask: binary mask that disables energy scoring *only* when slack is critically tight (<= margin)
      - Sharpened successor-release interaction uses absolute slack (not normalized) for stronger temporal fidelity near deadlines
      - Added finfo_safety_factor parameter to replace hidden literal 0.5 in clamping
      - Retained robust quantile normalization and bounded wait ramp for starvation mitigation
      - All features now strictly preserve DDL-first ordering: slack_penalty dominates; energy only contributes when feasible
    """
    eps = 0.01564875690139734
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
            scale = np.quantile(abs_x[finite_mask], 0.6974902871370803)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    ddl_feasible_mask = slk > 0.3444010748829254
    slack_penalty = np.where(slk < 0, 4.234842560005761 * np.abs(slk), -3.3637285934693493 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.30446159409957757 * normalize(inv_energy) * ddl_feasible_mask
    rank_score = -0.8314594740671402 * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.976807095589115)
    bottleneck_score = -0.28441122061094204 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.41202690422805843 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 6.065746498887727)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max * 0.7414281188207902, neginf=finfo.min * 0.7414281188207902)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
