import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with:
      - Hard wait-time clipping instead of exponential decay (more stable & interpretable)
      - Bottleneck-proximal term: upward_rank * remaining_work → identifies high-impact critical-path bottlenecks
      - Smooth sigmoid DDL protection gate → softly suppresses non-urgent tasks without hard thresholds
      - Removed inactive uncertainty-slack interaction per evidence; replaced with robust bottleneck term
      - All normalizations use median-based scaling (more outlier-resilient than mean-abs)
      - Strict adherence to feasibility-first: no energy terms dominate negative-slack tasks
    """
    eps = 0.0005106282728473122
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
        scale = np.median(abs_x[np.isfinite(abs_x)]) if np.any(np.isfinite(abs_x)) else eps
        scale = np.where(scale > eps, scale, eps)
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.3391090293654462 * slack_norm))
    slack_penalty = np.where(slk < 0, 4.3606231247918394 * np.abs(slack_norm), -0.33100769849582046 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.5698902694840433 * normalize(inv_energy)
    rank_score = -1.2243968808855394 * ddl_gate * normalize(rank + eps)
    bottleneck = rank * work
    bottleneck_score = -0.834109684654483 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.3407898526146696 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 38.249501480516265)
    wait_score = -normalize(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 15599.725374161026
    min_safe = -finfo.max / 15599.725374161026
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
