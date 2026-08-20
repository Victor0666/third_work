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
    eps = 2.498355730015663e-06
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
    ddl_gate = 1.0 / (1.0 + np.exp(0.012806189536347531 * slack_norm))
    slack_penalty = np.where(slk < 0, 8.68334877230205 * np.abs(slack_norm), -0.24639148883443276 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.1000417296288008 * normalize(inv_energy)
    rank_score = -0.7985238091473946 * ddl_gate * normalize(rank + eps)
    bottleneck = rank * work
    bottleneck_score = -0.38458781193564173 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.029643587761653017 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 10.114424365765924)
    wait_score = -normalize(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 25356489.784464765
    min_safe = -finfo.max / 25356489.784464765
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
