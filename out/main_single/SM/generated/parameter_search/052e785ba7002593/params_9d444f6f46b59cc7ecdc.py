import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Retains smooth sigmoid ddl_protection_gate (Parent 2) for graded feasibility enforcement
      - Keeps explicit criticality_weight AND bottleneck_proximity_weight (Parent 2) for decoupled control
      - Upgrades normalization from median to adaptive quantile-based scaling (novel improvement)
      - Uses robust quantile (e.g., 75th percentile of abs values) instead of median for better sensitivity
        to right-skewed outlier distributions common in cloud-edge execution times
      - Preserves hard wait-time clipping and bottleneck-proximal term (upward_rank * remaining_work)
      - All features normalized compatibly; no exponential saturations or fragile interactions
      - Strictly enforces DDL-first feasibility via slack_penalty dominating final score
    """
    eps = 0.0009762204059819264
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
            scale = np.quantile(abs_x[finite_mask], 0.8865927057298996)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.14832302088726232 * slack_norm))
    slack_penalty = np.where(slk < 0, 2.380973877529269 * np.abs(slack_norm), -1.9662151494534659 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.6983947631432268 * normalize(inv_energy)
    rank_score = -1.0197763144774088 * ddl_gate * normalize(rank + eps)
    bottleneck = rank * work
    bottleneck_score = -0.2452072725300026 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.2049106375006047 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 74.59674850473101)
    wait_score = -normalize(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 510909.658422063
    min_safe = -finfo.max / 510909.658422063
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
