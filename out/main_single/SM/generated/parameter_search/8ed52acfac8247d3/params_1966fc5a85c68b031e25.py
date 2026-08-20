import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining robust slack gating (Parent 2) with bottleneck-aware load coupling (Parent 1).
    
    Structural improvements:
      - Retains Parent 2's stable exponential wait saturation and slack-sign gating.
      - Integrates Parent 1's validated bottleneck_boost (upward_rank * remaining_work), but gated *only* on negative slack
        — avoids over-activation during slack-rich phases while targeting critical path bottlenecks under DDL stress.
      - Uses median-based normalization (more outlier-resilient than mean-abs) from Parent 1, preserving stability across heterogeneous workloads.
      - Keeps uncertainty-slack interaction from Parent 2 for direct DDL-risk–energy tradeoff modeling.
      - All components are additive, sign-consistent, and shape-preserving; no hidden state or loops.
    """
    eps = 0.0025508534969556585
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
        med = np.median(abs_x[finite_mask]) if np.any(finite_mask) else 0.0
        scale = med if np.isfinite(med) and med > eps else eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    slack_penalty = np.where(slk < 0, 10.351585867440619 * np.abs(slack_norm), -2.074264329653416 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.4262350936115239 * normalize(inv_energy)
    rank_active = np.where(slack_norm >= 0, rank, 0.0)
    rank_score = -0.9373579481733034 * normalize(rank_active + eps)
    bottleneck_boost = rank * work
    bottleneck_active = np.where(slk < 0, bottleneck_boost, 0.0)
    bottleneck_score = -0.4485248338570108 * normalize(bottleneck_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.5974803232149832 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.4214107941435382 * wait)
    wait_score = -normalize(wait_sat + eps)
    unc_slack_interaction = 0.6651888909815458 * uncert_norm * np.where(slack_norm < 0, np.abs(slack_norm), 0.0)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 356770.16679073963
    min_safe = -finfo.max / 356770.16679073963
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
