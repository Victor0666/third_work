import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with:
       - Bounded linear wait-time clipping instead of exponential saturation (more stable & interpretable)
       - Slack-gated criticality using upward_rank * remaining_work (bottleneck-aware importance)
       - DDL protection gate: smooth activation of criticality only when slack is near zero or negative
       - Robust median-based normalization to resist outliers (replaces mean-abs scaling)
       - Removed uncertainty-slack interaction: diagnosed inactive; replaced with cleaner ddl_protection_gate
       - All divisions guarded; all outputs finite; shape strictly (N,)
    """
    eps = 0.0009003357448706168
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
        scale = np.median(abs_x) if np.all(np.isfinite(abs_x)) and np.median(abs_x) > eps else eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    slack_penalty = np.where(slk < 0, 7.776189705251329 * np.abs(slack_norm), -0.8355779829433373 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.4592710681981787 * normalize(inv_energy)
    abs_slack = np.abs(slk)
    ddl_gate = np.clip(1.0 - abs_slack / (0.10000963273518745 + eps), 0.0, 1.0)
    bottleneck_importance = rank * work
    bottleneck_active = ddl_gate * bottleneck_importance
    bottleneck_score = -0.3882627982546904 * normalize(bottleneck_active + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.8869141120183688 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 0.8255236959047967)
    wait_normalized = normalize(wait_clipped + eps)
    wait_score = -wait_normalized
    score = slack_penalty + energy_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 173080.76585198156
    min_safe = -finfo.max / 173080.76585198156
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
