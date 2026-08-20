import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with 12 parameters:
      - Keeps Parent 2's smooth sigmoid ddl_protection_gate
      - Adopts Parent 1's bounded exponential starvation mitigation (1 - exp(-wait/θ))
      - Uses unified robust quantile normalization with [-2,2] clipping for signed features
      - Preserves successor-release interaction with sharpened bottleneck term
      - Energy score remains un-gated but is naturally suppressed by dominant slack_penalty
      - All operations are deterministic, finite, and safeguarded
    """
    eps = 0.016251207455189026
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_signed(x):
        x = np.asarray(x)
        finite_mask = np.isfinite(x)
        if np.any(finite_mask):
            scale = np.quantile(np.abs(x[finite_mask]), 0.6337162834364722)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        normed = x / (scale + eps)
        return np.clip(normed, -2.0, 2.0)

    def normalize_positive(x):
        x = np.asarray(x)
        finite_mask = np.isfinite(x) & (x >= 0)
        if np.any(finite_mask):
            scale = np.quantile(x[finite_mask], 0.6337162834364722)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        normed = x / (scale + eps)
        return np.clip(normed, 0.0, 2.0)
    slack_norm = normalize_signed(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.5834155382265748 * slack_norm))
    slack_penalty = np.where(slk < 0, 6.78901013434387 * np.abs(slack_norm), -0.48052535251911066 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.4737519700411235 * normalize_positive(inv_energy)
    rank_score = -1.2730083153693696 * ddl_gate * normalize_positive(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.6401588658939734)
    bottleneck_score = -2.116081449863995 * normalize_positive(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_positive(duration + eps)
    uncert_norm = normalize_positive(uncert + eps)
    dur_uncert_blend = dur_norm + 0.798701775377465 * uncert_norm
    dur_score = normalize_positive(dur_uncert_blend)
    wait_saturation = 1.0 - np.exp(-wait / (9.301405204462663 + eps))
    wait_score = -np.clip(wait_saturation, 0.0, 1.0)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 977889916.8602867
    min_safe = -finfo.max / 977889916.8602867
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
