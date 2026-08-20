import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with strict DDL-first enforcement and robust normalization.
    
    Key self-evolution improvements:
      - Replaces median-based normalization with clipped mean±std using tunable outlier_clip_factor.
      - Introduces hard DDL-protection gate: when slack is critically low (< ddl_protection_gate * ref_scale),
        only slack_penalty and energy_score remain active; all other components zero out → enforces feasibility-first.
      - Uses dynamic reference scale for gate threshold: computed from finite slack stats, avoiding NaN on all-negative slack.
      - Retains inverse energy + linear slack penalty but simplifies fairness term to linear duration-uncertainty blend.
      - Removes wait_decay_rate and uncertainty_slack_interaction per reflection — reduces overfitting and improves stability.
      - All operations are bounded, finite, and shape-preserving; no unbounded branches or hidden state.
    """
    eps = 4.3803032962879585e-06
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def robust_normalize(x):
        x = np.asarray(x)
        valid_mask = np.isfinite(x)
        if not np.any(valid_mask):
            return np.zeros_like(x)
        x_clean = x[valid_mask]
        center = np.mean(x_clean)
        scale = np.std(x_clean, ddof=0) + eps
        clip_bound = 2.8730583831029497 * scale
        x_centered = np.clip(x - center, -clip_bound, clip_bound)
        return x_centered / (scale + eps)
    slk_abs_clean = np.abs(slk[np.isfinite(slk)])
    ref_scale = np.mean(slk_abs_clean) if len(slk_abs_clean) > 0 else 1.0
    ref_scale = max(ref_scale, eps)
    ddl_safe_mask = slk >= -0.4282090047783439 * ref_scale
    slack_norm = robust_normalize(slk)
    slack_penalty = np.where(slk < 0, 6.6023726950521 * slack_norm ** 2, -2.030470473398617 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.6135473386136763 * robust_normalize(inv_energy)
    pos_slack_clean = slk[slk > eps]
    slack_margin = np.mean(pos_slack_clean) if len(pos_slack_clean) > 0 else ref_scale
    slack_margin = max(slack_margin, eps)
    rank_activation_threshold = -0.7945726430725201 * slack_margin
    rank_active = np.where(slk > rank_activation_threshold, rank, 0.0)
    rank_score = -0.08245328686485456 * robust_normalize(rank_active + eps)
    rank_score = np.where(ddl_safe_mask, rank_score, 0.0)
    duration = exec_t + comm_t
    dur_norm = robust_normalize(duration + eps)
    uncert_norm = robust_normalize(uncert + eps)
    dur_uncert_blend = (1.0 - 0.6254536798284387) * dur_norm + 0.6254536798284387 * uncert_norm
    dur_score = robust_normalize(dur_uncert_blend)
    dur_score = np.where(ddl_safe_mask, dur_score, 0.0)
    score = slack_penalty + energy_score + rank_score + dur_score
    finfo_max = np.finfo(np.float64).max
    finfo_min = np.finfo(np.float64).min
    score = np.nan_to_num(score, nan=np.median(score) if np.any(np.isfinite(score)) else 0.0, posinf=0.9405283692226304 * finfo_max, neginf=0.9405283692226304 * finfo_min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in final priority score'
    return score
