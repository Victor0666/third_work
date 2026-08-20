import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Fixed instability from additive slack-energy coupling: replaced with bounded multiplicative modulation.
      - Robust_slk_norm (0–1) now directly scales energy_score_base — guarantees |modulated| ≤ |base|, eliminating amplification.
      - All features max-abs normalized; no quantile, softplus, or unstable transforms.
      - Criticality, bottleneck, duration-uncertainty, and starvation terms preserved and stabilized.
      - Exactly 12 parameters — all declared, all used, no literals beyond {-2,-1,0,1,2}.
      - Deterministic, finite, shape-correct, and DDL-first safe.
    """
    eps = 0.047971845838020666
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
            scale = np.max(abs_x[finite_mask])
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    abs_slk = np.abs(slk) + eps
    robust_slk_mag = np.power(abs_slk, 1.312075806581289)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    gate_width = 4.921117570454538 + eps
    ddl_gate = np.where(slk_robust <= 0, 1.0, np.where(slk_robust <= gate_width, 1.0 - slk_robust / gate_width, 0.0))
    slack_penalty = np.where(slk_robust < 0, 7.629074167790924 * np.abs(slk_robust), -0.19459760483983318 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score_base = -2.062408611044453 * normalize(inv_energy)
    energy_score = energy_score_base * robust_slk_norm
    slack_decay = np.exp(-np.abs(slk_robust) * 0.23181574989281095)
    energy_score = energy_score * (1.0 - 0.3170109830202092 * ddl_gate) * slack_decay
    rank_powered = np.power(rank + eps, 2.99846855647339)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + eps
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -1.0591384389096583 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    clipped_uncert = np.clip(uncert, 0.0, 2.0)
    duration_fuzzy = duration * np.power(1.0 + clipped_uncert, 1.7654251006262318)
    dur_score = normalize(duration_fuzzy + eps)
    wait_sat = 1.0 - np.exp(-wait / (1.5777815184440847 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
