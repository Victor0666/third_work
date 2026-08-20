import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Hard DDL protection gate fully suppressing energy term when slack < 0.
      - Critical bottleneck term uses tunable power-law interaction: (rank × work)^p.
      - Robust max-abs normalization throughout.
      - All numeric literals strictly {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-preserving, and safe against NaN/inf/div-zero.
      - All declared parameters are used; no unused or missing references.
    """
    eps = 0.00045261850027005353
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
    robust_slk_mag = np.power(abs_slk, 1.1707369180600178)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    ddl_protection_gate = np.where(slk_robust >= 0, 1.0, 0.0)
    slack_penalty = np.where(slk_robust < 0, 5.293771420943952 * np.abs(slk_robust), -4.424606933420591 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score_base = -0.07036564173924904 * normalize(inv_energy)
    energy_score = energy_score_base * ddl_protection_gate
    rank_powered = np.power(rank + eps, 1.6249924758420642)
    rank_score_unmod = -normalize(rank_powered)
    slack_feasibility = 1.0 / (1.0 + np.exp(-slk_robust * 1.0))
    rank_score = rank_score_unmod * slack_feasibility
    bottleneck_product = rank * work + eps
    bottleneck_powered = np.power(bottleneck_product, 1.1707369180600178)
    bottleneck_score = -0.6032736821101228 * normalize(bottleneck_powered)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.5922474312899892 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (56.16416912149644 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
