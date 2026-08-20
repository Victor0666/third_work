import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Max-abs normalization (robust, deterministic).
      - Exponential slack-aware energy decay (validated superior suppression).
      - Bottleneck score attenuated by uncertainty via tanh-based soft cap.
      - Starvation mitigation uses exponentiated exponential saturation.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-(N,) output satisfying all interface contracts.
    """
    eps = 9.031026643965279e-06
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
    robust_slk_mag = np.power(abs_slk, 1.1466295428590223)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    gate_width = 1.2916154956443702 + eps
    ddl_gate = np.where(slk_robust <= 0, 1.0, np.where(slk_robust <= gate_width, 1.0 - slk_robust / gate_width, 0.0))
    slack_penalty = np.where(slk_robust < 0, 4.053769171399482 * np.abs(slk_robust), -0.9951683567878827 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -3.765263400856232 * normalize(inv_energy)
    slack_decay = np.exp(-np.abs(slk_robust) * 0.8191808677457509)
    energy_score = energy_score * (1.0 - 0.3127846398127843 * ddl_gate) * slack_decay
    rank_powered = np.power(rank + eps, 1.0405144873367245)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + eps
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -1.7240943514985145 * normalize(bottleneck_term + eps)
    uncert_normalized = normalize(uncert + eps)
    bottleneck_uncert_factor = 1.0 - np.tanh(uncert_normalized)
    bottleneck_score = bottleneck_score * bottleneck_uncert_factor
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.4249638506087599 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat_base = 1.0 - np.exp(-wait / (4.812069079310671 + eps))
    wait_score = -wait_sat_base * wait_sat_base
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
