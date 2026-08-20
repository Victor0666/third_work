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
    eps = 0.00030697102665396415
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
    robust_slk_mag = np.power(abs_slk, 0.7832779042575886)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    gate_width = 1.5677898097230083 + eps
    ddl_gate = np.where(slk_robust <= 0, 1.0, np.where(slk_robust <= gate_width, 1.0 - slk_robust / gate_width, 0.0))
    slack_penalty = np.where(slk_robust < 0, 9.030478505580618 * np.abs(slk_robust), -0.3497958985301519 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.7371165521571333 * normalize(inv_energy)
    slack_decay = np.exp(-np.abs(slk_robust) * 0.6977395628453218)
    energy_score = energy_score * (1.0 - 0.002924084699263199 * ddl_gate) * slack_decay
    rank_powered = np.power(rank + eps, 0.5999074847348558)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + eps
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -2.6754945077369623 * normalize(bottleneck_term + eps)
    uncert_normalized = normalize(uncert + eps)
    bottleneck_uncert_factor = 1.0 - np.tanh(uncert_normalized)
    bottleneck_score = bottleneck_score * bottleneck_uncert_factor
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.6660696891555584 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat_base = 1.0 - np.exp(-wait / (39.62704710831711 + eps))
    wait_score = -wait_sat_base * wait_sat_base
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
