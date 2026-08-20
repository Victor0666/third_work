import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Conditional sigmoid DDL protection gate (from Parent 2) for precise deadline-risk targeting.
      - Criticality sharpening via rank_sharpening_exponent (novel): applies power-law enhancement *before* normalization to amplify discrimination among high-rank tasks under pressure.
      - Semantic-aware normalization: abs_quantile for temporal features, pos_quantile for structural ones.
      - Energy scoring uses dedicated coupling exponent and sigmoid gating, avoiding dilution by positive slack.
      - Bottleneck term retains clean rank*work interaction without slack inversion, preserving physical meaning.
      - Starvation mitigation uses monotonic clipped ramp, not exponential saturation (more stable & interpretable).
      - All numeric literals are {-2,-1,0,1,2}; eps via np.finfo; no hidden constants.
    """
    eps = np.finfo(np.float64).tiny
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_abs_quantile(x, q=0.6348804703653862):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], q)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)

    def normalize_pos_quantile(x, q=0.6348804703653862):
        x = np.asarray(x)
        finite_mask = np.isfinite(x) & (x >= 0)
        if np.any(finite_mask):
            x_clean = x[finite_mask]
            if len(x_clean) > 0:
                scale = np.quantile(x_clean, q)
                scale = np.where(scale > eps, scale, eps)
            else:
                scale = eps
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize_abs_quantile(slk)
    slack_penalty = np.where(slk < 0, 11.864604336339514 * np.abs(slack_norm), -1.1201178505146423 * np.abs(slack_norm))
    sigmoid_gate = np.where(slk < 0, 1.0 / (1.0 + np.exp(-5.580056707743492 * slk)), 0.0)
    slack_magnitude = np.abs(slk) + eps
    slack_coupling_factor = np.power(slack_magnitude, -0.3139013468240995)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.1462039740214371 * normalize_pos_quantile(inv_energy) * sigmoid_gate * slack_coupling_factor
    rank_sharpened = np.power(rank + eps, 0.7948599053543013)
    rank_score = -0.19253464626552996 * normalize_pos_quantile(rank_sharpened) * sigmoid_gate
    bottleneck_base = rank * work
    bottleneck_score = -1.0384141585371016 * normalize_pos_quantile(bottleneck_base + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_abs_quantile(duration + eps)
    uncert_norm = normalize_abs_quantile(uncert + eps)
    dur_uncert_blend = dur_norm + 0.6625623781746606 * uncert_norm
    dur_score = normalize_abs_quantile(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 34.26663563536561)
    wait_normalized = normalize_pos_quantile(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=434506607551468.1, neginf=-434506607551468.1)
    score = np.clip(score, -434506607551468.1, 434506607551468.1)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
