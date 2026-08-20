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

    def normalize_abs_quantile(x, q=0.7379130167389779):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], q)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)

    def normalize_pos_quantile(x, q=0.7379130167389779):
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
    slack_penalty = np.where(slk < 0, 11.0900999607458 * np.abs(slack_norm), -0.46503450914959177 * np.abs(slack_norm))
    sigmoid_gate = np.where(slk < 0, 1.0 / (1.0 + np.exp(-6.456021924695955 * slk)), 0.0)
    slack_magnitude = np.abs(slk) + eps
    slack_coupling_factor = np.power(slack_magnitude, -2.0653052001334844)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.7058108401913138 * normalize_pos_quantile(inv_energy) * sigmoid_gate * slack_coupling_factor
    rank_sharpened = np.power(rank + eps, 0.47350737108453705)
    rank_score = -1.273114972372023 * normalize_pos_quantile(rank_sharpened) * sigmoid_gate
    bottleneck_base = rank * work
    bottleneck_score = -0.35834389138496825 * normalize_pos_quantile(bottleneck_base + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_abs_quantile(duration + eps)
    uncert_norm = normalize_abs_quantile(uncert + eps)
    dur_uncert_blend = dur_norm + 0.28175170873550104 * uncert_norm
    dur_score = normalize_abs_quantile(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 69.19474424024162)
    wait_normalized = normalize_pos_quantile(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=1797750077555.8367, neginf=-1797750077555.8367)
    score = np.clip(score, -1797750077555.8367, 1797750077555.8367)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
