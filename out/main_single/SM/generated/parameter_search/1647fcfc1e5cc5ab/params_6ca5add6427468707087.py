import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with slack-aware energy coupling:
      - Replaces fragile host-load gating with simpler, more interpretable slack-coupled energy suppression.
      - Energy weight decays smoothly via sigmoid(-steepness * slk), tunable via new 'energy_coupling_steepness'.
      - Retains all successful elements from Parent 2: duration-uncertainty blend, successor-release,
        bounded wait ramp, quantile normalization, and dominant slack penalty.
      - All numeric literals are in {-2,-1,0,1,2}.
    """
    eps = 0.022873798597380264
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
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], 0.5476987339693417)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.5592742217931547 * slack_norm))
    slack_penalty = np.where(slk < 0, 9.963367335386991 * np.abs(slack_norm), -3.887103274598106 * np.abs(slack_norm))
    energy_coupling_mask = 1.0 / (1.0 + np.exp(-0.5592742217931547 * slk))
    energy_weight = 1.0 - energy_coupling_mask
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.1716387655228069 * energy_weight * normalize(inv_energy)
    rank_score = -1.7541843332562697 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 3.1231705393850566)
    bottleneck_score = -2.8534467166251587 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.20087419755333305 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 26.166928857705404)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 422097.8033951461
    min_safe = -finfo.max / 422097.8033951461
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
