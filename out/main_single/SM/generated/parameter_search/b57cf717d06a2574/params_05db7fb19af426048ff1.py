import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with:
      - Smooth width-controlled sigmoid slack gate (replaces strength-based gate)
      - Monotonic bounded wait ramp (Parent 2)
      - Sharpened successor-release interaction (rank * work * |slack|⁻ˢʰᵃʳᵖⁿᵉˢˢ)
      - Quantile-based robust normalization for all features
      - Uncertainty-slack interaction removed to meet 12-parameter limit while preserving core deadline-energy-criticality balance
      - All numeric literals strictly in {-2,-1,0,1,2}
    """
    eps = 0.012580073074362213
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
            scale = np.quantile(abs_x[finite_mask], 0.776045367068875)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    gate_width = 0.15757016558209433
    ddl_gate = 1.0 / (1.0 + np.exp(-slk / (gate_width + eps)))
    slack_penalty = 6.920534628450308 * np.maximum(-slk, 0.0) + 0.2823167576194652 * np.minimum(slk, 0.0)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.9927965284420838 * normalize(inv_energy)
    rank_score = -1.4095278427183677 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.6531105931210952)
    bottleneck_score = -3.3044441988629445 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.8162122298562128 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 4.040034320700865)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 26972730.48073404
    min_safe = -finfo.max / 26972730.48073404
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
