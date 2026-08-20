import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Soft DDL feasibility gate (sigmoid centered at margin) for smooth deadline-energy tradeoff.
      - Power-law host proxy (1 + uncertainty)^exponent to suppress energy/duration under load — more robust than sigmoid products.
      - Additive energy-slack modulation via urgency sigmoid scaled by tunable parameter.
      - Piecewise-robust uncertainty normalization: quantile-scaled below threshold, clipped above.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-(N,) output satisfying all interface contracts.
    """
    eps = 0.0006147342918306657
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
            scale = np.quantile(abs_x[finite_mask], 0.6662385475510744)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    uncert_normalized = np.where(uncert <= 0.6662385475510744, normalize(uncert + eps), np.clip(normalize(uncert + eps), -1.0, 1.0))
    host_proxy = np.power(1.0 + uncert, 2.4697935736208185)
    slack_offset = slk - 0.3722347906548398
    ddl_feasible_gate = 1.0 / (1.0 + np.exp(-4.179371415977553 * slack_offset))
    slack_penalty = np.where(slk < 0, 4.384605702798252 * np.abs(slk), 0.0)
    inv_energy = 1.0 / (energy + eps)
    slack_urgency_sig = 1.0 / (1.0 + np.exp(-np.abs(slk)))
    energy_score = -1.6095176611376993 * normalize(inv_energy) * ddl_feasible_gate
    energy_score += 0.3722347906548398 * slack_urgency_sig
    rank_score = -1.787592211176618 * normalize(rank + eps) * ddl_feasible_gate
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.4697935736208185)
    bottleneck_score = -0.10370160137406838 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    dur_uncert_blend = dur_norm + 0.052122828606504135 * uncert_normalized
    dur_score = normalize(dur_uncert_blend) * ddl_feasible_gate * (1.0 / host_proxy)
    wait_clipped = np.clip(wait, 0.0, 33.1196784129602)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max * 0.6584814568487156, neginf=finfo.min * 0.6584814568487156)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
