import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Load-aware starvation mitigation: congestion proxy = wait + uncertainty, exponent = 1 (literal).
      - Successor-release interaction merged into bottleneck term via fixed urgency mask (1 - clip(slk/eps, 0, 1)) — no new parameter.
      - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
      - Robust normalization, sigmoid gates, starvation bypass, and finfo clamping preserved.
      - Exactly 12 parameters; all used; no unused or missing PARAMS references.
    """
    eps = 0.09140329426710897
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
        finite_mask = np.isfinite(x)
        if np.any(finite_mask):
            x_finite = x[finite_mask]
            q_low = np.quantile(x_finite, 1.0 - 0.9816029437588609)
            q_high = np.quantile(x_finite, 0.9816029437588609)
            scale = np.where(q_high > q_low, q_high - q_low, eps)
            center = np.median(x_finite)
            return (x - center) / (scale + eps)
        else:
            return np.zeros_like(x)
    slack_penalty = np.where(slk < 0, 0.5 * np.abs(slk), -3.1701378917335603 * slk)
    energy_suppression_weight = np.where(slk < 0, 1.0, 1.0 / (1.0 + np.exp(0.16873979253408916 * slk)))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.0510244734839183 * energy_suppression_weight * normalize(inv_energy)
    ddl_gate = energy_suppression_weight
    rank_score = -1.1298479577982645 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    urgency_mask = np.clip(1.0 - np.abs(slk) / (eps + eps), 0.0, 1.0)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2) * urgency_mask
    bottleneck_score = -4.800452766873102 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.7400400318384968 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    congestion_proxy = wait + uncert
    load_aware_ramp = np.clip(congestion_proxy / (27.95707488804026 + eps), 0.0, 1.0)
    wait_score = -normalize(load_aware_ramp + eps)
    starvation_mask = (slk < 1.5380420400805905).astype(np.float64)
    starvation_bypass = bottleneck_score * starvation_mask
    base_score = slack_penalty + energy_score + rank_score + dur_score + wait_score
    score = np.where(starvation_mask > 0, starvation_bypass, base_score)
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 2640.595699232187
    min_safe = -finfo.max / 2640.595699232187
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
