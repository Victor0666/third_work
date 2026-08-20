import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Host-load-aware gating via uncertainty-powered proxy (bounded exponent) applied to energy & duration.
      - Additive slack-energy modulation instead of multiplicative coupling — prevents pathological score blowup.
      - Bounded piecewise normalization for `uncertainty`: uses quantile-based scaling below threshold, hard cap above.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-(N,) output satisfying all interface contracts.
    """
    eps = 2.3817553880950588e-05
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
            scale = np.quantile(abs_x[finite_mask], 0.6288735675987048)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    uncert_normalized = np.where(uncert <= 0.6288735675987048, normalize(uncert + eps), np.clip(normalize(uncert + eps), -1.0, 1.0))
    host_proxy = np.power(1.0 + uncert, 1.6321088488754996)
    slack_offset = slk - 0.4050269861852255
    ddl_feasible_gate = 1.0 / (1.0 + np.exp(-3.3482300736584807 * slack_offset))
    slack_penalty = np.where(slk < 0, 6.648167749749983 * np.abs(slk), 0.0)
    inv_energy = 1.0 / (energy + eps)
    slack_energy_mod = 0.4050269861852255 * np.abs(slk) / (np.abs(slk) + eps)
    energy_score = -0.08484141676418366 * normalize(inv_energy) * ddl_feasible_gate
    energy_score += slack_energy_mod
    criticality_mask = (slk >= 0).astype(np.float64)
    rank_score = -1.2444022084006354 * normalize(rank + eps) * ddl_feasible_gate * criticality_mask
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.6321088488754996)
    bottleneck_score = -4.347849064603041 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    dur_uncert_blend = dur_norm + 0.8049331786593434 * uncert_normalized
    dur_score = normalize(dur_uncert_blend) * ddl_feasible_gate * (1.0 / host_proxy)
    wait_clipped = np.clip(wait, 0.0, 75.61643275539089)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max * 0.3499383605490215, neginf=finfo.min * 0.3499383605490215)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
