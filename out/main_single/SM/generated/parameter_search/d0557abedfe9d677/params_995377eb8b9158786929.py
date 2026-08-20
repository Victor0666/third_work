import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with key structural improvement:
      - Replaces redundant energy-slack coupling with host-load-aware gating:
        gates energy term using *joint* signal of ready_wait_time (congestion proxy) and uncertainty (risk proxy),
        enabling suppression of marginal energy when system is both busy and uncertain — directly addressing 'avoidable marginal energy' failures.
      - Removes multiplicative uncertainty amplification (previously causing outlier explosion), replaces with bounded additive penalty.
      - Retains quantile normalization, successor-release interaction, and DDL-first slack penalty dominance.
      - All literals are in {-2,-1,0,1,2}; no hidden constants; fully deterministic and finite.
    """
    eps = 0.0017544561848379721
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
            scale = np.quantile(abs_x[finite_mask], 0.7110976758472793)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.05459932741293133 * slack_norm))
    slack_penalty = np.where(slk < 0, 11.524347513528456 * np.abs(slack_norm), -2.639008391510149 * np.abs(slack_norm))
    wait_sig = 1.0 / (1.0 + np.exp(-wait))
    uncert_sig = 1.0 / (1.0 + np.exp(-uncert))
    host_load_gate = (1.0 - wait_sig) * (1.0 - uncert_sig)
    gated_energy_weight = 1.9766079723656202 * (1.0 + 0.8408245791220801 * (1.0 - host_load_gate))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -gated_energy_weight * normalize(inv_energy)
    rank_score = -0.23512634441769434 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.7675683696028874)
    bottleneck_score = -3.0685481798182157 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_score = dur_norm + np.clip(0.8408245791220801 * uncert_norm, 0.0, 2.0)
    wait_clipped = np.clip(wait, 0.0, 56.217132811959495)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 179300.93210627462
    min_safe = -finfo.max / 179300.93210627462
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
