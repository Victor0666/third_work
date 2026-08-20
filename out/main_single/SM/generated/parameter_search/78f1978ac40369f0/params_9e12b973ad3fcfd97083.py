import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's strict DDL gating and percentile normalization
    with Parent 1's successor-release interaction — now *conditionally activated only under slack<0*.
    
    Structural improvements:
      - Strict hard gate on all non-slack components: energy_score and rank_score only active when slack >= 0.
      - Successor-release term added *only when slack < 0*, acting as a corrective release mechanism for high-risk paths.
      - Wider percentile range (5th–95th) improves robustness across heterogeneous workflow scales.
      - All normalizations use same percentile_normalize() for consistency and stability.
      - Removed all fragile uncertainty-slack couplings and host-load modulations per performance analysis.
      - Uses quadratic slack penalty (Parent 2) + linear urgency gain (Parent 2), but with improved clipping.
    """
    eps = 2.3694993885131674e-07
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def percentile_normalize(x):
        x = np.asarray(x)
        p_low = np.percentile(x, 11.12345464166302)
        p_high = np.percentile(x, 98.76438756308183)
        rng = p_high - p_low
        rng_safe = np.where(np.isfinite(rng) & (rng > eps), rng, eps)
        center = (p_low + p_high) / 2.0
        return (x - center) / (rng_safe + eps)
    duration = exec_t + comm_t
    slack_norm = percentile_normalize(slk)
    slack_penalty = np.where(slk < 0, 6.875185447071576 * np.abs(slack_norm) ** 2, -5.088297110498106 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_active = np.where(slk >= 0, inv_energy, 0.0)
    energy_score = -4.257963836402616 * percentile_normalize(energy_active + eps)
    rank_active = np.where(slk >= 0, rank, 0.0)
    rank_score = -1.336158285302758 * percentile_normalize(rank_active + eps)
    successor_active = np.where(slk < 0, work, 0.0)
    successor_norm = percentile_normalize(successor_active + eps)
    urgency_norm = percentile_normalize(np.clip(-slk, 0.0, np.inf) + eps)
    successor_score = -1.4892701616984334 * successor_norm * urgency_norm
    dur_norm = percentile_normalize(duration + eps)
    uncert_norm = percentile_normalize(uncert + eps)
    denom = dur_norm + 0.033445666231190045 * uncert_norm + eps
    dur_uncert_blend = (1.0 + 0.033445666231190045 * uncert_norm) / denom
    dur_score = percentile_normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.0028687723678885147 * wait)
    wait_score = -percentile_normalize(wait_sat + eps)
    score = slack_penalty + energy_score + rank_score + successor_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 6167079.493739514
    min_safe = -finfo.max / 6167079.493739514
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
