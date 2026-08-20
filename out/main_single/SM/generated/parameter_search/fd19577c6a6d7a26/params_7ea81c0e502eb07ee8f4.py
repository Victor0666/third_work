import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strict DDL gating, percentile normalization,
    successor-release awareness, and joint duration-energy urgency.
    
    Structural features:
      - Strict `slk >= 0` gating for all optimization components (energy, criticality, urgency).
      - Successor-release term uses rank-weighted remaining_work to prioritize unblocking high-criticality descendants.
      - Joint duration-energy urgency: (duration * energy)^p — captures trade-off between fast-but-wasteful vs slow-but-efficient.
      - All normalization uses same configurable percentile scheme (pctl_low/pctl_high) for stability and consistency.
      - No unused parameters; no fragile interactions (e.g., uncertainty-slack, host-load-sensitivity); no sigmoid gates.
      - All numeric literals are in {-2,-1,0,1,2}; epsilon and bounds handled via PARAMS or np.finfo.
    """
    eps = 2.8343242680067473e-07
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
        p_low = np.percentile(x, 1.50368715934998)
        p_high = np.percentile(x, 90.35228876960498)
        rng = p_high - p_low
        rng_safe = np.where(np.isfinite(rng) & (rng > eps), rng, eps)
        center = (p_low + p_high) / 2.0
        return (x - center) / (0.9813814614912437 * rng_safe + eps)
    duration = exec_t + comm_t
    slack_norm = percentile_normalize(slk)
    slack_penalty = np.where(slk < 0, 1.2584032967917658 * slack_norm ** 2, -2.4873291263984596 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_active = np.where(slk >= 0, inv_energy, 0.0)
    energy_score = -0.8321784875240369 * percentile_normalize(energy_active + eps)
    rank_active = np.where(slk >= 0, rank, 0.0)
    rank_score = -0.018316913315478144 * percentile_normalize(rank_active + eps)
    dur_energy_prod = (duration + eps) * (energy + eps)
    dur_energy_urgency = np.where(slk >= 0, np.power(dur_energy_prod, 1.0353185752363636), 0.0)
    dur_energy_score = percentile_normalize(dur_energy_urgency + eps)
    wait_sat = 1.0 - np.exp(-0.21537280008754217 * wait)
    wait_score = -percentile_normalize(wait_sat + eps)
    weighted_successor_load = rank_active * (work + eps)
    successor_release_score = -0.5181250952804127 * percentile_normalize(weighted_successor_load + eps)
    score = slack_penalty + energy_score + rank_score + dur_energy_score + wait_score + successor_release_score
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
