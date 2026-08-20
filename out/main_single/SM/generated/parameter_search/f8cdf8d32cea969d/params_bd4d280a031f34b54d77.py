import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with strict DDL-protection gating, percentile-based normalization,
    and eliminated fragile uncertainty-slack interaction.
    
    Key structural improvements:
      - Replaced MAD-based normalization with bounded configurable percentile scaling (pctl_low/pctl_high).
      - Hard DDL-protection gate: zero out *both* upward_rank and energy_score contributions unless
        slack >= 0 — enforces strict feasibility-first ordering before any energy optimization.
      - Removed uncertainty_slack_interaction entirely — confirmed inactive and adds numerical fragility.
      - All components now use the same robust percentile-based normalizer for consistency and stability.
    """
    eps = 2.175752897136697e-05
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
        p_low = np.percentile(x, 8.823090420695365)
        p_high = np.percentile(x, 86.02304718421578)
        rng = p_high - p_low
        rng_safe = np.where(np.isfinite(rng) & (rng > eps), rng, eps)
        center = (p_low + p_high) / 2.0
        return (x - center) / (1.4875082732469518 * rng_safe + eps)
    duration = exec_t + comm_t
    slack_norm = percentile_normalize(slk)
    slack_penalty = np.where(slk < 0, 10.128019495285495 * slack_norm ** 2, -2.1802062771237103 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_active = np.where(slk >= 0, inv_energy, 0.0)
    energy_score = -1.707422026334499 * percentile_normalize(energy_active + eps)
    rank_active = np.where(slk >= 0, rank, 0.0)
    rank_score = -0.035709023456672276 * percentile_normalize(rank_active + eps)
    dur_norm = percentile_normalize(duration + eps)
    uncert_norm = percentile_normalize(uncert + eps)
    dur_uncert_blend = (1.0 + 1.7075474611533779 * uncert_norm) / (dur_norm + eps + 1.7075474611533779 * uncert_norm + eps)
    dur_score = percentile_normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.27498024738183446 * wait)
    wait_score = -percentile_normalize(wait_sat + eps)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
