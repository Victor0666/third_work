import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining robust MAD normalization, adaptive slack margin,
    successor-release coupling, and bottleneck-aware urgency.
    
    Structural improvements vs Parent 2:
      - Removes latency_sensitivity_exponent (reducing parameter count from 13→12).
      - Replaces exponential amplification with *binary bottleneck boost*: tasks above
        PARAMS["bottleneck_ratio_quantile"] of rank-to-work ratio receive fixed coupling strength
        only when slack <= slack_margin — preserving sharp deadline awareness without extra parameter.
      - All numeric literals are {-2,-1,0,1,2}; epsilon uses PARAMS["epsilon"]; no hardcoded quantiles.
      - Unified robust_normalize() ensures stability; DDL-first gating remains strict.
    """
    eps = 2.0619526294088377e-08
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def robust_normalize(x):
        x = np.asarray(x)
        center = np.median(x)
        mad = np.median(np.abs(x - center)) + eps
        scale = 0.27169102315491483 * mad + eps
        return (x - center) / scale
    pos_slack = slk[slk > eps]
    if len(pos_slack) > 0:
        slack_margin = np.quantile(pos_slack, 0.6878064621549179)
        slack_margin = max(slack_margin, eps)
    else:
        slack_margin = 1.0
    slack_raw = np.zeros_like(slk)
    slack_raw[slk < 0] = -slk[slk < 0] / (slack_margin + eps)
    slack_raw[slk >= 0] = slk[slk >= 0] / (slack_margin + eps)
    norm_slack_raw = robust_normalize(slack_raw + eps)
    slack_penalty = np.where(slk < 0, 5.690592614849594 * np.abs(norm_slack_raw) ** 2, -5.318658971932022 * np.abs(norm_slack_raw))
    inv_energy = 1.0 / (energy + eps)
    energy_active = np.where(slk > -eps, inv_energy, 0.0)
    norm_energy = robust_normalize(energy_active + eps)
    energy_score = -1.4954713725249649 * norm_energy
    rank_active = np.where(slk > -eps, rank, 0.0)
    norm_rank = robust_normalize(rank_active + eps)
    rank_score = -0.8249262883739261 * norm_rank
    duration = exec_t + comm_t
    norm_duration = robust_normalize(duration + eps)
    norm_uncert = robust_normalize(uncert + eps)
    dur_uncert_signal = (1.0 + 0.5845240779944859 * norm_uncert) / (norm_duration + eps + 0.5845240779944859 * norm_uncert + eps)
    dur_score = robust_normalize(dur_uncert_signal)
    wait_sat = 1.0 - np.exp(-0.09784755680413801 * wait)
    norm_wait_sat = robust_normalize(wait_sat + eps)
    wait_score = -norm_wait_sat
    slack_stress = np.where(slk < 0, np.abs(slk) / (slack_margin + eps), 0.0)
    unc_slack_interaction = 1.8230029162530414 * norm_uncert * slack_stress
    rank_to_work_ratio = rank / (work + eps)
    valid_ratios = rank_to_work_ratio[rank_to_work_ratio > eps]
    if len(valid_ratios) > 0:
        ratio_threshold = np.quantile(valid_ratios, 0.7431323136608639)
    else:
        ratio_threshold = 0.0
    bottleneck_boost = np.where((rank_to_work_ratio >= ratio_threshold + eps) & (slk <= slack_margin), 1.0, 0.0)
    successor_coupling = 3.968908013715533 * bottleneck_boost
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + successor_coupling
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
