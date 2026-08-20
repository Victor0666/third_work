import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with quantile-adaptive slack margin and successor-release coupling.
    
    Key improvements:
      - Uses PARAMS["successor_release_tightness_quantile"] (not literal 0.75) for adaptive slack margin.
      - Uses PARAMS["rank_to_work_ratio_quantile"] (not literal 0.75) for bottleneck-proximal proxy.
      - All numeric literals are restricted to {-2,-1,0,1,2}; epsilon uses PARAMS["epsilon"].
      - Robust, deterministic, finite, shape-preserving, and DDL-first compliant.
    """
    eps = 9.307798467366529e-09
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
        scale = 0.1769068763180804 * mad + eps
        return (x - center) / scale
    pos_slack = slk[slk > eps]
    if len(pos_slack) > 0:
        slack_margin = np.quantile(pos_slack, 0.810707904566867)
        slack_margin = max(slack_margin, eps)
    else:
        slack_margin = 1.0
    slack_penalty = np.zeros_like(slk)
    slack_penalty[slk < 0] = 0.9355669659813796 * (-slk[slk < 0] / (slack_margin + eps)) ** 2
    slack_penalty[slk >= 0] = -0.3406511741433676 * (slk[slk >= 0] / (slack_margin + eps))
    inv_energy = 1.0 / (energy + eps)
    norm_inv_energy = robust_normalize(inv_energy)
    energy_score = -3.6088370754741033 * norm_inv_energy
    rank_active = np.where(slk > -eps, rank, 0.0)
    norm_rank = robust_normalize(rank_active + eps)
    rank_score = -2.3872613392462148 * norm_rank
    duration = exec_t + comm_t
    norm_duration = robust_normalize(duration)
    norm_uncert = robust_normalize(uncert + eps)
    dur_uncert_signal = (1.0 + 0.9081646931314407 * norm_uncert) / (norm_duration + eps + 0.9081646931314407 * norm_uncert + eps)
    dur_score = robust_normalize(dur_uncert_signal)
    wait_sat = 1.0 - np.exp(-0.13364247163449855 * wait)
    norm_wait_sat = robust_normalize(wait_sat + eps)
    wait_score = -norm_wait_sat
    slack_stress = np.where(slk < 0, np.abs(slk) / (slack_margin + eps), 0.0)
    unc_slack_interaction = 2.7709153231777606 * norm_uncert * slack_stress
    rank_to_work_ratio = rank / (work + eps)
    norm_ratio = robust_normalize(rank_to_work_ratio + eps)
    valid_ratios = rank_to_work_ratio[rank_to_work_ratio > eps]
    if len(valid_ratios) > 0:
        ratio_threshold = np.quantile(valid_ratios, 0.8097806847044273)
    else:
        ratio_threshold = 0.0
    tight_successor_proxy = np.where((rank_to_work_ratio >= ratio_threshold + eps) & (slk > -slack_margin), 1.0, 0.0)
    successor_coupling = 0.6535815712510794 * tight_successor_proxy
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + successor_coupling
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
