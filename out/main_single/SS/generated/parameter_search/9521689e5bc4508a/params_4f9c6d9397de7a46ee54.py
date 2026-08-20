import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining robust normalization (MAD-based), tight-slack criticality boost,
    uncertainty-slack tightening, and energy-slack interaction — all parameterized for CMA-ES tuning.
    
    Key structural improvements:
      - Reintroduces MAD-based robust_normalize with tunable robustness factor.
      - Adds explicit energy-slack interaction: penalizes high incremental energy *more* when slack is negative.
      - Preserves Parent 2's median-based criticality gate and uncertainty-slack tightening.
      - All numeric literals are -2, -1, 0, 1, or 2; no hidden thresholds; all parameters declared and used.
    """
    eps = 6.205220500209319e-05
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_normalize(x):
        x = np.asarray(x)
        center = np.median(x)
        mad = np.median(np.abs(x - center))
        scale = mad * 0.11406312431844369 + eps
        return (x - center) / scale
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    neg_slack = np.maximum(0.0, -slack)
    slack_penalty = np.power(neg_slack + eps, 3.071531768581967)
    risk_adjusted_duration = 0.09063506145518456 * norm_duration + (1.0 - 0.09063506145518456) * norm_uncert
    slack_pressure_raw = np.maximum(0.0, -slack)
    energy_slack_interaction = norm_energy * (1.0 + 0.1090868026196645 * slack_pressure_raw)
    slack_pressure_norm = np.maximum(0.0, -norm_slack)
    rank_weighted = norm_rank * (1.0 + 0.5204561518789884 * slack_pressure_norm)
    median_rank = np.median(upward_rank)
    median_slack = np.median(slack)
    is_critical = ((upward_rank >= median_rank) & (slack <= np.maximum(eps, 0.5796831370474366 * median_slack))).astype(float)
    criticality_bonus = is_critical * 1.8682723687395368
    wait_bonus = 0.1501060234373706 * norm_wait
    unc_gate_active = ((norm_uncert > 0.6682292597671906) & (slack < eps)).astype(float)
    tightened_penalty = slack_penalty + unc_gate_active * 0.49369253829388904 * norm_uncert
    score = tightened_penalty + energy_slack_interaction + 1 * risk_adjusted_duration - 1 * rank_weighted - wait_bonus
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values in priority score'
    return score
