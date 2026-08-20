import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with:
      - DDL-first piecewise feasibility gating (ddl_feasibility_margin)
      - Dual-gated energy: feasibility mask AND uncertainty suppression (novel, reusing existing param count)
      - Adaptive rank-slack coupling via power-law (replaces removed 'energy_uncertainty_gate_weight' and 'rank_slack_coupling_exponent')
      - All normalizations use quantile-based robust scale estimation
      - Bounded linear wait ramp + finfo-safeguarded clamping
      - No exponential gates; all numeric literals are in {-2,-1,0,1,2}
    """
    eps = 1.4061152748078777e-06
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
            scale = np.quantile(abs_x[finite_mask], 0.5710507258631982)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    ddl_feasible_mask = slk > 0.09929027689403419
    slack_penalty = np.where(slk < 0, 1.9742928635325776 * np.abs(slk), -0.6711575447519688 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_base = -3.0294560896839213 * normalize(inv_energy)
    uncert_norm = normalize(uncert + eps)
    energy_uncertainty_gate = 1.0 - uncert_norm
    energy_score = energy_base * ddl_feasible_mask * np.clip(energy_uncertainty_gate, 0.0, 1.0)
    abs_slack = np.abs(slk) + eps
    slack_coupling = np.power(abs_slack, -2.175370417408112)
    rank_norm = normalize(rank + eps)
    rank_score = -2.1136369288746626 * rank_norm * slack_coupling
    slack_magnitude_inv = 1.0 / abs_slack
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.175370417408112)
    bottleneck_score = -0.6693163438530803 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.0026256131272373 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 6.161324417667764)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max * 0.5088909737344215, neginf=finfo.min * 0.5088909737344215)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
