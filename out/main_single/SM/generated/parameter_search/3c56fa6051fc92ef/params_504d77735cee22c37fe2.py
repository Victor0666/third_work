import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with hard DDL protection gate and clipped-percentile scaling.
    
    Key structural improvements:
      - Hard feasibility gate: masks non-feasible tasks (slack < threshold) before any scoring → ensures DDL-first policy.
      - Clipped-percentile scaling replaces median/MAD: preserves urgency ordering near deadline boundaries while suppressing outlier skew.
      - All components computed only on DDL-feasible subset; infeasible tasks receive max-penalty baseline.
      - No successor-risk coupling (removed per reflection), reducing complexity and aligning with feasibility-first principle.
      - All operations numerically guarded; deterministic, finite, and shape-compliant.
    """
    eps = 4.777413997209414e-08
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)
    nonzero_slack = slk[slk != 0]
    mean_abs_slack = np.mean(np.abs(nonzero_slack)) if len(nonzero_slack) > 0 else 1.0
    ddl_threshold = 0.08833421295029936 * mean_abs_slack
    ddl_feasible = slk >= -ddl_threshold
    feasible_mask = ddl_feasible.astype(np.float64)
    infeasible_mask = 1.0 - feasible_mask

    def clipped_scale(x):
        x = np.asarray(x)
        lo = np.percentile(x, 11.534957574946677) if np.any(np.isfinite(x)) else 0.0
        hi = np.percentile(x, 86.15668736254798) if np.any(np.isfinite(x)) else 1.0
        clipped = np.clip(x, lo + eps, hi - eps)
        scale = hi - lo + 2 * eps
        return 2.0 * (clipped - lo - eps) / scale - 1.0
    duration = exec_t + comm_t
    norm_duration = clipped_scale(duration)
    inv_energy = 1.0 / (energy + eps)
    norm_inv_energy = clipped_scale(inv_energy)
    energy_score = -1.1503734927859899 * norm_inv_energy
    norm_slack = clipped_scale(slk)
    slack_penalty = np.where(slk < 0, 7.434343942520893 * norm_slack ** 2, -1.214953277322862 * np.abs(norm_slack))
    rank_score = np.zeros_like(rank)
    rank_active = np.where(ddl_feasible, rank, 0.0)
    norm_rank = clipped_scale(rank_active + eps)
    rank_score = -1.3754675013632098 * norm_rank * feasible_mask
    norm_uncert = clipped_scale(uncert + eps)
    dur_uncert_blend = (1.0 + 1.2663928989041473 * norm_uncert) / (np.abs(norm_duration) + eps + 1.2663928989041473 * np.abs(norm_uncert) + eps)
    dur_score = clipped_scale(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.3389205980143272 * wait)
    norm_wait_sat = clipped_scale(wait_sat + eps)
    wait_score = -norm_wait_sat * feasible_mask
    slack_stress = np.where((slk < 0) & ddl_feasible, np.abs(norm_slack), 0.0)
    unc_slack_interaction = 2.531343486841555 * norm_uncert * slack_stress
    base_score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(np.float64)
    inf_penalty = finfo.max / 477984.31388689135
    score = base_score * feasible_mask + inf_penalty * infeasible_mask
    score = np.nan_to_num(score, nan=np.median(score) if np.any(np.isfinite(score)) else 0.0, posinf=inf_penalty, neginf=-inf_penalty)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
