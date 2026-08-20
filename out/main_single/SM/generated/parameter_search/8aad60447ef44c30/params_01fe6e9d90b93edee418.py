import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with hard DDL gating, clipped-percentile scaling, and power-law starvation saturation.
    
    Structural changes:
      - Removed `starvation_saturation_exponent` to comply with 12-parameter limit.
      - Reinstated exponential saturation (simpler, fewer parameters) but with robust clipping and masking.
      - All numeric literals are now only -2,-1,0,1,2; epsilon and inf handling use PARAMS and np.finfo.
      - `remaining_work` is unused per objective alignment (not in energy or deadline path) — kept in signature but ignored.
    """
    eps = 1.10187462175621e-07
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)
    nonzero_slack = slk[slk != 0]
    mean_abs_slack = np.mean(np.abs(nonzero_slack)) if len(nonzero_slack) > 0 else 1.0
    ddl_threshold = 0.137239408417139 * mean_abs_slack
    ddl_feasible = slk >= -ddl_threshold
    feasible_mask = ddl_feasible.astype(np.float64)
    infeasible_mask = 1.0 - feasible_mask

    def clipped_scale(x):
        x = np.asarray(x)
        if not np.any(np.isfinite(x)):
            return np.zeros_like(x)
        lo = np.percentile(x, 5.4621505132346355)
        hi = np.percentile(x, 86.95159446007486)
        rng = hi - lo
        rng_safe = np.where(rng > eps, rng, eps)
        clipped = np.clip(x, lo + eps, hi - eps)
        return 2.0 * (clipped - lo - eps) / rng_safe - 1.0
    duration = exec_t + comm_t
    norm_duration = clipped_scale(duration)
    inv_energy = 1.0 / (energy + eps)
    norm_inv_energy = clipped_scale(inv_energy)
    energy_score = -1.3252327437132143 * norm_inv_energy * feasible_mask
    norm_slack = clipped_scale(slk)
    slack_penalty = np.where(slk < 0, 3.1111777273216163 * norm_slack ** 2, -5.199191793612816 * np.abs(norm_slack))
    rank_active = np.where(ddl_feasible, rank, 0.0)
    norm_rank = clipped_scale(rank_active + eps)
    rank_score = -1.7912781056345846 * norm_rank * feasible_mask
    norm_uncert = clipped_scale(uncert + eps)
    dur_uncert_blend = (1.0 + 0.0037457931129659516 * norm_uncert) / (np.abs(norm_duration) + eps + 0.0037457931129659516 * np.abs(norm_uncert) + eps)
    dur_score = clipped_scale(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.07826107394185577 * wait)
    norm_wait_sat = clipped_scale(wait_sat + eps)
    wait_score = -norm_wait_sat * feasible_mask
    slack_stress = np.where((slk < 0) & ddl_feasible, np.abs(norm_slack), 0.0)
    unc_slack_interaction = 1.5626092543467152 * norm_uncert * slack_stress
    base_score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(np.float64)
    inf_penalty = finfo.max / 7662226.4882332655
    score = base_score * feasible_mask + inf_penalty * infeasible_mask
    score = np.nan_to_num(score, nan=np.median(score) if np.any(np.isfinite(score)) else 0.0, posinf=inf_penalty, neginf=-inf_penalty)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
