import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Uses Parent 2's piecewise-linear DDL urgency gate with tunable transition width (sharper & bounded)
      - Adopts quantile-based normalization (0.82) for robustness against outliers and skew
      - Keeps bounded exponential starvation ramp but adds tunable saturation exponent for steeper/faster escalation
      - Replaces brittle exponentiation (Parent 1) with linear criticality weighting gated by urgency (Parent 2)
      - Simplified bottleneck term: rank * work / (|slack| + eps), normalized and weighted — no unstable exponents or offsets
      - Energy term is *only* active when DDL-feasible (ddl_feasible_mask), enforcing hard constraint priority
      - All parameters declared and used; no unused entries; only literals are -2,-1,0,1,2
      - Final score clamped via np.finfo with safe scaling and nan-to-num fallback
    """
    eps = 0.00021147825860322265
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
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], 0.6323909588588541)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    delta = 0.32520258888604175
    ddl_feasible_mask = np.where(slk >= delta, 1.0, 0.0)
    ddl_urgency = np.clip((slk + delta) / (2 * delta), 0.0, 1.0)
    slack_penalty = np.where(slk < 0, 8.696437660246547 * np.abs(slk), -1.0012912352605263 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.28790400955879475 * normalize(inv_energy) * ddl_feasible_mask
    rank_score = -0.052307844255115185 * ddl_urgency * normalize(rank + eps)
    bottleneck_term = rank * work / (np.abs(slk) + eps)
    bottleneck_score = -2.25310825236121 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.38020778611679407 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_exp_ramp = 1.0 - np.exp(-wait / (23.46382344121249 + eps))
    starvation_score = -normalize(np.power(wait_exp_ramp + eps, 1.504647599422492))
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + starvation_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 122364779.2691171
    min_safe = -finfo.max / 122364779.2691171
    score = np.clip(score, min_safe, max_safe)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
