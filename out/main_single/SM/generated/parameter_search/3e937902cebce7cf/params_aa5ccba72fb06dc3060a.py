import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with:
      - Replaced sigmoid DDL gate with interpretable piecewise-linear urgency transition (sharper, bounded, no asymptotes)
      - Added explicit starvation-aware wait-saturation term using bounded exponential ramp (1 - exp(-wait/θ)) instead of linear clip
      - Unified bottleneck scoring via robust slack-normalized critical-path importance: upward_rank * remaining_work / (|slack| + eps)
      - Energy term now gated by DDL feasibility mask (binary) rather than soft sigmoid, enforcing hard feasibility boundary
      - All normalizations use quantile-based scaling; no mean/median bias
      - Final score clamped to safe finite range using np.finfo directly
    """
    eps = 3.685294596902722e-05
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
            scale = np.quantile(abs_x[finite_mask], 0.885638577278813)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    delta = 0.836726734863784
    ddl_feasible_mask = np.where(slk >= delta, 1.0, 0.0)
    ddl_urgency = np.clip((slk + delta) / (2 * delta), 0.0, 1.0)
    slack_penalty = np.where(slk < 0, 5.0157533387703115 * np.abs(slk), -1.2089828077672047 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.753316973876018 * normalize(inv_energy) * ddl_feasible_mask
    rank_score = -0.006670864735788425 * ddl_urgency * normalize(rank + eps)
    bottleneck_term = rank * work / (np.abs(slk) + eps)
    bottleneck_score = -1.0950384937642648 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.9647014146358734 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_exp_ramp = 1.0 - np.exp(-wait / (20.939568020529666 + eps))
    wait_score = -normalize(wait_exp_ramp + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 121349.01200298996
    min_safe = -finfo.max / 121349.01200298996
    score = np.clip(score, min_safe, max_safe)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
