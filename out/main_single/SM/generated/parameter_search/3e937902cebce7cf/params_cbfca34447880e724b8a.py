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
    eps = 0.0028873436477794504
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
            scale = np.quantile(abs_x[finite_mask], 0.8007538923652222)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    delta = 0.4547723092021781
    ddl_feasible_mask = np.where(slk >= delta, 1.0, 0.0)
    ddl_urgency = np.clip((slk + delta) / (2 * delta), 0.0, 1.0)
    slack_penalty = np.where(slk < 0, 2.250977295481536 * np.abs(slk), -0.10287212931908604 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -3.9982560632613398 * normalize(inv_energy) * ddl_feasible_mask
    rank_score = -1.2528601976264548 * ddl_urgency * normalize(rank + eps)
    bottleneck_term = rank * work / (np.abs(slk) + eps)
    bottleneck_score = -2.0291134770617147 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.20843281203335026 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_exp_ramp = 1.0 - np.exp(-wait / (44.997558722533405 + eps))
    wait_score = -normalize(wait_exp_ramp + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 3384.933532325604
    min_safe = -finfo.max / 3384.933532325604
    score = np.clip(score, min_safe, max_safe)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
