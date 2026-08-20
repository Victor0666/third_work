import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Hard DDL feasibility mask (from Parent 2) for strict deadline-first enforcement.
      - Inverse-slack urgency (from Parent 1) for stable positive-slack scaling — replaces linear gain.
      - Host-load proxy (Parent 2) for congestion-aware energy gating.
      - Bottleneck term uses min(exec_time, comm_time) — robust to dominance shifts, no slack singularity.
      - All normalizations use quantile-based robust scaling (Parent 2).
      - Final score clamping uses machine-precision-aware bounds (Parent 1 style with log-transformed parameter).
      - Removed redundant soft gates and sigmoid ramps: hard mask + inverse urgency provides cleaner control.
    """
    eps = 0.030979985811443116
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
            scale = np.quantile(abs_x[finite_mask], 0.9459301820846217)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slk_norm = normalize(slk)
    rank_norm = normalize(rank + eps)
    exec_norm = normalize(exec_t + eps)
    comm_norm = normalize(comm_t + eps)
    energy_norm = normalize(energy + eps)
    wait_norm = normalize(wait + eps)
    uncert_norm = normalize(uncert + eps)
    slack_penalty = np.where(slk < 0, 6.671565677457392 * np.abs(slk_norm), -np.power(np.abs(slk) + eps, -1.8001739258788694))
    ddl_feasibility_margin = 0.1066008383032565
    ddl_feasible_mask = np.where(slk >= -ddl_feasibility_margin, 1.0, 0.0)
    host_load_proxy = np.power(wait + uncert + eps, 0.7744915946371979)
    host_load_gate = 1.0 / (1.0 + host_load_proxy)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.5087666478644017 * normalize(inv_energy) * ddl_feasible_mask * host_load_gate
    rank_score = -0.9098215137370971 * rank_norm
    min_duration_bound = np.minimum(exec_t, comm_t)
    bottleneck_interaction = rank_norm * normalize(min_duration_bound + eps)
    bottleneck_score = -2.6554942635311107 * normalize(bottleneck_interaction + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    dur_uncert_blend = dur_norm + 0.8039186255775216 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_saturation = 1.0 - np.exp(-wait / (19.316003453688218 + eps))
    wait_score = -normalize(wait_saturation + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = 4707675497.446309
    min_safe = -max_safe
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    score = np.clip(score, min_safe, max_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
