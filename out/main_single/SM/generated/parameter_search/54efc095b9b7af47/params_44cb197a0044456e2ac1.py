import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths:
      - Uses robust piecewise linear DDL gate from Parent 2 (monotonic, interpretable, numerically safe).
      - Adds slack-aware energy coupling: prioritizes energy efficiency more strongly for non-urgent tasks (slack >= 0),
        reducing aggressive energy minimization when deadlines are at risk — improves feasibility-energy tradeoff.
      - Retains stabilized rank normalization with explicit floor to avoid degeneracy.
      - Replaces exponential/wait-saturation with clipped linear wait scoring (bounded, deterministic, CMA-ES-friendly).
      - All normalizations use outlier-resilient median scaling with floor guard.
      - No nonlinearities that cause fragility (no sigmoids, softplus, or exp-based saturation); only linear/piecewise logic.
      - Strictly enforces smaller score = higher priority; DDL feasibility dominates via slack_penalty.
    """
    eps = 0.0185825109240689
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x, floor=eps):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            med = np.median(abs_x[finite_mask])
            scale = np.maximum(med, floor)
        else:
            scale = floor
        return x / (scale + eps)
    width = 0.6977664305223384
    gate_linear_region = (slk >= -width) & (slk <= 0.0)
    ddl_gate = np.where(slk <= -width, 1.0, np.where(gate_linear_region, 1.0 + slk / width, 0.0))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    slk_scale = np.median(finite_abs_slk) if len(finite_abs_slk) > 0 else eps
    slk_scale = np.where(slk_scale > eps, slk_scale, eps)
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 1.7698505425190028, -2.991336802613623) * np.abs(slack_norm)
    inv_energy = 1.0 / (energy + eps)
    energy_score_base = -2.3644164653316584 * normalize(inv_energy)
    slack_sign_gate = np.where(slk >= 0, 1.0, 0.0)
    energy_score = energy_score_base * (1.0 + 0.0843113703524387 * slack_sign_gate)
    rank_stable = rank + 0.002124322374936792
    rank_score = -0.03278804440935041 * ddl_gate * normalize(rank_stable)
    bottleneck = rank * work
    bottleneck_score = -2.5501760106424634 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.12411721090724909 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 22.396637815794016)
    wait_score = -normalize(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 8351.84599371848
    min_safe = -finfo.max / 8351.84599371848
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
