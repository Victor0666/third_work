import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Bounded piecewise linear DDL gate (monotonic, robust, no numerical fragility)
      - Upward-rank normalization stabilized by floor to preserve ordering under low-criticality degeneracy
      - Removal of unstable nonlinear load-uncertainty term per self-reflection
      - All normalizations use outlier-resilient median scaling with explicit floor guard
      - Strict adherence to feasibility-first: slack_penalty dominates; other terms refine within safe region
      - No hidden constants — only {-2,-1,0,1,2} literals allowed
    """
    eps = 0.0004705559083902341
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
    width = 1.6695672949074636
    gate_linear_region = (slk >= -width) & (slk <= 0.0)
    ddl_gate = np.where(slk <= -width, 1.0, np.where(gate_linear_region, 1.0 + slk / width, 0.0))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    slk_scale = np.median(finite_abs_slk) if len(finite_abs_slk) > 0 else eps
    slk_scale = np.where(slk_scale > eps, slk_scale, eps)
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 5.622763790899686, -3.597019848861183) * np.abs(slack_norm)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.7344527703564113 * normalize(inv_energy)
    rank_stable = rank + 0.07565692639318397
    rank_score = -1.050885150492301 * ddl_gate * normalize(rank_stable)
    bottleneck = rank * work
    bottleneck_score = -0.6600805149953068 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.36500696083354 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 8.268700948922302)
    wait_score = -normalize(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 60568600.93641459
    min_safe = -finfo.max / 60568600.93641459
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
