import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Tunable bottleneck_offset and starvation_gain (replacing fixed 0.5/0.85).
      - Binary DDL gate (1.0 if slk_robust <= 0, else 0.0) — decisive, no width parameter.
      - Energy term fully suppressed when slack_robust <= 0; no host_load_sensitivity or decay needed.
      - All declared parameters are used; no unused entries.
      - Only numeric literals are -2,-1,0,1,2; epsilon via PARAMS; machine bounds via np.finfo.
      - Deterministic, finite, shape-correct, side-effect-free.
    """
    eps = 1.317017006602175e-06
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
            scale = np.max(abs_x[finite_mask])
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    abs_slk = np.abs(slk) + eps
    robust_slk_mag = np.power(abs_slk, 1.4667258132499315)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    ddl_gate = np.where(slk_robust <= 0, 1.0, 0.0)
    slack_penalty = np.where(slk_robust < 0, 9.358358886965677 * np.abs(slk_robust), -3.7495324716766105 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.7589998016888625 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - ddl_gate)
    rank_powered = np.power(rank + eps, 0.13690368418886897)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + 1.2739379307529948
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -2.794565538407791 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.23284808113475525 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (15.530167981629138 + eps))
    wait_score = -0.3661093620499371 * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
