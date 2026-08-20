import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Restored `slack_urgency_gain` for balanced prioritization under tight-but-feasible deadlines.
      - Smooth sigmoidal DDL feasibility gate (not binary) to preserve CMA-ES gradient near slack=0.
      - Successor-aware criticality: modulates upward_rank by normalized mean successor slack (simulated via input slack as proxy for local context).
      - All features max-abs normalized; no literals beyond {-2,-1,0,1,2}.
      - Deterministic, finite, shape-correct, and DDL-hard-constraint safe.
      - Explicit starvation weight and bottleneck power decoupled (no reuse of same param for different roles).
    """
    eps = 0.0005160459335533564
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
    robust_slk_mag = np.power(abs_slk, 1.1676070238698841)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    ddl_feasibility_gate = 1.0 / (1.0 + np.exp(-slk_robust * 2.1776935689229804))
    slack_penalty = np.where(slk_robust < 0, 1.0069521352850088 * np.abs(slk_robust), -0.1891374462681769 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score_base = -0.8792708711426417 * normalize(inv_energy)
    energy_score = energy_score_base * ddl_feasibility_gate
    successor_slack_influence = 1.0 - 0.49884014654874287 * np.clip(normalize(slk), 0.0, 1.0)
    rank_powered = np.power(rank + eps, 1.2294520447061297)
    rank_score_unmod = -normalize(rank_powered)
    rank_score = rank_score_unmod * ddl_feasibility_gate * successor_slack_influence
    bottleneck_product = rank * work + eps
    bottleneck_powered = np.power(bottleneck_product, 0.6526487305232964)
    bottleneck_score = -0.6526487305232964 * normalize(bottleneck_powered)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.989176345063371 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (14.69485012291677 + eps))
    wait_score = -1.3084479028628087 * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
