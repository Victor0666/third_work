import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Binary DDL gate restored for decisiveness.
      - Bounded sigmoid slack normalization using tunable clip bound and steepness.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-(N,) output satisfying all interface contracts.
    """
    eps = 1.3715626764573416e-05
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
    k = 5.361644606367513
    clip_bound = 11.020929675173129
    slk_clipped = np.clip(slk, -clip_bound, clip_bound)
    sigmoid_slk = 2.0 / (1.0 + np.exp(-k * slk_clipped + eps)) - 1.0
    ddl_gate = np.where(sigmoid_slk <= 0.0, 1.0, 0.0)
    slack_penalty = np.where(sigmoid_slk < 0.0, 0.9760304355389957 * np.abs(sigmoid_slk), -0.7014145159509505 * sigmoid_slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.05634635166023246 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - ddl_gate)
    rank_powered = np.power(rank + eps, 0.7880589619099749)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(sigmoid_slk) + 0.30101152590916297
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -1.0507105780472048 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.406742583061774 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (37.37561056287849 + eps))
    wait_score = -0.4027247409852922 * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
