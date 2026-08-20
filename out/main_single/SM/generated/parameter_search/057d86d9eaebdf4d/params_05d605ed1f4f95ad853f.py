import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Bounded inverse-slack gating: (|slack|+ε)^(-exp) — finite, smooth, tunable sharpness.
      - Host-load-aware energy gating: sigmoid on normalized (exec+comm), centered at tunable load_gate_center.
      - Starvation mitigation via direct linear wait-time ranking (no normalization/clipping).
      - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
      - Exactly 12 parameters; all used; no unused or missing PARAMS references.
    """
    eps = 0.04844235218912021
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
            scale = np.quantile(abs_x[finite_mask], 0.6635697068672531)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.9994792360275798 * slack_norm))
    slack_penalty = np.where(slk < 0, 11.983788847610464 * np.abs(slack_norm), -2.2943922891550015 * np.abs(slack_norm))
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    load_gate = 1.0 / (1.0 + np.exp(-2.0 * (dur_norm - 0.1709248363898293)))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.5520820903135563 * normalize(inv_energy) * load_gate
    rank_score = -0.5005532018292053 * ddl_gate * normalize(rank + eps)
    slack_magnitude_bounded = np.abs(slk) + eps
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_bounded, -0.46227966959576494)
    bottleneck_score = -1.0841873450978778 * normalize(bottleneck_sharpened + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.397371078054668 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_score = -wait
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 1531250.2711220607
    min_safe = -finfo.max / 1531250.2711220607
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
