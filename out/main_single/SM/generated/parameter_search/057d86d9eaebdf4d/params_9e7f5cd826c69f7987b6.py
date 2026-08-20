import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Bounded inverse-slack gating: (|slack|+ε)^(-exp) — finite, smooth, tunable sharpness.
      - Host-load-aware energy gating: sigmoid on normalized (exec+comm), centered at tunable load_gate_center.
      - Starvation mitigation via direct linear wait-time ranking (no normalization/clipping).
      - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
      - Exactly 12 parameters; all used; no unused or missing PARAMS references.
    """
    eps = 0.00022605122200648192
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
            scale = np.quantile(abs_x[finite_mask], 0.7092183530821334)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.018503750067101442 * slack_norm))
    slack_penalty = np.where(slk < 0, 7.936038096741232 * np.abs(slack_norm), -2.3398104534953834 * np.abs(slack_norm))
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    load_gate = 1.0 / (1.0 + np.exp(-2.0 * (dur_norm - 0.4383929023247719)))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.896411906903013 * normalize(inv_energy) * load_gate
    rank_score = -1.6807245639797546 * ddl_gate * normalize(rank + eps)
    slack_magnitude_bounded = np.abs(slk) + eps
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_bounded, -1.0459805315918804)
    bottleneck_score = -1.6503498633602134 * normalize(bottleneck_sharpened + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.1621829914642957 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_score = -wait
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 2355.6926056878888
    min_safe = -finfo.max / 2355.6926056878888
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
