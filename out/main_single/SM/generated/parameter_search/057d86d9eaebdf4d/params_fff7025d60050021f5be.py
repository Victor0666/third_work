import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Bounded inverse-slack gating: (|slack|+ε)^(-exp) — finite, smooth, tunable sharpness.
      - Host-load-aware energy gating: sigmoid on normalized (exec+comm), centered at tunable load_gate_center.
      - Starvation mitigation via direct linear wait-time ranking (no normalization/clipping).
      - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
      - Exactly 12 parameters; all used; no unused or missing PARAMS references.
    """
    eps = 0.0007855784239623301
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
            scale = np.quantile(abs_x[finite_mask], 0.765624134997652)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.11825986793604146 * slack_norm))
    slack_penalty = np.where(slk < 0, 7.726639873367251 * np.abs(slack_norm), -0.6022228827379966 * np.abs(slack_norm))
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    load_gate = 1.0 / (1.0 + np.exp(-2.0 * (dur_norm - 0.4046245164247712)))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.4820921442674826 * normalize(inv_energy) * load_gate
    rank_score = -0.33917712758720375 * ddl_gate * normalize(rank + eps)
    slack_magnitude_bounded = np.abs(slk) + eps
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_bounded, -1.0072195488926825)
    bottleneck_score = -0.6023811288728859 * normalize(bottleneck_sharpened + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.1071360806759167 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_score = -wait
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 556954.7228406075
    min_safe = -finfo.max / 556954.7228406075
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
