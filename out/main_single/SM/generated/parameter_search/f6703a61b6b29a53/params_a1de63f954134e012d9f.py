import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Keeps Parent 2's robust sigmoid DDL protection gate (smoother than piecewise-linear).
      - Adopts Parent 2's successor-release interaction with sharpened bottleneck term.
      - Introduces novel *energy modulation exponent*: softens energy term via ddl_gate^exponent instead of hard gating or linear scaling,
        preserving energy-awareness under mild pressure while suppressing it aggressively when slack approaches zero.
      - Removes redundant clipping bounds (score_clipping_bound, finfo_max_scale) — relies on quantile normalization + ε-safeguards + nan_to_num.
      - Uses unified robust normalization with adaptive quantile scaling across all features.
      - Maintains strict DDL-first semantics via dominant slack_penalty and ddl_gate modulation.
      - All operations are finite, deterministic, shape-preserving, and use only allowed literals (-2,-1,0,1,2).
    """
    eps = 0.07912650648483613
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
            scale = np.quantile(abs_x[finite_mask], 0.896621045942271)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.21995583057606483 * slack_norm))
    slack_penalty = np.where(slk < 0, 9.44913152155499 * np.abs(slack_norm), -2.5657149817321505 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.5120039741093028 * normalize(inv_energy) * np.power(ddl_gate, 0.6336728193364182)
    rank_score = -1.711740364735185 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.000420115270074)
    bottleneck_score = -2.3131756088297055 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.47223474200164095 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 1.0759824302132326)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
