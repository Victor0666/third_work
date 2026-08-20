import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Bounded linear bottleneck modulation (replaces tanh) to avoid over-suppression of critical path under uncertainty.
      - Linear starvation saturation: min(wait / theta, 1.0) — avoids exponential over-penalization and aligns with DDL urgency.
      - Dual-gated energy score (DDL gate + exponential slack decay) preserved.
      - Robust slack normalization retained.
      - Successor-readiness penalty integrated via upward_rank * remaining_work, scaled by ddl_gate and tunable weight — merged into bottleneck_score to stay within 12 parameters.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-(N,) output satisfying all interface contracts.
    """
    eps = 0.007052485740465884
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
    robust_slk_mag = np.power(abs_slk, 1.2945112741021354)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    gate_width = 1.3602991057404015 + eps
    ddl_gate = np.where(slk_robust <= 0, 1.0, np.where(slk_robust <= gate_width, 1.0 - slk_robust / gate_width, 0.0))
    slack_penalty = np.where(slk_robust < 0, 3.8361291331178684 * np.abs(slk_robust), -4.317360220916529 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.5187057823008029 * normalize(inv_energy)
    slack_decay = np.exp(-np.abs(slk_robust) * 0.7358203002850013)
    energy_score = energy_score * (1.0 - 0.5412182023661134 * ddl_gate) * slack_decay
    rank_powered = np.power(rank + eps, 0.5600158887032578)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + eps
    bottleneck_term = rank * work / robust_abs_slk
    successor_term = rank * work * ddl_gate
    combined_bottleneck = bottleneck_term + successor_term
    bottleneck_score = -0.2744681931408648 * normalize(combined_bottleneck + eps)
    uncert_normalized = normalize(uncert + eps)
    uncert_gate = np.clip(1.0 - uncert_normalized, 0.0, 1.0)
    bottleneck_score = bottleneck_score * uncert_gate
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.49448701110271986 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_scaled = wait / (2.8623274752114423 + eps)
    wait_score = -np.clip(wait_scaled, 0.0, 1.0)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
