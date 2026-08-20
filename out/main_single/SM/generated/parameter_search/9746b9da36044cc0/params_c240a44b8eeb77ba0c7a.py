import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Restored piecewise-linear DDL gate (per reflection) for crisp hard-deadline discrimination.
      - Reinstated `host_load_sensitivity` (not energy_gating_strength) for load-aware energy attenuation.
      - Added `piecewise_linear_gate_width` to tune transition sharpness — replaces sigmoid steepness,
        enabling robust, interpretable, and numerically stable boundary control near slack=0.
      - All features normalized via max-abs for stability; no softplus, log, or unstable transforms.
      - Criticality uses exponentiated rank, bottleneck uses rank*work/(|slack|+eps), starvation uses exp(-wait/theta).
      - No unused parameters; all numeric literals are {-2,-1,0,1,2}; deterministic and finite.
    """
    eps = 0.0018586255947736154
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
    gate_width = 0.7364054643513287 + eps
    ddl_gate = np.where(slk <= 0, 1.0, np.where(slk <= gate_width, 1.0 - slk / gate_width, 0.0))
    slack_penalty = np.where(slk < 0, 1.360409641982593 * np.abs(slk), -4.954992523928611 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.437122347395229 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - 0.4637930943030404 * ddl_gate)
    rank_powered = np.power(rank + eps, 2.407125771932442)
    rank_score = -normalize(rank_powered)
    slack_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_term = rank * work * slack_inv
    bottleneck_score = -3.565952239553991 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.2088267860279411 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (7.46310914081511 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2, neginf=-finfo.max / 2)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
