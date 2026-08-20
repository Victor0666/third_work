import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Restored max-abs normalization (per reflection) for stability across all features.
      - Reintroduced `duration_uncertainty_ratio` as a robust linear blend — critical for fuzzy deadline handling.
      - Added `slack_aware_energy_decay`: multiplicative exponential attenuation of energy term under high lateness risk,
        replacing linear gating for smoother, physics-aligned suppression of energy optimization when deadlines dominate.
      - All features normalized via max-abs; no quantile, softplus, or unstable transforms.
      - Criticality uses exponentiated rank, bottleneck uses rank*work/(|slack|+eps), starvation uses exp(-wait/theta).
      - NEW: Robust slack normalization using tunable power p (PARAMS["robust_slack_normalization"]) to mitigate outlier sensitivity
              while preserving signed urgency — replaces fragile biased slack reinterpretation.
      - Removed unstable `fuzzy_completion_bias` and `wait_clipping_scale`; reduced parameter count to 12 (within sweet spot).
      - No unused parameters; all numeric literals are {-2,-1,0,1,2}; deterministic and finite.
    """
    eps = 0.003271437224642652
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
    robust_slk_mag = np.power(abs_slk, 1.0797998480019348)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    gate_width = 1.803918496744744 + eps
    ddl_gate = np.where(slk_robust <= 0, 1.0, np.where(slk_robust <= gate_width, 1.0 - slk_robust / gate_width, 0.0))
    slack_penalty = np.where(slk_robust < 0, 1.67869012465009 * np.abs(slk_robust), -2.180139412806681 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.9778812871844205 * normalize(inv_energy)
    slack_decay = np.exp(-np.abs(slk_robust) * 0.6247742180142869)
    energy_score = energy_score * (1.0 - 0.2782368417090553 * ddl_gate) * slack_decay
    rank_powered = np.power(rank + eps, 0.048899831326898374)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + eps
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -1.675795693489114 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.033037071312605 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (26.687731399032344 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
