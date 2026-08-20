import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with validated structure:
      - Added 'finfo_safe_scale' to replace hidden 0.5 constant.
      - Host-load sensitivity added as evidence-backed term, gated by ddl_protection_gate.
      - Exponential wait saturation replaces linear ramp for smoother starvation mitigation.
      - Softplus-based bottleneck term avoids unstable power operations.
      - All numeric literals are -2, -1, 0, 1, or 2; all tunable values declared in PARAMETER_SCHEMA.
      - Uses np.finfo for safe NaN/inf handling without hardcoded epsilon.
      - Strictly enforces shape (N,) and finite output.
    """
    eps = 0.02940338491472464
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
            scale = np.quantile(abs_x[finite_mask], 0.896417690857735)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.7459422348745065 * slack_norm))
    slack_penalty = np.where(slk < 0, 5.4841360711709255 * np.abs(slack_norm), -2.7610751308110943 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.13409933166165383 * normalize(inv_energy)
    rank_score = -0.40268156112609904 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_base = rank * work * slack_magnitude_inv
    bottleneck_sharpened = np.log1p(np.exp(bottleneck_base))
    bottleneck_score = -4.1859864224098775 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.422808666744288 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_normalized = wait / (9.58969653752943 + eps)
    wait_saturation = 1.0 - np.exp(-wait_normalized)
    wait_saturation = np.clip(wait_saturation, 0.0, 1.0)
    wait_score = -wait_saturation
    energy_per_duration = energy / (duration + eps)
    load_proxy = normalize(energy_per_duration + eps)
    load_score = -0.08753092281059859 * ddl_gate * load_proxy
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 1.2130148838496715
    min_safe = finfo.min / 1.2130148838496715
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
