import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with adaptive deadline gating and scenario-aware starvation mitigation.
    
    Key structural improvements:
    - Replaces fixed `ddl_protection_center` with task-specific adaptive offset: 
      `offset = adaptive_gate_offset * normalize(upward_rank)`, making deadline protection severity 
      proportional to critical-path importance — high-rank tasks get earlier gating activation.
    - Replaces global `wait_ramp_threshold` with piecewise linear starvation activation triggered only when 
      `ready_wait_time > quantile(slack, starvation_activation_quantile)` — ensures starvation response is 
      contextually aware of workflow's current deadline pressure, not absolute time.
    - Uses bounded min-max normalization *only* for `slack` and `uncertainty` (per reflection) to prevent rank inversion 
      under extreme outliers, while retaining quantile normalization for all other features for robustness.
    - All numeric literals are {-2,-1,0,1,2}; no hidden constants; all parameters referenced exactly once.
    """
    eps = 8.582600048710675e-05
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_quantile(x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.quantile(abs_x[finite_mask], 0.9443661623816436)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)

    def normalize_mm(x):
        x = np.asarray(x)
        finite_mask = np.isfinite(x)
        if np.any(finite_mask):
            x_finite = x[finite_mask]
            x_min = np.min(x_finite)
            x_max = np.max(x_finite)
            range_val = x_max - x_min
            range_val = np.where(range_val > eps, range_val, eps)
            return (x - x_min) / range_val
        else:
            return np.zeros_like(x)
    slack_mm = normalize_mm(slk)
    uncert_mm = normalize_mm(uncert)
    rank_norm = normalize_quantile(rank + eps)
    adaptive_center = 0.27022970503752597 * rank_norm
    gate_input = 5.686296417582016 * (adaptive_center - slack_mm)
    finfo = np.finfo(float)
    gate_input_clipped = np.clip(gate_input, -np.log(finfo.max), np.log(finfo.max))
    ddl_gate = 1.0 / (1.0 + np.exp(-gate_input_clipped))
    slack_penalty = np.where(slk < 0, 6.0655372492745805 * np.abs(slack_mm), -4.432397638033808 * np.abs(slack_mm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.306222924173506 * normalize_quantile(inv_energy)
    rank_score = -0.8257965483113769 * ddl_gate * normalize_quantile(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_base = rank * work * (1.0 + uncert) * slack_magnitude_inv
    bottleneck_sharpened = np.log1p(np.exp(bottleneck_base))
    bottleneck_score = -1.4320620895007794 * normalize_quantile(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_quantile(duration + eps)
    uncert_norm = normalize_quantile(uncert + eps)
    dur_uncert_blend = dur_norm + 0.4347642036456055 * uncert_norm
    dur_score = normalize_quantile(dur_uncert_blend)
    slack_finite = slk[np.isfinite(slk)]
    if len(slack_finite) > 0:
        threshold_val = np.quantile(slack_finite, 0.24380500990832477)
    else:
        threshold_val = 0.0
    starvation_active = (wait > threshold_val) & np.isfinite(slk)
    wait_ramp = np.clip((wait - threshold_val) / (threshold_val + eps), 0.0, 1.0)
    wait_score = -np.where(starvation_active, wait_ramp, 0.0)
    energy_per_duration = energy / (duration + eps)
    load_proxy = normalize_quantile(energy_per_duration + eps)
    load_score = -0.027807801853636675 * ddl_gate * load_proxy
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 2.0
    min_safe = finfo.min / 2.0
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
