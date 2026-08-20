import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining Parent 1's smooth sigmoid gating with Parent 2's robust quantile normalization and bottleneck modeling.
    
    Key improvements:
    - Replaces hard gate with smooth sigmoid using steepness/center (from Parent 1) for differentiable, stable deadline protection
    - Keeps Parent 2's quantile-based normalization (0.75) for outlier resilience
    - Retains softplus-transformed bottleneck term (rank * work * (1+uncert) / |slack|) for critical path sensitivity without explosion
    - Integrates host-load proxy (energy/duration) gated by the new smooth sigmoid for resource-aware energy efficiency
    - Uses unified slack-driven gating across all components (not just criticality), enabling coordinated behavior under deadline pressure
    - All operations guarded against NaN/inf via np.nan_to_num and finfo bounds
    """
    eps = 0.09746180622295358
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
            scale = np.quantile(abs_x[finite_mask], 0.7074915688758296)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    gate_input = 3.355040539554391 * (0.1046239285158209 - slack_norm)
    gate_input_clipped = np.clip(gate_input, -np.log(np.finfo(float).max), np.log(np.finfo(float).max))
    ddl_gate = 1.0 / (1.0 + np.exp(-gate_input_clipped))
    slack_penalty = np.where(slk < 0, 3.6760947416976997 * np.abs(slack_norm), -2.828711772228521 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.4057496052754652 * normalize(inv_energy)
    rank_score = -0.1974177431472771 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_base = rank * work * (1.0 + uncert) * slack_magnitude_inv
    bottleneck_sharpened = np.log1p(np.exp(bottleneck_base))
    bottleneck_score = -3.5629591581790296 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.6876166683726912 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_normalized = wait / (22.268406266271462 + eps)
    wait_saturation = 1.0 - np.exp(-wait_normalized)
    wait_saturation = np.clip(wait_saturation, 0.0, 1.0)
    wait_score = -wait_saturation
    energy_per_duration = energy / (duration + eps)
    load_proxy = normalize(energy_per_duration + eps)
    load_score = -0.4729291391694934 * ddl_gate * load_proxy
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 2.0
    min_safe = finfo.min / 2.0
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
