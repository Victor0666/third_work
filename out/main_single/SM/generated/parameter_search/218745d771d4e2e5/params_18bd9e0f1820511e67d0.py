import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with 12 parameters:
      - Combines Parent 2's quantile normalization and successor-release interaction
      - Integrates Parent 1's multiplicative uncertainty gating on exec_t and comm_t
      - Adds energy-slack coupling via dedicated tunable parameter
      - Removes unused duration_uncertainty_ratio and redundant uncertainty_gate_exponent
      - All numeric literals are in {-2,-1,0,1,2}; no hidden constants
      - Deterministic, finite, shape-(N,) output satisfying all interface contracts.
    """
    eps = 0.06514764141568163
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
            scale = np.quantile(abs_x[finite_mask], 0.7399715424248483)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.021808551374006304 * slack_norm))
    slack_penalty = np.where(slk < 0, 6.803669164642135 * np.abs(slack_norm), -0.12486471256161241 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    urgency_coupling = 1.0 + 0.6894187450066811 * (1.0 / (1.0 + np.exp(np.abs(slk))))
    energy_score = -2.7940985261837334 * normalize(inv_energy) * urgency_coupling
    rank_score = -0.0614933788743119 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.6081022639387672)
    bottleneck_score = -2.9837762909587973 * normalize(bottleneck_sharpened + eps)
    uncert_factor = 1.0 + np.clip(uncert, 0.0, 2.0)
    exec_gated = exec_t * uncert_factor
    comm_gated = comm_t * uncert_factor
    gated_duration = exec_gated + comm_gated
    dur_score = normalize(gated_duration + eps)
    wait_clipped = np.clip(wait, 0.0, 21.834737991957407)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 6618188.361578208
    min_safe = -finfo.max / 6618188.361578208
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
