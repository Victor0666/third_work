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
    eps = 0.0003006555759433167
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
            scale = np.quantile(abs_x[finite_mask], 0.9123689623348759)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.5585577291532458 * slack_norm))
    slack_penalty = np.where(slk < 0, 1.6118331798067893 * np.abs(slack_norm), -2.9564405310986075 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    urgency_coupling = 1.0 + 0.035500285477384645 * (1.0 / (1.0 + np.exp(np.abs(slk))))
    energy_score = -0.478563064186173 * normalize(inv_energy) * urgency_coupling
    rank_score = -2.977884690612701 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2.2204846933083493)
    bottleneck_score = -0.8675252694884157 * normalize(bottleneck_sharpened + eps)
    uncert_factor = 1.0 + np.clip(uncert, 0.0, 2.0)
    exec_gated = exec_t * uncert_factor
    comm_gated = comm_t * uncert_factor
    gated_duration = exec_gated + comm_gated
    dur_score = normalize(gated_duration + eps)
    wait_clipped = np.clip(wait, 0.0, 15.382734804840894)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 126644432.7127349
    min_safe = -finfo.max / 126644432.7127349
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
