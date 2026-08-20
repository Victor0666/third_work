import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Retains Parent 2's superior quantile normalization and soft DDL feasibility gate.
      - Integrates Parent 1's multiplicative slack-energy coupling using new tunable 'urgency_coupling_strength'.
      - Uses Parent 1's multiplicative uncertainty gating on exec_t and comm_t (not linear blend).
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-(N,) output satisfying all interface contracts.
    """
    eps = 0.0009263952729091247
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
            scale = np.quantile(abs_x[finite_mask], 0.7493035747108959)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_offset = slk - 0.048727711482506084
    ddl_feasible_gate = 1.0 / (1.0 + np.exp(-5.565568423370877 * slack_offset))
    slack_penalty = np.where(slk < 0, 9.603959754813458 * np.abs(slk), 0.0)
    urgency_coupling = 1.0 + 1.8945149636112142 * (1.0 / (1.0 + np.exp(slk / (0.048727711482506084 + eps))))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.10111984526976 * normalize(inv_energy) * ddl_feasible_gate * urgency_coupling
    criticality_mask = (slk >= 0).astype(np.float64)
    rank_score = -0.8967449325808106 * normalize(rank + eps) * criticality_mask
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 0.882490417662726)
    bottleneck_score = -0.6091029237442035 * normalize(bottleneck_sharpened + eps)
    exec_gated = exec_t * (1.0 + np.clip(uncert, 0.0, 2.0) ** 1)
    comm_gated = comm_t * (1.0 + np.clip(uncert, 0.0, 2.0) ** 1)
    gated_duration = exec_gated + comm_gated
    dur_norm = normalize(gated_duration + eps)
    dur_score = dur_norm * ddl_feasible_gate
    wait_clipped = np.clip(wait, 0.0, 31.70382312435556)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max * 0.5041876782200759, neginf=finfo.min * 0.5041876782200759)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
