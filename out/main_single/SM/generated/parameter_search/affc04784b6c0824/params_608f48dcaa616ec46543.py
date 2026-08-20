import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Retained Parent 2's proven successor-release interaction and finfo_max_scale stability
      - Reintroduced *light* dynamic energy suppression via ddl_gate^exponent (simpler than Parent 1's sensitivity coupling)
      - Removed host_load_sensitivity_exponent (redundant with ddl_gate exponentiation) to stay within 12 parameters
      - Keeps bounded linear wait ramp and quantile-based robust normalization
      - Slack penalty remains dominant to enforce hard deadline feasibility first
      - All numeric literals are in {-2,-1,0,1,2}; eps and bounds handled via PARAMS or np.finfo
    """
    eps = 0.0002476346173231419
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
            scale = np.quantile(abs_x[finite_mask], 0.6979168188451428)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.12117326339345852 * slack_norm))
    slack_penalty = np.where(slk < 0, 3.6406846791411165 * np.abs(slack_norm), -0.7608257576722031 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.261849390699722 * normalize(inv_energy) * np.power(ddl_gate, 1.0)
    rank_score = -0.8508063971344499 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 3.1075005324364167)
    bottleneck_score = -1.4069416121469962 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.4347728481649386 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 1.000102352907051)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 944114.5110493685
    min_safe = -finfo.max / 944114.5110493685
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
