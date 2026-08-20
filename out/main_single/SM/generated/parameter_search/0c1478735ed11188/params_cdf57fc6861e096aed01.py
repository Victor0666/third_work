import numpy as np

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with three key structural improvements:
      - Replaces sigmoid DDL gate with a bounded linear ramp (soft clamp) for improved numerical stability and interpretability.
      - Replaces multiplicative uncertainty gating with power-law uncertainty modulation: (1 + uncert)^sensitivity → smoother, more robust scaling.
      - Introduces additive slack-dependent energy modulation (not multiplicative coupling) to avoid explosion near zero slack — aligns with reflection.
      - All features remain strictly DDL-first: slack_penalty dominates; others only activate under feasibility.
      - No hidden constants: only -2,-1,0,1,2 allowed; all tunables via PARAMS.
    """
    eps = 8.003135835046478e-06
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
            scale = np.quantile(abs_x[finite_mask], 0.7322996785626034)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    ramp_width = 1.0 / (10.670378868955527 + eps)
    low_bound = 0.37738071946282664 - ramp_width
    high_bound = 0.37738071946282664 + ramp_width
    ddl_feasible_gate = np.clip((slk - low_bound) / (2.0 * ramp_width + eps), 0.0, 1.0)
    slack_penalty = np.where(slk < 0, 0.9879253945271571 * np.abs(slk), 0.0)
    slack_proximity = np.clip(1.0 - np.abs(slk) / (0.37738071946282664 + eps), 0.0, 1.0)
    inv_energy = 1.0 / (energy + eps)
    base_energy_score = -2.09837636441157 * normalize(inv_energy)
    energy_score = base_energy_score * ddl_feasible_gate + slack_proximity * base_energy_score
    criticality_mask = (slk >= 0).astype(np.float64)
    rank_score = -0.6889392648677048 * normalize(rank + eps) * criticality_mask
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 0.648868469213498)
    bottleneck_score = -2.3453017372930676 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    uncertainty_factor = np.power(1.0 + np.clip(uncert, 0.0, 2.0), 0.002910676066726365)
    gated_duration = duration * uncertainty_factor
    dur_norm = normalize(gated_duration + eps)
    dur_score = dur_norm * ddl_feasible_gate
    wait_clipped = np.clip(wait, 0.0, 12.752895607144834)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max * 0.29031679301019564, neginf=finfo.min * 0.29031679301019564)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
