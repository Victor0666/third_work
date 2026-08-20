import numpy as np
RULE_METADATA = {'structure_hash': 'e655074af6d5979bb5a4806a604fc699828a85bda251f44bc217bb697b863c35', 'parameter_schema_hash': '03c41473fa94640f0099817297073f3400555983abe86245490b94cd08b8807f', 'best_parameter_hash': '21f07bff2830e10ab5b52cbb2f8c987a90c0ef1ec409cbba625f825f79a5dde1', 'best_parameters': {'epsilon': 0.000365317655059332, 'slack_risk_penalty': 0.5936915754728244, 'slack_urgency_gain': 5.903134788297231, 'energy_efficiency_weight': 1.8282658403842071, 'criticality_weight': 0.13679874773082715, 'duration_uncertainty_ratio': 0.008370155433501785, 'wait_clip_threshold': 20.059542218770794, 'bottleneck_proximity_weight': 0.9794307098201321, 'ddl_protection_gate_width': 1.039694008139709, 'finfo_max_scale': 46125.76465846312, 'upward_rank_normalization_floor': 0.006048121226412684}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '4aa8af93320c6f3f215c7569d8166bf1100db1304a062b7ebaed82a77224bc94', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Bounded piecewise linear DDL gate (monotonic, robust, no numerical fragility)
      - Upward-rank normalization stabilized by floor to preserve ordering under low-criticality degeneracy
      - Removal of unstable nonlinear load-uncertainty term per self-reflection
      - All normalizations use outlier-resilient median scaling with explicit floor guard
      - Strict adherence to feasibility-first: slack_penalty dominates; other terms refine within safe region
      - No hidden constants — only {-2,-1,0,1,2} literals allowed
    """
    eps = 0.000365317655059332
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x, floor=eps):
        x = np.asarray(x)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            med = np.median(abs_x[finite_mask])
            scale = np.maximum(med, floor)
        else:
            scale = floor
        return x / (scale + eps)
    width = 1.039694008139709
    gate_linear_region = (slk >= -width) & (slk <= 0.0)
    ddl_gate = np.where(slk <= -width, 1.0, np.where(gate_linear_region, 1.0 + slk / width, 0.0))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    slk_scale = np.median(finite_abs_slk) if len(finite_abs_slk) > 0 else eps
    slk_scale = np.where(slk_scale > eps, slk_scale, eps)
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 0.5936915754728244, -5.903134788297231) * np.abs(slack_norm)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.8282658403842071 * normalize(inv_energy)
    rank_stable = rank + 0.006048121226412684
    rank_score = -0.13679874773082715 * ddl_gate * normalize(rank_stable)
    bottleneck = rank * work
    bottleneck_score = -0.9794307098201321 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.008370155433501785 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 20.059542218770794)
    wait_score = -normalize(wait_clipped + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 46125.76465846312
    min_safe = -finfo.max / 46125.76465846312
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
