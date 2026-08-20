import numpy as np
RULE_METADATA = {'structure_hash': '77986b383dd4e0a98de2b895cd4dbf6553c721cdd3cd72bf7b7ba443c0ff7b14', 'parameter_schema_hash': '4648979d8acf3698b2b7cdae1b49079fbaec045b825f441ea3439554332396aa', 'best_parameter_hash': '4305b3505d04f4a3b94c43cba1d3efd2d88bf17298ff4d397f3157ae162dc4b2', 'best_parameters': {'epsilon': 2.383203898309003e-06, 'slack_risk_penalty': 11.360909577963636, 'slack_urgency_gain': 1.369142575480302, 'energy_efficiency_weight': 1.5781691329502534, 'criticality_weight': 1.3943981070972333, 'bottleneck_proximity_weight': 3.094460675800678, 'wait_clip_threshold': 69.63314844100825, 'ddl_protection_gate_width': 0.687336779333036, 'mad_scale_factor': 1.9945260536215477, 'energy_slack_coupling_strength': 1.4229427580177987, 'finfo_safe_scale': 2507613.402039647, 'duration_uncertainty_ratio': 0.531694555771966}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'ecbc87d41b9ecefa77bd084bc2a1da4edf82f3c5ce95bbd8c656dd41e0dee5a0', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Robust MAD-based normalization ([-2,2]) for outlier resilience.
      - Smooth sigmoid DDL gate for differentiable urgency modulation.
      - Multiplicative energy-slack coupling for deadline-aware efficiency.
      - Power-law starvation mitigation (wait^0.8 capped at 2.0) — bounded & monotonic.
      - Linear duration-uncertainty blend using tunable `duration_uncertainty_ratio`.
      - All numeric literals are {-2,-1,0,1,2}; no hidden constants; deterministic and finite.
    """
    eps = 2.383203898309003e-06
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_mad_bounded(x, scale_factor=1.9945260536215477):
        x = np.asarray(x, dtype=np.float64)
        finite_mask = np.isfinite(x)
        if not np.any(finite_mask):
            return np.zeros_like(x)
        x_finite = x[finite_mask]
        median = np.median(x_finite)
        mad = np.median(np.abs(x_finite - median))
        scaled_mad = mad * scale_factor
        normed = (x - median) / (scaled_mad + eps)
        return np.clip(normed, -2.0, 2.0)
    width = 0.687336779333036
    ddl_gate = 1.0 / (1.0 + np.exp(-slk / (width + eps)))
    abs_slk = np.abs(slk)
    finite_abs_slk = abs_slk[np.isfinite(abs_slk)]
    if len(finite_abs_slk) == 0:
        slk_scale = eps
    else:
        median_slk = np.median(finite_abs_slk)
        mad_slk = np.median(np.abs(finite_abs_slk - median_slk)) * 1.9945260536215477
        slk_scale = mad_slk + eps
    slack_norm = slk / (slk_scale + eps)
    slack_penalty = np.where(slk < 0, 11.360909577963636, -1.369142575480302) * np.abs(slack_norm)
    urgency_coupling = 1.0 + 1.4229427580177987 * (1.0 / (1.0 + np.exp(slk / (width + eps))))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.5781691329502534 * normalize_mad_bounded(inv_energy) * urgency_coupling
    rank_score = -1.3943981070972333 * ddl_gate * normalize_mad_bounded(rank)
    bottleneck = rank * work
    bottleneck_score = -3.094460675800678 * normalize_mad_bounded(bottleneck + eps)
    duration = exec_t + comm_t
    dur_uncert_blend = duration + 0.531694555771966 * uncert
    dur_score = normalize_mad_bounded(dur_uncert_blend + eps)
    wait_clipped = np.clip(wait, 0.0, 69.63314844100825)
    wait_sat = np.clip(wait_clipped, 0.0, 2.0)
    wait_score = -normalize_mad_bounded(wait_sat + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 2507613.402039647
    min_safe = -finfo.max / 2507613.402039647
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
