import numpy as np
RULE_METADATA = {'structure_hash': 'f177bcec8d8d07a692ed9bb9f0d4bd4fba5759205e7cbc5ca6f28d0e5704d255', 'parameter_schema_hash': '65e8f80534767504ed662a260745878a613e2702c8dd66720e9550d315f51a67', 'best_parameter_hash': '05c4950c610a966505679f3c792f312bb78704ed19c6eddb0af964bbf1a3412a', 'best_parameters': {'epsilon': 1.0547592895764202e-06, 'slack_risk_penalty': 8.888437025410308, 'slack_urgency_gain': 5.063223743122293, 'energy_efficiency_weight': 2.0892040648967765, 'criticality_weight': 0.2960775355577623, 'bottleneck_proximity_weight': 1.074843829303064, 'duration_uncertainty_ratio': 1.141493319608905, 'wait_ramp_threshold': 3.2139718637415755, 'ddl_protection_gate': 0.37574908035205684, 'finfo_max_scale': 6639.808631387916, 'robust_minmax_quantile': 0.8854986104277948, 'starvation_slack_threshold': 0.37230543822434153}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '596dd1d17baff45dc84aa9031c682dee05823a618feb46c82a4c706d258c527e', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule with:
      - Robust min-max normalization using high quantile for outlier resilience.
      - Sigmoid ddl_protection_gate for smooth feasibility-aware weighting (replaces piecewise).
      - Exponential wait-saturation (1 - exp(-wait/θ)) for monotonic starvation mitigation.
      - Bottleneck term uses fixed exponent 2 (allowed literal) instead of tunable one — reduces parameter count.
      - Energy suppression reuses same sigmoid gate for coherence and simplicity.
      - Starvation guard bypasses non-bottleneck terms when slack is critically low.
      - All numeric literals are in {-2,-1,0,1,2}; no hidden constants.
      - Final score clamped via np.finfo with parameterized scale.
    """
    eps = 1.0547592895764202e-06
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
        finite_mask = np.isfinite(x)
        if np.any(finite_mask):
            x_finite = x[finite_mask]
            q_low = np.quantile(x_finite, 1.0 - 0.8854986104277948)
            q_high = np.quantile(x_finite, 0.8854986104277948)
            scale = np.where(q_high > q_low, q_high - q_low, eps)
            center = np.median(x_finite)
            return (x - center) / (scale + eps)
        else:
            return np.zeros_like(x)
    slack_penalty = np.where(slk < 0, 8.888437025410308 * np.abs(slk), -5.063223743122293 * slk)
    energy_suppression_weight = np.where(slk < 0, 1.0, 1.0 / (1.0 + np.exp(0.37574908035205684 * slk)))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.0892040648967765 * energy_suppression_weight * normalize(inv_energy)
    ddl_gate = energy_suppression_weight
    rank_score = -0.2960775355577623 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 2)
    bottleneck_score = -1.074843829303064 * normalize(bottleneck_sharpened + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.141493319608905 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_exp_ramp = 1.0 - np.exp(-wait / (3.2139718637415755 + eps))
    wait_score = -normalize(wait_exp_ramp + eps)
    starvation_mask = (slk < 0.37230543822434153).astype(np.float64)
    starvation_bypass = bottleneck_score * starvation_mask
    base_score = slack_penalty + energy_score + rank_score + dur_score + wait_score
    score = np.where(starvation_mask > 0, starvation_bypass, base_score)
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 6639.808631387916
    min_safe = -finfo.max / 6639.808631387916
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
