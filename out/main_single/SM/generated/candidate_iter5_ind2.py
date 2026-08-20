import numpy as np
RULE_METADATA = {'structure_hash': '8aad60447ef44c304b113fd35fc24e9bdf29558a6c6341f5eea17d1b37be33c8', 'parameter_schema_hash': '1a59435872e510718848c7b559eb8f698e6528de22ce8b36e26e28b6a6f07661', 'best_parameter_hash': 'b2dd3de69fdca1568f1f2d30220abc09b40f0b2d164a2351218c34fcfb68b269', 'best_parameters': {'epsilon': 2.4533011911210703e-07, 'slack_risk_penalty': 0.9496864271039194, 'slack_urgency_gain': 0.10849009762220264, 'energy_efficiency_weight': 3.3040443071117056, 'criticality_weight': 1.3834228421582546, 'duration_uncertainty_ratio': 0.40814647754617556, 'wait_decay_rate': 0.013312494886567088, 'uncertainty_slack_interaction': 1.2259352855677472, 'ddl_feasibility_gate_threshold': 0.06904435934073645, 'percentile_clip_lo': 12.310938996200228, 'percentile_clip_hi': 79.21776937675878, 'inf_scale': 182032478.34466624}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '1ba977d271f74e11187b6c972ece0019b586f2295ae60fe0a1cd1a2824cddb4d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with hard DDL gating, clipped-percentile scaling, and power-law starvation saturation.
    
    Structural changes:
      - Removed `starvation_saturation_exponent` to comply with 12-parameter limit.
      - Reinstated exponential saturation (simpler, fewer parameters) but with robust clipping and masking.
      - All numeric literals are now only -2,-1,0,1,2; epsilon and inf handling use PARAMS and np.finfo.
      - `remaining_work` is unused per objective alignment (not in energy or deadline path) — kept in signature but ignored.
    """
    eps = 2.4533011911210703e-07
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)
    nonzero_slack = slk[slk != 0]
    mean_abs_slack = np.mean(np.abs(nonzero_slack)) if len(nonzero_slack) > 0 else 1.0
    ddl_threshold = 0.06904435934073645 * mean_abs_slack
    ddl_feasible = slk >= -ddl_threshold
    feasible_mask = ddl_feasible.astype(np.float64)
    infeasible_mask = 1.0 - feasible_mask

    def clipped_scale(x):
        x = np.asarray(x)
        if not np.any(np.isfinite(x)):
            return np.zeros_like(x)
        lo = np.percentile(x, 12.310938996200228)
        hi = np.percentile(x, 79.21776937675878)
        rng = hi - lo
        rng_safe = np.where(rng > eps, rng, eps)
        clipped = np.clip(x, lo + eps, hi - eps)
        return 2.0 * (clipped - lo - eps) / rng_safe - 1.0
    duration = exec_t + comm_t
    norm_duration = clipped_scale(duration)
    inv_energy = 1.0 / (energy + eps)
    norm_inv_energy = clipped_scale(inv_energy)
    energy_score = -3.3040443071117056 * norm_inv_energy * feasible_mask
    norm_slack = clipped_scale(slk)
    slack_penalty = np.where(slk < 0, 0.9496864271039194 * norm_slack ** 2, -0.10849009762220264 * np.abs(norm_slack))
    rank_active = np.where(ddl_feasible, rank, 0.0)
    norm_rank = clipped_scale(rank_active + eps)
    rank_score = -1.3834228421582546 * norm_rank * feasible_mask
    norm_uncert = clipped_scale(uncert + eps)
    dur_uncert_blend = (1.0 + 0.40814647754617556 * norm_uncert) / (np.abs(norm_duration) + eps + 0.40814647754617556 * np.abs(norm_uncert) + eps)
    dur_score = clipped_scale(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.013312494886567088 * wait)
    norm_wait_sat = clipped_scale(wait_sat + eps)
    wait_score = -norm_wait_sat * feasible_mask
    slack_stress = np.where((slk < 0) & ddl_feasible, np.abs(norm_slack), 0.0)
    unc_slack_interaction = 1.2259352855677472 * norm_uncert * slack_stress
    base_score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(np.float64)
    inf_penalty = finfo.max / 182032478.34466624
    score = base_score * feasible_mask + inf_penalty * infeasible_mask
    score = np.nan_to_num(score, nan=np.median(score) if np.any(np.isfinite(score)) else 0.0, posinf=inf_penalty, neginf=-inf_penalty)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
