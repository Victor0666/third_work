import numpy as np
RULE_METADATA = {'structure_hash': '8da67a6b4de339124cc74a572061527f2d27da75d5b6d230ff178c35ffa2976b', 'parameter_schema_hash': '3f2df70c7826217bef68374a75120c64753bd964d57ba0c5d18db375803663c9', 'best_parameter_hash': '63c17a48107c3d1afe6e2600974fe9eb3cb17c27754485e2409c91ae13faa1ac', 'best_parameters': {'epsilon': 3.831411424820295e-05, 'slack_risk_penalty': 3.1591639739129507, 'slack_urgency_gain': 3.573551111454666, 'energy_efficiency_weight': 1.3063714969211717, 'criticality_weight': 0.2960612467783512, 'bottleneck_proximity_weight': 0.6380397096672836, 'wait_ramp_threshold': 57.68867669622848, 'ddl_protection_gate': 2.149318114798606e-05, 'robust_normalization_quantile': 0.6629332927760905, 'successor_release_sharpness': 0.5117640860209507, 'host_load_interaction_weight': 0.7174724902726594, 'finfo_safety_scale': 3.054274078722407}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'bb8b0739bc8144efb32cdea7bbb977b5c0691ee8c53aa1898d1b27d08e1727e3', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule incorporating evidence-driven structural changes:
      - Added host-load interaction: modulates energy term using uncertainty and slack coupling to avoid over-selecting low-energy tasks under high-risk conditions
      - Replaced static duration-uncertainty blend with uncertainty-slack-coupled urgency term that amplifies priority when both risk and deadline pressure are high
      - Kept all normalization robust (quantile-based), kept ddl_protection_gate and slack_penalty dominant
      - Removed finfo_max_scale (inactive per diagnostics) and replaced hidden 1e-12 with tunable finfo_safety_scale
      - All operations protected against NaN/inf/zero; no unbounded loops or side effects
      - Uses only allowed numeric literals: -2, -1, 0, 1, 2
    """
    eps = 3.831411424820295e-05
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
            scale = np.quantile(abs_x[finite_mask], 0.6629332927760905)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(2.149318114798606e-05 * slack_norm))
    slack_penalty = np.where(slk < 0, 3.1591639739129507 * np.abs(slack_norm), -3.573551111454666 * np.abs(slack_norm))
    slack_magnitude = np.abs(slk) + eps
    unc_slk_coupling = 0.7174724902726594 * normalize(uncert) * (1.0 / slack_magnitude)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.3063714969211717 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - np.clip(unc_slk_coupling, 0.0, 1.0))
    rank_score = -0.2960612467783512 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 0.5117640860209507)
    bottleneck_score = -0.6380397096672836 * normalize(bottleneck_sharpened + eps)
    coupled_urgency = normalize(uncert) * np.abs(slack_norm)
    coupled_score = -0.7174724902726594 * coupled_urgency
    wait_clipped = np.clip(wait, 0.0, 57.68867669622848)
    wait_normalized = normalize(wait_clipped + eps)
    wait_ramp = np.clip(wait_normalized, 0.0, 1.0)
    wait_score = -wait_ramp
    score = slack_penalty + energy_score + rank_score + bottleneck_score + coupled_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / np.power(2.0, 3.054274078722407)
    min_safe = -max_safe
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
