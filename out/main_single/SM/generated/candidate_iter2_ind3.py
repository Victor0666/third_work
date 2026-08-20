import numpy as np
RULE_METADATA = {'structure_hash': '30ea081003e64bc8454a959dbac0650300a564fba6a6c842963c4a66b273ab9a', 'parameter_schema_hash': '7e661ea2c597f4865a7fa6387a57d222f49dd94b899ab3956db5f7e0ec00f826', 'best_parameter_hash': 'd3a0e25118911c5be755501b7fe6cba672ad3ac6e50ed0625d0659e62548b7a1', 'best_parameters': {'epsilon': 1.2330202289594882e-09, 'slack_risk_penalty': 9.871962002362158, 'slack_urgency_gain': 2.99788794273101, 'energy_efficiency_weight': 0.05947866125903391, 'criticality_weight': 0.21995489071190363, 'duration_uncertainty_ratio': 0.8253945541928118, 'wait_decay_rate': 0.07195137372272213, 'uncertainty_slack_interaction': 2.999894265121199, 'fairness_exponent': 1.8712470184993388, 'robust_scale_factor': 1.5421015127129447}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'd6c839deafd165677d54ec5d83f7c6df5548ae85c61d31678d2fe90d63eabd3d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining best practices from both parents:
    - Uses inverse-energy efficiency (Parent 2) for sharper discrimination.
    - Applies exponential wait decay (Parent 2) for bounded starvation mitigation.
    - Integrates robust MAD-based normalization (Parent 1) with tunable scale.
    - Adds fairness exponent to penalize extreme durations non-linearly.
    - Keeps slack-gated criticality and uncertainty-slack interaction (both).
    - All components are finite, deterministic, and shape-strict (N,)."""
    eps = 1.2330202289594882e-09
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def robust_normalize(x):
        x = np.asarray(x)
        center = np.median(x)
        dev = np.abs(x - center)
        scale = np.median(dev) if len(dev) > 0 else 1.0
        scale = max(scale * 1.5421015127129447, eps)
        return (x - center) / (scale + eps)
    duration = exec_t + comm_t
    slack_norm = robust_normalize(slk)
    slack_penalty = np.where(slk < 0, 9.871962002362158 * slack_norm ** 2, -2.99788794273101 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.05947866125903391 * robust_normalize(inv_energy)
    rank_active = np.where(slk > -eps, rank, 0.0)
    rank_score = -0.21995489071190363 * robust_normalize(rank_active + eps)
    dur_norm = robust_normalize(duration + eps)
    uncert_norm = robust_normalize(uncert + eps)
    dur_uncert_blend = (1.0 + 0.8253945541928118 * uncert_norm) / (dur_norm + eps + 0.8253945541928118 * uncert_norm + eps)
    dur_score = robust_normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.07195137372272213 * wait)
    wait_score = -robust_normalize(wait_sat + eps)
    slack_stress = np.where(slk < 0, np.abs(slack_norm), 0.0)
    unc_slack_interaction = 2.999894265121199 * uncert_norm * slack_stress
    norm_dur_dev = robust_normalize(duration)
    fairness_term = np.sign(norm_dur_dev) * np.abs(norm_dur_dev) ** 1.8712470184993388
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + fairness_term
    score = np.nan_to_num(score, nan=np.median(score), posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
