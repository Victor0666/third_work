import numpy as np
RULE_METADATA = {'structure_hash': '6b8fe71257acd40db19358870d24ac0367195a6221bf550d4e49e4445e025e85', 'parameter_schema_hash': '685f6cb8e77e0bfc5c6ac0abaaea11131ed080551426d3227ad71933c9b4fe76', 'best_parameter_hash': 'eb2a707492c78a8c49b96039c63ebfc34bb80c5d5a3d60c1c57f37afefc05fcc', 'best_parameters': {'epsilon': 2.6206753838173246e-05, 'slack_risk_penalty': 10.60459229544891, 'slack_urgency_gain': 0.10012531753002646, 'energy_efficiency_weight': 0.4911832265557574, 'criticality_weight': 0.6130869514349426, 'bottleneck_proximity_weight': 4.969245737758101, 'wait_decay_rate': 0.03127972640385146, 'uncertainty_slack_interaction': 2.065433291495529, 'ddl_protection_sigmoid_scale': 5.993490244487868, 'z_score_clip_sigma': 1.125140593048866, 'finfo_max_scale': 26511.72130110283, 'successor_release_sharpness': 1.189535167680274}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '7a5a49b3c0f4cee9ce481e0d4a5c29e6d8c92ef87755f2c93b6ceb622d186da0', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining best practices from both parents:
      - Uses Parent 2's robust z-score normalization with ±sigma clipping for discriminability and stability.
      - Adopts Parent 2's soft sigmoid-based DDL protection gating (not hard `np.where`) for graceful attenuation.
      - Integrates Parent 1's successor-release sharpening (`power(1/|slack|, sharpness)`) into bottleneck term.
      - Refines bottleneck activation: requires *both* negative slack *and* non-negligible work *and* applies sharpening.
      - Replaces duration-uncertainty blend with uncertainty-slack-coupled urgency term (from Parent 1), preserving joint-risk capture.
      - Keeps exponential starvation mitigation (Parent 2) but adds bounded ramp fallback for numerical safety.
      - All operations guarded against NaN/inf/zero; uses only allowed literals (-2,-1,0,1,2).
    """
    eps = 2.6206753838173246e-05
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
        finite_mask = np.isfinite(x) & (np.abs(x) > eps)
        if not np.any(finite_mask):
            return np.zeros_like(x)
        x_finite = x[finite_mask]
        mu = np.mean(x_finite)
        std = np.std(x_finite) if len(x_finite) > 1 else eps
        z = (x - mu) / (std + eps)
        clip_bound = 1.125140593048866
        z_clipped = np.clip(z, -clip_bound, clip_bound)
        max_abs = np.maximum(np.max(np.abs(z_clipped)), eps)
        return z_clipped / (max_abs + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(5.993490244487868 * slack_norm))
    slack_penalty = np.where(slk < 0, 10.60459229544891 * np.abs(slack_norm), -0.10012531753002646 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.4911832265557574 * normalize(inv_energy) * ddl_gate
    rank_score = -0.6130869514349426 * normalize(rank) * ddl_gate
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_sharpened = rank * work * np.power(slack_magnitude_inv, 1.189535167680274)
    bottleneck_active = np.where((slk < 0) & (work > eps), bottleneck_sharpened, 0.0)
    bottleneck_score = -4.969245737758101 * normalize(bottleneck_active + eps)
    uncert_norm = normalize(uncert + eps)
    unc_slack_coupling = 2.065433291495529 * uncert_norm * np.where(slack_norm < 0, np.abs(slack_norm), 0.0)
    coupled_score = -unc_slack_coupling
    wait_sat = 1.0 - np.exp(-0.03127972640385146 * wait)
    wait_fallback = np.clip(wait / (0.03127972640385146 + eps), 0.0, 1.0)
    wait_combined = np.where(np.isfinite(wait_sat), wait_sat, wait_fallback)
    wait_score = -normalize(wait_combined + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + coupled_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 26511.72130110283
    min_safe = -finfo.max / 26511.72130110283
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
