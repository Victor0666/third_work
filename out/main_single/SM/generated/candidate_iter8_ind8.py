import numpy as np
RULE_METADATA = {'structure_hash': '3886d643e8c68d98f05fb6038a02f688e271a054a296cd65d54fadb9f33bd23a', 'parameter_schema_hash': '9ea3aae34e8e945dc17b94b3a46fc078ee6263a0b485c1542f14fd2907c73523', 'best_parameter_hash': 'f06f3287d1e3c05df7adf31fe3c706ed62cb9b9ccfeeac7b938530bed1dd7da5', 'best_parameters': {'epsilon': 8.097190852977194e-05, 'slack_risk_penalty': 7.613009259607554, 'slack_urgency_gain': 0.2382473789603707, 'energy_efficiency_weight': 1.2540773856756042, 'criticality_weight': 0.32392344431477593, 'bottleneck_proximity_weight': 3.600811126340399, 'duration_uncertainty_ratio': 1.811822417958868, 'wait_clip_threshold': 18.407818774453794, 'ddl_protection_gate': 0.11979933831513266, 'finfo_max_scale': 195566.96603995468, 'slack_normalization_scale': 0.6321081895503778, 'energy_slack_interaction_weight': 0.9445091149280734}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'ad7db1fbd91717c6bee18ae04714ebd32f327bd19a622dfd4aa8e133ea67d7b0', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Keeps Parent 2's smooth sigmoid DDL gate and bottleneck-proximal term (rank * work)
      - Adds Parent 1's explicit wait-time clipping (more stable than exponential decay)
      - Introduces novel slack-energy interaction: amplifies energy preference under tight slack
      - Uses median-based normalization with explicit finite-filtering for robustness
      - Replaces hard threshold gating with continuous sigmoid modulation across all components
      - All tunable parameters declared; no hidden literals beyond -2..2; deterministic & finite.
    """
    eps = 8.097190852977194e-05
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
        if not np.any(finite_mask):
            return np.zeros_like(x)
        x_finite = x[finite_mask]
        scale = np.median(np.abs(x_finite))
        scale = np.where(scale > eps, scale, eps)
        return x / (scale + eps)
    slack_norm = normalize(slk) * 0.6321081895503778
    ddl_gate = 1.0 / (1.0 + np.exp(0.11979933831513266 * slack_norm))
    slack_penalty = np.where(slk < 0, 7.613009259607554 * np.abs(slack_norm), -0.2382473789603707 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.2540773856756042 * normalize(inv_energy)
    energy_slack_interaction = ddl_gate * energy_score * 0.9445091149280734
    rank_score = -0.32392344431477593 * ddl_gate * normalize(rank + eps)
    bottleneck = rank * work
    bottleneck_score = -3.600811126340399 * normalize(bottleneck + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.811822417958868 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_clipped = np.clip(wait, 0.0, 18.407818774453794)
    wait_score = -normalize(wait_clipped + eps)
    score = slack_penalty + energy_score + energy_slack_interaction + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 195566.96603995468
    min_safe = -finfo.max / 195566.96603995468
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
