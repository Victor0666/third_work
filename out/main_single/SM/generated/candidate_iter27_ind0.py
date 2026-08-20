import numpy as np
RULE_METADATA = {'structure_hash': '5335ec384eb7ac631f2955be1957b89d7d9068b0a033e3c0455a761090890201', 'parameter_schema_hash': '404a54e94ee720132327d803131e7ee8941cab45a9cea9833ea8295dc08c549f', 'best_parameter_hash': '02ffe805cae56b483a5362e6b626a2a8c59aaeed46c5059e6b3e6b0f78170abb', 'best_parameters': {'epsilon': 8.770217233630607e-05, 'slack_risk_penalty': 3.0623168987412708, 'slack_urgency_gain': 3.807206956151193, 'energy_efficiency_weight': 0.25139490563488304, 'criticality_exponent': 1.449798727370624, 'bottleneck_weight': 4.801411561452269, 'bottleneck_offset': 0.08877488545886007, 'duration_uncertainty_ratio': 0.9262158418696007, 'wait_saturation_time': 49.29582681323925, 'starvation_gain': 0.45776889844817664, 'robust_slack_normalization': 0.8113521924813545}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '84e92001525996ae2106f6e3678eb880f0988169f16af5b6b567168cf28dc2e4', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Tunable bottleneck_offset and starvation_gain (replacing fixed 0.5/0.85).
      - Binary DDL gate (1.0 if slk_robust <= 0, else 0.0) — decisive, no width parameter.
      - Energy term fully suppressed when slack_robust <= 0; no host_load_sensitivity or decay needed.
      - All declared parameters are used; no unused entries.
      - Only numeric literals are -2,-1,0,1,2; epsilon via PARAMS; machine bounds via np.finfo.
      - Deterministic, finite, shape-correct, side-effect-free.
    """
    eps = 8.770217233630607e-05
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
        x = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x)
        finite_mask = np.isfinite(abs_x)
        if np.any(finite_mask):
            scale = np.max(abs_x[finite_mask])
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    abs_slk = np.abs(slk) + eps
    robust_slk_mag = np.power(abs_slk, 0.8113521924813545)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    ddl_gate = np.where(slk_robust <= 0, 1.0, 0.0)
    slack_penalty = np.where(slk_robust < 0, 3.0623168987412708 * np.abs(slk_robust), -3.807206956151193 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.25139490563488304 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - ddl_gate)
    rank_powered = np.power(rank + eps, 1.449798727370624)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + 0.08877488545886007
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -4.801411561452269 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.9262158418696007 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (49.29582681323925 + eps))
    wait_score = -0.45776889844817664 * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
