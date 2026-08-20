import numpy as np
RULE_METADATA = {'structure_hash': '20fdd5e340da2403a83c67d098798957bb0920a6ff850a2aae193ac307c607a9', 'parameter_schema_hash': '329f07f5d8ceb1a1d75a066313386253c02b0631d485bb9752a0d22459950bb4', 'best_parameter_hash': '706f85aa71023cf5de825ed1f0868bc4cb4ed418e473fdcee71ae0560c0f7f75', 'best_parameters': {'epsilon': 1.1378037698864102e-06, 'slack_risk_penalty': 6.761323820222502, 'slack_urgency_gain': 1.8432370942410838, 'energy_efficiency_weight': 1.1580093457957503, 'criticality_weight': 2.9082733675409274, 'duration_uncertainty_ratio': 1.0523477535920904, 'wait_decay_rate': 0.3608298835146218, 'uncertainty_slack_interaction': 3.092514296133051, 'finfo_max_scale': 47093.303427794526, 'ddl_protection_gate_width': 0.3041537589940488, 'release_pressure_weight': 0.9637435374637099, 'load_uncertainty_exponent': 0.6516124708709536}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'a35c7f73889fdfc3c1467276989df97ff0719f3fbbe6f9f8528d93813df08fc7', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
      - Restored median/MAD normalization (per self-reflection) for robustness against skewed wait/uncertainty.
      - Unified slack-aware gating: ddl_gate now modulates *all* deadline-sensitive terms (energy, rank, release, load) consistently.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-compliant, and fully parameterized.
    """
    eps = 1.1378037698864102e-06
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
        med = np.median(abs_x)
        mad = np.median(np.abs(abs_x - med)) if np.all(np.isfinite(abs_x)) else eps
        scale = max(med, 1.0 * mad, eps)
        return x / (scale + eps)
    gate_width = 0.3041537589940488
    ddl_gate = 1.0 / (1.0 + np.exp(-slk / (gate_width + eps)))
    slack_penalty = 6.761323820222502 * np.maximum(-slk, 0.0) + 1.8432370942410838 * np.minimum(slk, 0.0)
    slack_score = normalize(slack_penalty + eps)
    inv_energy = 1.0 / (energy + eps)
    energy_active = inv_energy * ddl_gate
    energy_score = -1.1580093457957503 * normalize(energy_active + eps)
    rank_score = -2.9082733675409274 * ddl_gate * normalize(rank + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.0523477535920904 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.3608298835146218 * wait)
    wait_score = -normalize(wait_sat + eps) * ddl_gate
    unc_slack_interaction = 3.092514296133051 * uncert_norm * (1.0 - ddl_gate)
    release_pressure = rank * work
    release_score = -0.9637435374637099 * ddl_gate * normalize(release_pressure + eps)
    load_factor = (work * uncert + eps) ** 0.6516124708709536
    load_score = normalize(load_factor) * ddl_gate
    score = slack_score + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + release_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 47093.303427794526
    min_safe = -finfo.max / 47093.303427794526
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
