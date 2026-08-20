import numpy as np
RULE_METADATA = {'structure_hash': 'dfb98d0d8ac4c8bde83fdb58e16f1b531397042d78dc9eae363147a35d712ac7', 'parameter_schema_hash': 'e33f117b370062bcb0b1fbb302da70d9f60a8fddcad055e514ba7e4c1d02550a', 'best_parameter_hash': '3d6f48a6aa804b7311a6e05320e9b61d9987501e74e8c645a482b6458aa4e47d', 'best_parameters': {'epsilon': 0.021242420542999622, 'slack_risk_penalty': 6.3241906279789335, 'slack_urgency_gain': 3.3527251509433778, 'energy_efficiency_weight': 0.856206257496663, 'criticality_weight': 0.17960302557623017, 'duration_uncertainty_ratio': 1.7913837346731663, 'wait_decay_rate': 0.0010557715376774165, 'uncertainty_slack_interaction': 4.2147470503664115, 'ddl_protection_gate': 0.7545068661627915, 'finfo_max_scale': 244190.83892955532}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '4ae211c9474a9a050bb1130a9d3c6399f1cc550bd589dedcd385ad81795776f5', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's stability with Parent 1's DDL-protection gating.

    Key structural improvements:
      - Introduces *soft DDL-protection gate* (sigmoid-based) that smoothly attenuates
        energy, criticality, and duration terms under slack stress (slk < 0), preserving
        gradient continuity while enforcing hard deadline compliance as top priority.
      - Retains Parent 2's robust normalized-linear slack gating (not quadratic) for penalty/gain,
        but now modulated by the gate — ensuring energy/criticality influence fades *gradually*
        as deadline pressure increases, not abruptly.
      - Keeps bounded linear duration-uncertainty blend (exec+comm + ratio×uncert) for numerical
        stability and interpretability.
      - Drops successor_pressure and sigmoid_denom_clip_max (no cross-gen evidence) and avoids
        exponential wait saturation fallbacks — uses pure exponential decay only.
      - All normalizations use mean-abs scale + epsilon; all divisions guarded; outputs finite & deterministic.
    Smaller score = higher priority.
    """
    eps = 0.021242420542999622
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize(x):
        x = np.asarray(x)
        abs_x = np.abs(x)
        scale = np.mean(abs_x) if np.all(np.isfinite(abs_x)) and np.mean(abs_x) > eps else eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    slack_penalty = np.where(slk < 0, 6.3241906279789335 * np.abs(slack_norm), -3.3527251509433778 * np.abs(slack_norm))
    gate_input = np.where(slk < 0, -slk, 0.0)
    ddl_gate = 1.0 / (1.0 + np.exp(-0.7545068661627915 * (gate_input + eps)))
    inv_energy = 1.0 / (energy + eps)
    energy_score_base = -0.856206257496663 * normalize(inv_energy)
    energy_score = energy_score_base * ddl_gate
    rank_active = np.where(slack_norm >= 0, rank, 0.0)
    rank_score_base = -0.17960302557623017 * normalize(rank_active + eps)
    rank_score = rank_score_base * ddl_gate
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.7913837346731663 * uncert_norm
    dur_score = normalize(dur_uncert_blend) * ddl_gate
    wait_sat = 1.0 - np.exp(-0.0010557715376774165 * wait)
    wait_score = -normalize(wait_sat + eps)
    unc_slack_interaction = 4.2147470503664115 * uncert_norm * np.where(slack_norm < 0, np.abs(slack_norm), 0.0)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 244190.83892955532
    min_safe = -finfo.max / 244190.83892955532
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
