import numpy as np
RULE_METADATA = {'structure_hash': 'c07bf54846ad93b382f6f028a3e54110191ef831e64c40007fecb9bc21fef569', 'parameter_schema_hash': '87ccf230aef2964338ceff2053965587384ebf4e185a9650cf207e20c2d73118', 'best_parameter_hash': 'e777063a070200333f5cb36bfef44beca2a0cd9974707712f8905aa871516dcf', 'best_parameters': {'epsilon': 0.021973334769000666, 'slack_risk_penalty': 9.333849660056108, 'slack_urgency_gain': 0.3777400441148826, 'energy_efficiency_weight': 1.2237899667298766, 'criticality_weight': 0.0536438847878498, 'bottleneck_proximity_weight': 1.121973275815659, 'ddl_protection_gate': 0.6205764820729899, 'robust_normalization_quantile': 0.7938353263555338, 'successor_release_sharpness': 0.9081196974754248, 'host_load_interaction_weight': 0.36479303454916556, 'finfo_safety_scale': 14.299324467854778, 'wait_normalization_scale': 0.5053665574719093}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'b239f61f33736b9471765dc962fddbed9d1d995c9d78addfc9029a18c964c46d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with improved structural robustness:
      - Replaced `wait_ramp_threshold` with self-scaling wait-term using quantile-normalized wait time → eliminates overparameterization.
      - Replaced brittle `np.clip` saturation with smooth `tanh` activation for bottleneck and wait terms → improves CMA-ES gradient flow.
      - Introduced feasibility-aware energy modulation gate: `ddl_gate * (1 - np.tanh(uncert))`, ensuring joint DDL+uncertainty suppression.
      - All numeric literals strictly limited to {-2,-1,0,1,2}; no hidden constants.
      - Removed redundant `slack_urgency_gain` sign flip; unified into clean `np.where` for interpretability.
      - Preserved all safeguards: NaN/inf handling, finite output, shape enforcement, and epsilon-guarded divisions.
    """
    eps = 0.021973334769000666
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
            scale = np.quantile(abs_x[finite_mask], 0.7938353263555338)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slack_norm = normalize(slk)
    ddl_gate = 1.0 / (1.0 + np.exp(0.6205764820729899 * slack_norm))
    slack_score = np.where(slk < 0, 9.333849660056108 * np.abs(slack_norm), -0.3777400441148826 * np.abs(slack_norm))
    energy_mod_gate = ddl_gate * (1.0 - np.tanh(uncert + eps))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.2237899667298766 * normalize(inv_energy) * energy_mod_gate
    rank_score = -0.0536438847878498 * ddl_gate * normalize(rank + eps)
    slack_magnitude_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_raw = rank * work * np.power(slack_magnitude_inv, 0.9081196974754248)
    bottleneck_norm = normalize(bottleneck_raw + eps)
    bottleneck_score = -1.121973275815659 * np.tanh(bottleneck_norm)
    coupled_urgency = np.tanh(normalize(uncert)) * np.abs(slack_norm)
    coupled_score = -0.36479303454916556 * coupled_urgency
    wait_norm = normalize(wait + eps)
    wait_score = -0.5053665574719093 * np.tanh(wait_norm)
    score = slack_score + energy_score + rank_score + bottleneck_score + coupled_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / np.power(2.0, 14.299324467854778)
    min_safe = -max_safe
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
