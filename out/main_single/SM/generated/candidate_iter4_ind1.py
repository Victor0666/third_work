import numpy as np
RULE_METADATA = {'structure_hash': '43d9e983264072a70654eb42012e70b77c371522b06d4385df52756a66b9557e', 'parameter_schema_hash': 'e73557c36abbe9c3c2507dab17783acdbf43d6fe2125fbfeef6e50c2b1b149dc', 'best_parameter_hash': '04f8b2ffecbac65710c5dc1d2154b5331be6f787e51c5c4af700479e278007f1', 'best_parameters': {'epsilon': 0.0007216584920015106, 'slack_risk_penalty': 7.625738098454818, 'slack_urgency_gain': 4.471848296120757, 'energy_efficiency_weight': 1.7245374921345729, 'criticality_weight': 0.012615798678214267, 'duration_uncertainty_ratio': 1.0435525240133805, 'wait_decay_rate': 0.12193792459759477, 'uncertainty_slack_interaction': 1.0454494428624315, 'finfo_max_scale': 86509549.69573405, 'ddl_protection_gate_width': 0.2138687182345232, 'load_successor_release_weight': 0.0707474650823588, 'host_load_interaction_exponent': 0.6744469963289176}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'd8e06a942da810c4c9d7266ec6d477a94acac7c979702cb4b8c653f0f8e5505f', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule featuring:
      - Smooth sigmoid DDL protection gate (replaces hard sign threshold)
      - Successor-release load-aware term: combines remaining_work and upward_rank to estimate release pressure
      - Nonlinear host-load interaction: pow(remaining_work * uncertainty + eps, exponent) to amplify load-awareness under stress
      - Median-based robust normalization (more stable than mean under skew)
      - All operations guarded against NaN/inf/zero; deterministic and finite output.
    """
    eps = 0.0007216584920015106
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
        scale = max(med, mad, eps)
        return x / (scale + eps)
    gate_center = 0.0
    gate_width = 0.2138687182345232
    ddl_gate = 1.0 / (1.0 + np.exp(-(slk - gate_center) / (gate_width + eps)))
    slack_penalty = 7.625738098454818 * np.maximum(-slk, 0.0) + 4.471848296120757 * np.minimum(slk, 0.0)
    slack_penalty = normalize(slack_penalty + eps)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.7245374921345729 * normalize(inv_energy)
    rank_score = -0.012615798678214267 * ddl_gate * normalize(rank + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.0435525240133805 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-0.12193792459759477 * wait)
    wait_score = -normalize(wait_sat + eps)
    unc_slack_interaction = 1.0454494428624315 * uncert_norm * (1.0 - ddl_gate)
    release_pressure = rank * work
    release_score = -0.0707474650823588 * normalize(release_pressure + eps)
    load_factor = (work * uncert) ** 0.6744469963289176
    load_score = normalize(load_factor + eps)
    score = slack_penalty + energy_score + rank_score + dur_score + wait_score + unc_slack_interaction + release_score + load_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 86509549.69573405
    min_safe = -finfo.max / 86509549.69573405
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
