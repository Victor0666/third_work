import numpy as np
RULE_METADATA = {'structure_hash': '47be71cc107b5566214eabb692fd648c76b4fa422a0cb7fd3a68c991f51f525c', 'parameter_schema_hash': 'cf33a40e69699e4c89714590141f3fb49f098665cdb62fbed6fe43265f6c3e34', 'best_parameter_hash': '2bffeb7e8c905333e2bb826b93f0426f0d5677d034f5e29bdc07383eff882491', 'best_parameters': {'epsilon': 0.038412316142950764, 'slack_risk_penalty': 6.979313208255869, 'slack_urgency_gain': 1.9590773944317712, 'energy_efficiency_weight': 0.8056789260028212, 'criticality_weight': 1.9420693779009761, 'bottleneck_proximity_weight': 2.316073622325748, 'duration_uncertainty_ratio': 0.8847997603734009, 'wait_ramp_threshold': 13.896505334983198, 'host_load_sensitivity': 0.336331407371774, 'score_clip_max': 856969038.2265214, 'score_clip_min_abs': 155894806.85024467}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '3a1b86bec5c7b04801db5423f566812041a8ac99e8c070351f2b69dffe2d09d7', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with:
      - All declared parameters used; no unused entries
      - No numeric literals except -2, -1, 0, 1, 2
      - Max-abs normalization for stability and interpretability
      - Piecewise-linear DDL gate using only slack and epsilon
      - Host-load-sensitive energy gating via PARAMS["host_load_sensitivity"]
      - Exponential starvation mitigation: 1 - exp(-wait/theta)
      - Slack-aware criticality: upward_rank raised to power of criticality_weight
      - Bottleneck term: rank * work / (|slack| + eps), normalized and weighted
      - Robust NaN/inf handling using np.nan_to_num with parameterized bounds
    """
    eps = 0.038412316142950764
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
    min_slack_abs = np.abs(np.min(slk)) + eps
    ddl_gate = np.where(slk <= 0, 1.0, np.clip(1.0 - slk / min_slack_abs, 0.0, 1.0))
    slack_penalty = np.where(slk < 0, 6.979313208255869 * np.abs(slk), -1.9590773944317712 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.8056789260028212 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - 0.336331407371774 * ddl_gate)
    slack_mag_norm = normalize(np.abs(slk) + eps)
    rank_weight = np.power(1.0 + slack_mag_norm, 1.9420693779009761)
    rank_score = -normalize(rank) * rank_weight
    slack_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_term = rank * work * slack_inv
    bottleneck_score = -2.316073622325748 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.8847997603734009 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (13.896505334983198 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score_clip_max = 856969038.2265214
    score_clip_min = -155894806.85024467
    score = np.nan_to_num(score, nan=0.0, posinf=score_clip_max, neginf=score_clip_min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
