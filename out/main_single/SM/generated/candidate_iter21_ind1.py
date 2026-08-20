import numpy as np
RULE_METADATA = {'structure_hash': '958daff4b51f87c128deeb8666b162c4d09519540ebd6407710ae5b840b905a5', 'parameter_schema_hash': '0431ff60dc4323457eb43ffcc8035613a1366351b945641611d1ebb09ee01117', 'best_parameter_hash': '2dd3d7bc3181662de0352729491b1cd7b0de074195ba2257f8810b11d75c7504', 'best_parameters': {'epsilon': 2.2035229582562578e-05, 'slack_risk_penalty': 1.1787181232555146, 'slack_urgency_gain': 2.2616555630406503, 'energy_efficiency_weight': 2.515812904419168, 'criticality_exponent': 2.999870476375901, 'bottleneck_weight': 0.32799031161597314, 'duration_uncertainty_ratio': 0.7235495924702342, 'wait_saturation_time': 21.597920172939606, 'host_load_sensitivity': 0.6914190915941587, 'piecewise_linear_gate_width': 3.496725418789364, 'slack_aware_energy_decay': 0.4574837989084146}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '10683ab22cd282fb22e8d22a2d94d225b41d4dcc3ba74e733b5023d467027ad7', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Restored max-abs normalization (per reflection) for stability across all features.
      - Reintroduced `duration_uncertainty_ratio` as a robust linear blend — critical for fuzzy deadline handling.
      - Added `slack_aware_energy_decay`: multiplicative exponential attenuation of energy term under high lateness risk,
        replacing linear gating for smoother, physics-aligned suppression of energy optimization when deadlines dominate.
      - All features normalized via max-abs; no quantile, softplus, or unstable transforms.
      - Criticality uses exponentiated rank, bottleneck uses rank*work/(|slack|+eps), starvation uses exp(-wait/theta).
      - No unused parameters; all numeric literals are {-2,-1,0,1,2}; deterministic and finite.
    """
    eps = 2.2035229582562578e-05
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
    gate_width = 3.496725418789364 + eps
    ddl_gate = np.where(slk <= 0, 1.0, np.where(slk <= gate_width, 1.0 - slk / gate_width, 0.0))
    slack_penalty = np.where(slk < 0, 1.1787181232555146 * np.abs(slk), -2.2616555630406503 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.515812904419168 * normalize(inv_energy)
    slack_decay = np.exp(-np.abs(slk) * 0.4574837989084146)
    energy_score = energy_score * (1.0 - 0.6914190915941587 * ddl_gate) * slack_decay
    rank_powered = np.power(rank + eps, 2.999870476375901)
    rank_score = -normalize(rank_powered)
    slack_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_term = rank * work * slack_inv
    bottleneck_score = -0.32799031161597314 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.7235495924702342 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (21.597920172939606 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
