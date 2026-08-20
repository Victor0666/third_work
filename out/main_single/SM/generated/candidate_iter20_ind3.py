import numpy as np
RULE_METADATA = {'structure_hash': 'b9173d9102af7b7b6923f37e806c0a0230f97a84d4e15c85c30a0ac50c35664a', 'parameter_schema_hash': '259afc723037871140a7d2f868c48c8bd6d0bd531497ea85706c9e74f3b6cc34', 'best_parameter_hash': '679b63446ac7306361d4d9696ad1b915bd7cd419e8ca021041c2fd51a00db575', 'best_parameters': {'epsilon': 0.00018184131379874594, 'slack_risk_penalty': 5.231868419345928, 'slack_urgency_gain': 2.5593377485353828, 'energy_efficiency_weight': 1.8058082436528178, 'criticality_exponent': 0.33451953219529473, 'bottleneck_weight': 1.0465206944383598, 'wait_saturation_time': 19.000511066361053, 'host_load_sensitivity': 0.4305778884904462, 'piecewise_linear_gate_width': 3.2195250997774845, 'temporal_risk_sharpening': 2.070126472610175, 'load_bottleneck_coupling_strength': 0.3315905938892084}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '09bf39d911fec2aa638ad5e751ef6c5df0b9611c9488fbd92ac14506d04ca3d2', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Crisp piecewise-linear DDL gate for robust deadline boundary control.
      - Max-abs normalization for stability and interpretability.
      - Temporal risk sharpening: (exec_t + comm_t + uncert)^p before normalization.
      - Load-bottleneck coupling: energy_per_duration × bottleneck_term × ddl_gate —
        enabling energy-aware bottleneck prioritization only where deadline safety permits.
      - Exponential starvation mitigation and exponentiated criticality.
      - All numeric literals are {-2,-1,0,1,2}; no unused parameters; deterministic & finite.
    """
    eps = 0.00018184131379874594
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
    gate_width = 3.2195250997774845 + eps
    ddl_gate = np.where(slk <= 0, 1.0, np.where(slk <= gate_width, 1.0 - slk / gate_width, 0.0))
    slack_penalty = np.where(slk < 0, 5.231868419345928 * np.abs(slk), -2.5593377485353828 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.8058082436528178 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - 0.4305778884904462 * ddl_gate)
    rank_powered = np.power(rank + eps, 0.33451953219529473)
    rank_score = -normalize(rank_powered)
    slack_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_term = rank * work * slack_inv
    bottleneck_score = -1.0465206944383598 * normalize(bottleneck_term + eps)
    temporal_risk = exec_t + comm_t + uncert
    temporal_risk_sharpened = np.power(temporal_risk + eps, 2.070126472610175)
    dur_score = normalize(temporal_risk_sharpened)
    wait_sat = 1.0 - np.exp(-wait / (19.000511066361053 + eps))
    wait_score = -wait_sat
    energy_per_duration = energy / (exec_t + comm_t + eps)
    load_proxy = normalize(energy_per_duration + eps)
    coupled_term = load_proxy * bottleneck_term * ddl_gate
    coupled_score = -0.3315905938892084 * normalize(coupled_term + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score + coupled_score
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2, neginf=-finfo.max / 2)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
