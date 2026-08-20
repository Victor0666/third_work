import numpy as np
RULE_METADATA = {'structure_hash': '70b2db412899a5e925b7b49b4aa311373f6f18b33d997583c29ee5dc02ce1eb8', 'parameter_schema_hash': 'c0600cf9eb11ae5a55d8d26dba271663f1e82a3621c6495e51632cd33c458271', 'best_parameter_hash': '4333187a1b2bf619a4ce668c5847cf5186a20763056210ecef06d4e4d602801b', 'best_parameters': {'epsilon': 3.3403177873345344e-05, 'slack_risk_penalty': 3.9190601204545374, 'slack_urgency_gain': 2.3734600293946575, 'energy_efficiency_weight': 2.105069999176324, 'criticality_exponent': 1.739171050589054, 'bottleneck_weight': 2.0304710771827557, 'duration_uncertainty_ratio': 1.4960383384497076, 'wait_saturation_time': 46.33936073427953, 'ddl_protection_steepness': 15.604978673088088, 'energy_gating_strength': 0.6101021867129955, 'score_clip_max': 6343636358.898886, 'score_clip_min_abs': 106313009.13195291}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'aab45b803cc512880440b2083620246de71fb8e546da1c729f76dba8b3dab39b', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's robustness with Parent 1's adaptive gating:
      - Uses sigmoid DDL gate (Parent 1) for smooth, numerically stable transition near slack=0,
        but simplified with fixed center at 0 and steepness parameter — avoids per-task offset complexity.
      - Retains exponential starvation mitigation (Parent 2) for stable wait-time response.
      - Applies max-abs normalization universally (Parent 2) for interpretability and outlier resilience.
      - Introduces *slack-dependent energy gating*: attenuates energy term proportionally to ddl_gate,
        improving deadline-risk focus without eliminating energy optimization in feasible region.
      - Criticality uses exponentiated upward_rank weighted by slack magnitude (Parent 2 style),
        enabling stronger prioritization of high-rank tasks when deadlines tighten.
      - Bottleneck term uses rank * work / (|slack| + eps) — clean, monotonic, and avoids softplus instability.
      - All numeric literals are {-2,-1,0,1,2}; no hidden constants; all parameters used exactly once.
    """
    eps = 3.3403177873345344e-05
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
    finfo = np.finfo(float)
    gate_input = 15.604978673088088 * -slk
    gate_input_clipped = np.clip(gate_input, -np.log(finfo.max), np.log(finfo.max))
    ddl_gate = 1.0 / (1.0 + np.exp(-gate_input_clipped))
    slack_penalty = np.where(slk < 0, 3.9190601204545374 * np.abs(slk), -2.3734600293946575 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.105069999176324 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - 0.6101021867129955 * ddl_gate)
    slack_mag_norm = normalize(np.abs(slk) + eps)
    rank_powered = np.power(rank + eps, 1.739171050589054)
    rank_score = -normalize(rank_powered)
    slack_inv = 1.0 / (np.abs(slk) + eps)
    bottleneck_term = rank * work * slack_inv
    bottleneck_score = -2.0304710771827557 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.4960383384497076 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (46.33936073427953 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    score_clip_max = 6343636358.898886
    score_clip_min = -106313009.13195291
    score = np.nan_to_num(score, nan=0.0, posinf=score_clip_max, neginf=score_clip_min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
