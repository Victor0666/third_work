import numpy as np
RULE_METADATA = {'structure_hash': '177a1b4410f188a3f893721c248bfbe44e8a6eb6731919e4045dcdb012fdbbe6', 'parameter_schema_hash': '6d3b0f1a755e9eb7826563028771feff8fce7fb68527a17e0d5aea450573d106', 'best_parameter_hash': '85a50657f0c097cb44dc2bb1da3caa184655860ebd077bec7b21a5fc52278302', 'best_parameters': {'epsilon': 0.0076939338589137405, 'slack_risk_penalty': 5.097982242259469, 'slack_inverse_urgency_exponent': 0.20924985668057094, 'energy_efficiency_weight': 2.7514907083026947, 'criticality_weight': 0.7072660345403072, 'bottleneck_proximity_weight': 0.24779196611054494, 'duration_uncertainty_ratio': 0.19878971761719622, 'wait_ramp_threshold': 19.52065378285357, 'ddl_feasibility_margin': 0.9882078505307674, 'robust_normalization_quantile': 0.7924581803361429, 'host_load_proxy_exponent': 0.9382721880064973, 'score_clipping_bound': 24386727662979.074}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'a121cc7603a620c4bd7801c52ae781b654eaff021a02676e5b4007952b6846f8', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths from both parents:
      - Hard DDL feasibility mask (from Parent 2) for strict deadline-first enforcement.
      - Inverse-slack urgency (from Parent 1) for stable positive-slack scaling — replaces linear gain.
      - Host-load proxy (Parent 2) for congestion-aware energy gating.
      - Bottleneck term uses min(exec_time, comm_time) — robust to dominance shifts, no slack singularity.
      - All normalizations use quantile-based robust scaling (Parent 2).
      - Final score clamping uses machine-precision-aware bounds (Parent 1 style with log-transformed parameter).
      - Removed redundant soft gates and sigmoid ramps: hard mask + inverse urgency provides cleaner control.
    """
    eps = 0.0076939338589137405
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
            scale = np.quantile(abs_x[finite_mask], 0.7924581803361429)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    slk_norm = normalize(slk)
    rank_norm = normalize(rank + eps)
    exec_norm = normalize(exec_t + eps)
    comm_norm = normalize(comm_t + eps)
    energy_norm = normalize(energy + eps)
    wait_norm = normalize(wait + eps)
    uncert_norm = normalize(uncert + eps)
    slack_penalty = np.where(slk < 0, 5.097982242259469 * np.abs(slk_norm), -np.power(np.abs(slk) + eps, -0.20924985668057094))
    ddl_feasibility_margin = 0.9882078505307674
    ddl_feasible_mask = np.where(slk >= -ddl_feasibility_margin, 1.0, 0.0)
    host_load_proxy = np.power(wait + uncert + eps, 0.9382721880064973)
    host_load_gate = 1.0 / (1.0 + host_load_proxy)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.7514907083026947 * normalize(inv_energy) * ddl_feasible_mask * host_load_gate
    rank_score = -0.7072660345403072 * rank_norm
    min_duration_bound = np.minimum(exec_t, comm_t)
    bottleneck_interaction = rank_norm * normalize(min_duration_bound + eps)
    bottleneck_score = -0.24779196611054494 * normalize(bottleneck_interaction + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    dur_uncert_blend = dur_norm + 0.19878971761719622 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_saturation = 1.0 - np.exp(-wait / (19.52065378285357 + eps))
    wait_score = -normalize(wait_saturation + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = 24386727662979.074
    min_safe = -max_safe
    score = np.nan_to_num(score, nan=0.0, posinf=max_safe, neginf=min_safe)
    score = np.clip(score, min_safe, max_safe)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
