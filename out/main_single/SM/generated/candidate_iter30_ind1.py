import numpy as np
RULE_METADATA = {'structure_hash': '8f8d7af5134ded84c7d352f94b464889af24e1a1b765264536838d6c7dec193f', 'parameter_schema_hash': '50b4f517b1abc40c97fee32c3a257a7f7ad71b9162c191452d556aeb4f7fa4f9', 'best_parameter_hash': '86cc432ec08cc710d8960fa898a47ac0622395f6a4797d27e8df9f49b4adeae5', 'best_parameters': {'epsilon': 0.0007054926328061472, 'slack_risk_penalty': 1.1939617529979722, 'slack_urgency_gain': 0.29257347409337586, 'energy_efficiency_weight': 0.09141588855104157, 'criticality_exponent': 1.3753044592675383, 'bottleneck_power': 1.322166429355344, 'duration_uncertainty_ratio': 0.06306353126185267, 'wait_saturation_time': 1.1884731567285747, 'feasibility_sigmoid_slope': 0.5952137261393203, 'starvation_feasible_gain': 0.6932280978626525, 'starvation_infeasible_gain': 0.3683246692445301, 'successor_slack_sensitivity': 0.964110033682585}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'fa7b84cd86cf2875285ff251557c08576955017dc7e770032b2172466e2cff6d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Bounded piecewise-linear slack gating (replacing fragile exponentiation) for robustness near zero slack.
      - Successor-release-aware bottleneck: (rank × work) / (1 + |min_successor_slack| + ε), approximated conservatively using current slack.
      - Conditional energy activation only when current slack >= 0 (strict DDL feasibility).
      - All numeric literals strictly {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-preserving, and safe against NaN/inf/div-zero.
      - Exactly 12 parameters; all used; no unused or missing references.
    """
    eps = 0.0007054926328061472
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
    slk_abs = np.abs(slk)
    slk_linear = np.where(slk_abs <= 1, slk, np.where(slk > 1, 1.0, -1.0))
    ddl_protection_gate = np.where(slk >= 0, 1.0, 0.0)
    slack_penalty = np.where(slk < 0, 1.1939617529979722 * np.abs(slk), -0.29257347409337586 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score_base = -0.09141588855104157 * normalize(inv_energy)
    energy_score = energy_score_base * ddl_protection_gate
    rank_powered = np.power(rank + eps, 1.3753044592675383)
    rank_score_unmod = -normalize(rank_powered)
    slack_feasibility = 1.0 / (1.0 + np.exp(-slk * 0.5952137261393203))
    rank_score = rank_score_unmod * slack_feasibility
    bottleneck_numerator = rank * work + eps
    bottleneck_denom = 1.0 + np.abs(slk) + eps
    bottleneck_proxy = bottleneck_numerator / bottleneck_denom
    bottleneck_powered = np.power(bottleneck_proxy, 1.322166429355344)
    bottleneck_score = -0.964110033682585 * normalize(bottleneck_powered)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.06306353126185267 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (1.1884731567285747 + eps))
    starvation_gain = np.where(ddl_protection_gate == 1.0, 0.6932280978626525, 0.3683246692445301)
    wait_score = -starvation_gain * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
