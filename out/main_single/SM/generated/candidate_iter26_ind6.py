import numpy as np
RULE_METADATA = {'structure_hash': 'd038d51789f4d3041978c5cb1b80f3664fac80c02b170a33c2e383393ea45ebd', 'parameter_schema_hash': '2dcfbda959a9f458b0c6f4fa271fb787668ec7d2582748310896249883fafbbc', 'best_parameter_hash': '5589b02e4cde0046202c80ca550ada160a242624d60de3bb7d02872205bcfbf0', 'best_parameters': {'epsilon': 0.00016942339039339893, 'slack_risk_penalty': 2.6132988116763625, 'slack_urgency_gain': 5.999994072979094, 'energy_efficiency_weight': 0.5296369167453986, 'criticality_exponent': 1.862450841594631, 'bottleneck_weight': 3.9206846497401004, 'duration_uncertainty_ratio': 0.6835801431569499, 'wait_saturation_time': 17.143284272376356, 'host_load_sensitivity': 0.6203015275705674, 'piecewise_linear_gate_width': 0.7293445749636437, 'slack_aware_energy_decay': 0.9289953495846182, 'robust_slack_normalization': 1.0498523784269989}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '4914e20c18c9c3ea72d76a93971b29b9456f4d5c447c116c919bfc194ae65c2b', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with hybrid duration-uncertainty coupling:
      - Retains max-abs normalization and robust slack power-law normalization from Parent 2.
      - Replaces *either* linear blend *or* pure power-law with a **single tunable exponent**
        on the uncertainty factor: (exec+comm) * (1 + clipped_uncert)^power.
        This preserves structural simplicity (12 params), avoids convex interpolation,
        and directly generalizes both parents: power=0 → linear; power>0 → nonlinear scaling.
      - Keeps dual-gated energy term (ddl_gate * slack_decay) for strict DDL-first safety.
      - Preserves exponential starvation mitigation and bottleneck proximity via robust slack.
      - All numeric literals are {-2,-1,0,1,2}; no hidden constants; fully deterministic and finite.
    """
    eps = 0.00016942339039339893
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
    abs_slk = np.abs(slk) + eps
    robust_slk_mag = np.power(abs_slk, 1.0498523784269989)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    gate_width = 0.7293445749636437 + eps
    ddl_gate = np.where(slk_robust <= 0, 1.0, np.where(slk_robust <= gate_width, 1.0 - slk_robust / gate_width, 0.0))
    slack_penalty = np.where(slk_robust < 0, 2.6132988116763625 * np.abs(slk_robust), -5.999994072979094 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.5296369167453986 * normalize(inv_energy)
    slack_decay = np.exp(-np.abs(slk_robust) * 0.9289953495846182)
    energy_score = energy_score * (1.0 - 0.6203015275705674 * ddl_gate) * slack_decay
    rank_powered = np.power(rank + eps, 1.862450841594631)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + eps
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -3.9206846497401004 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    uncert_clipped = np.clip(uncert, 0.0, 2.0)
    duration_with_uncert = duration * np.power(1.0 + uncert_clipped, 0.6835801431569499)
    dur_score = normalize(duration_with_uncert + eps)
    wait_sat = 1.0 - np.exp(-wait / (17.143284272376356 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
