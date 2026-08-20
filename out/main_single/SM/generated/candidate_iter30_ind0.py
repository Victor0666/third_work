import numpy as np
RULE_METADATA = {'structure_hash': '0dfce62b2b14d0dcc1934c6cfdb208fcda5790841f83928923178e1b79da5b58', 'parameter_schema_hash': '3ea8018aed54dca68c5539ecb12bbb0d749ceabb7ff2e12a00b23204cce1c813', 'best_parameter_hash': '40bd8d7833760c9ebf0a949d32e92930a2384c70bee7b4fbb237e6ad255be3b8', 'best_parameters': {'epsilon': 0.0034386975053897454, 'slack_risk_penalty': 1.5952392136759683, 'slack_urgency_gain': 4.8693422888265525, 'energy_efficiency_weight': 0.2322403121356562, 'criticality_exponent': 2.1610205901553194, 'bottleneck_weight': 0.7900504180907967, 'bottleneck_offset': 0.611516828293609, 'duration_uncertainty_ratio': 0.14750841900056805, 'wait_saturation_time': 19.898503310346683, 'starvation_gain': 0.3665673684517344, 'robust_slack_normalization': 0.9626705076281643, 'energy_suppression_steepness': 7.968395183482213}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '84a58c4dcea6067c44fa1d367270a8709d0012bd3ec7bcdcddb836c0076dc0d4', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Smooth sigmoidal energy suppression (replacing threshold gate) for robust feasibility-energy tradeoff.
      - Robust min-max normalization for slack, energy, and uncertainty to stabilize cross-scenario behavior.
      - Unified risk-weighted execution score via bounded tanh: avoids linear blending fragility and ensures monotonic, saturating response to duration+uncertainty.
      - All parameters used; no unused entries; only {-2,-1,0,1,2} literals; epsilon & machine bounds via PARAMS/np.finfo.
      - Deterministic, finite, shape-(N,), side-effect-free.
    """
    eps = 0.0034386975053897454
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    uncert = np.asarray(uncertainty, dtype=np.float64)

    def normalize_mm(x):
        x = np.asarray(x, dtype=np.float64)
        finite_mask = np.isfinite(x)
        if np.any(finite_mask):
            x_min = np.min(x[finite_mask])
            x_max = np.max(x[finite_mask])
            range_val = x_max - x_min
            range_val = np.where(range_val > eps, range_val, eps)
            return (x - x_min) / range_val
        else:
            return np.zeros_like(x)
    abs_slk = np.abs(slk) + eps
    robust_slk_mag = np.power(abs_slk, 0.9626705076281643)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    energy_suppress_gate = 1.0 / (1.0 + np.exp(-7.968395183482213 * np.abs(slk_robust)))
    slack_penalty = np.where(slk_robust < 0, 1.5952392136759683 * np.abs(slk_robust), -4.8693422888265525 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.2322403121356562 * normalize_mm(inv_energy)
    energy_score = energy_score * energy_suppress_gate
    rank_powered = np.power(rank + eps, 2.1610205901553194)
    rank_score = -normalize_mm(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + 0.611516828293609
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -0.7900504180907967 * normalize_mm(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize_mm(duration + eps)
    uncert_norm = normalize_mm(uncert + eps)
    risk_execution_raw = np.tanh(0.14750841900056805 * (dur_norm + uncert_norm))
    risk_execution_score = normalize_mm(risk_execution_raw + 1.0)
    wait_sat = 1.0 - np.exp(-wait / (19.898503310346683 + eps))
    wait_score = -0.3665673684517344 * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + risk_execution_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
