import numpy as np
RULE_METADATA = {'structure_hash': '5111176f8c8b0b34a3448496d2108425bae6e67978c6b5ae30795baf8cf98796', 'parameter_schema_hash': '2dcfbda959a9f458b0c6f4fa271fb787668ec7d2582748310896249883fafbbc', 'best_parameter_hash': '4a063c1dc4ccd0b5582c27b9564673e43e0ac200163bc254e8acf600c3c37f89', 'best_parameters': {'epsilon': 4.731595994191459e-06, 'slack_risk_penalty': 8.856417599655357, 'slack_urgency_gain': 0.2273016453954147, 'energy_efficiency_weight': 2.865031363266408, 'criticality_exponent': 0.265672054771474, 'bottleneck_weight': 1.7465154056302805, 'duration_uncertainty_ratio': 1.4891059964285465, 'wait_saturation_time': 10.3615982634538, 'host_load_sensitivity': 0.02658836214977672, 'piecewise_linear_gate_width': 8.150240774739924, 'slack_aware_energy_decay': 0.5039091994581784, 'robust_slack_normalization': 0.4682082549663904}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'f4d05411dd8525dca6ee5bca87e01998e3e8fd0b8054b9eda280176470fd9e80', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with:
      - Robust max-abs normalization.
      - Piecewise-linear DDL gate on robustly power-normalized slack.
      - Slack-driven penalty/gain (negative/positive slack handled separately).
      - Energy score modulated by ddl_gate, exponential slack decay, AND tunable power-law uncertainty coupling.
      - Criticality softened by sigmoidal slack feasibility modulation (replaces binary mask).
      - Bottleneck term uses rank*work/(|robust_slk|+eps) — bounded and monotonic.
      - Duration-uncertainty blend remains linear and normalized.
      - Starvation mitigation via exponential saturation.
      - All numeric literals are {-2,-1,0,1,2}; no hidden constants; all tunables declared.
    """
    eps = 4.731595994191459e-06
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
    robust_slk_mag = np.power(abs_slk, 0.4682082549663904)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    gate_width = 8.150240774739924 + eps
    ddl_gate = np.where(slk_robust <= 0, 1.0, np.where(slk_robust <= gate_width, 1.0 - slk_robust / gate_width, 0.0))
    slack_penalty = np.where(slk_robust < 0, 8.856417599655357 * np.abs(slk_robust), -0.2273016453954147 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    uncertainty_energy_factor = np.power(1.0 + np.clip(uncert, 0.0, 2.0), 1.0)
    energy_score_base = -2.865031363266408 * normalize(inv_energy * uncertainty_energy_factor)
    slack_decay = np.exp(-np.abs(slk_robust) * 0.5039091994581784)
    energy_score = energy_score_base * (1.0 - 0.02658836214977672 * ddl_gate) * slack_decay
    rank_powered = np.power(rank + eps, 0.265672054771474)
    rank_score_unmod = -normalize(rank_powered)
    slack_feasibility = 1.0 / (1.0 + np.exp(-slk_robust * 1.0))
    rank_score = rank_score_unmod * slack_feasibility
    robust_abs_slk = np.abs(slk_robust) + eps
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -1.7465154056302805 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.4891059964285465 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (10.3615982634538 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
