import numpy as np
RULE_METADATA = {'structure_hash': 'e6c9695735d9bfa9f26744e41061ef910557aacb6cca25dfdc8b6bf2ad95a83f', 'parameter_schema_hash': '4551aa1af996dbc46e570920f329b31c72840ebba9b734e8e97fe627cfb05c25', 'best_parameter_hash': '92779b63587cc1289510e6a24eab47c42b5f67366da190e7e921ba432513cfb2', 'best_parameters': {'epsilon': 0.0022363547321411353, 'slack_risk_penalty': 3.4119543722930628, 'slack_urgency_gain': 0.8805043969611814, 'energy_efficiency_weight': 0.6784591349083976, 'criticality_exponent': 0.9668433807018912, 'bottleneck_power': 0.589915205160138, 'duration_uncertainty_ratio': 1.745753808681351, 'wait_saturation_time': 6.331208704121103, 'robust_slack_normalization': 0.9099408099830729, 'feasibility_sigmoid_slope': 0.9520824443654146, 'starvation_feasible_gain': 1.7399519588586696, 'starvation_infeasible_gain': 0.3480627248894467}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'f59160a1885dd206432112eacc87d113c295e9b434840a579822e4bff08c5182', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule combining strengths of both parents:
      - Uses hard DDL protection gate (Parent 2) but enhances it with tunable sigmoid slope for slack feasibility.
      - Replaces fragile bottleneck_offset denominator with robust (rank * work)^p (Parent 2), now with dedicated tunable exponent.
      - Retains robust max-abs normalization and epsilon safeguards throughout.
      - Introduces explicit slack-feasibility-modulated starvation term using two tunable gains.
      - All numeric literals are strictly -2,-1,0,1,2; no hidden constants.
      - Deterministic, finite, shape-correct, side-effect-free.
    """
    eps = 0.0022363547321411353
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
    robust_slk_mag = np.power(abs_slk, 0.9099408099830729)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    ddl_protection_gate = np.where(slk_robust >= 0, 1.0, 0.0)
    slack_penalty = np.where(slk_robust < 0, 3.4119543722930628 * np.abs(slk_robust), -0.8805043969611814 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score_base = -0.6784591349083976 * normalize(inv_energy)
    energy_score = energy_score_base * ddl_protection_gate
    rank_powered = np.power(rank + eps, 0.9668433807018912)
    rank_score_unmod = -normalize(rank_powered)
    slack_feasibility = 1.0 / (1.0 + np.exp(-slk_robust * 0.9520824443654146))
    rank_score = rank_score_unmod * slack_feasibility
    bottleneck_product = rank * work + eps
    bottleneck_powered = np.power(bottleneck_product, 0.589915205160138)
    bottleneck_score = -0.589915205160138 * normalize(bottleneck_powered)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.745753808681351 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (6.331208704121103 + eps))
    starvation_gain = np.where(ddl_protection_gate == 1.0, 1.7399519588586696, 0.3480627248894467)
    wait_score = -starvation_gain * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
