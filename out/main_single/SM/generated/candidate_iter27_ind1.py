import numpy as np
RULE_METADATA = {'structure_hash': 'b9487795a90c194120023310371092ca05e40090f7f15166d2079dd608cadd27', 'parameter_schema_hash': '4d63cd05e0a04a863759ec39878dc498a5a782f2eaf40ee580e62d7c23434c93', 'best_parameter_hash': '16a211fc83a4c400522a762aecb440c025a76a2606d313a59bf2180580d7dcbf', 'best_parameters': {'epsilon': 0.08114551332793327, 'slack_risk_penalty': 6.990598772850439, 'slack_urgency_gain': 4.88422696338494, 'energy_efficiency_weight': 1.873834266064207, 'criticality_exponent': 2.521024463107273, 'bottleneck_weight': 1.2115948522766564, 'duration_uncertainty_exponent': 2.157171288354777, 'wait_saturation_time': 87.5287575419425, 'host_load_sensitivity': 0.49974231726792717, 'piecewise_linear_gate_width': 3.945856128843779, 'slack_aware_energy_decay': 0.6732405619583668, 'robust_slack_normalization': 1.265129131772174}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '9767752cc628452e6b2d8210a704584108c5b157ca1521751d34c37d52b3a2b6', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Fixed instability from additive slack-energy coupling: replaced with bounded multiplicative modulation.
      - Robust_slk_norm (0–1) now directly scales energy_score_base — guarantees |modulated| ≤ |base|, eliminating amplification.
      - All features max-abs normalized; no quantile, softplus, or unstable transforms.
      - Criticality, bottleneck, duration-uncertainty, and starvation terms preserved and stabilized.
      - Exactly 12 parameters — all declared, all used, no literals beyond {-2,-1,0,1,2}.
      - Deterministic, finite, shape-correct, and DDL-first safe.
    """
    eps = 0.08114551332793327
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
    robust_slk_mag = np.power(abs_slk, 1.265129131772174)
    robust_slk_norm = robust_slk_mag / (robust_slk_mag + eps)
    slk_robust = robust_slk_norm * np.sign(slk)
    gate_width = 3.945856128843779 + eps
    ddl_gate = np.where(slk_robust <= 0, 1.0, np.where(slk_robust <= gate_width, 1.0 - slk_robust / gate_width, 0.0))
    slack_penalty = np.where(slk_robust < 0, 6.990598772850439 * np.abs(slk_robust), -4.88422696338494 * slk_robust)
    inv_energy = 1.0 / (energy + eps)
    energy_score_base = -1.873834266064207 * normalize(inv_energy)
    energy_score = energy_score_base * robust_slk_norm
    slack_decay = np.exp(-np.abs(slk_robust) * 0.6732405619583668)
    energy_score = energy_score * (1.0 - 0.49974231726792717 * ddl_gate) * slack_decay
    rank_powered = np.power(rank + eps, 2.521024463107273)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(slk_robust) + eps
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -1.2115948522766564 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    clipped_uncert = np.clip(uncert, 0.0, 2.0)
    duration_fuzzy = duration * np.power(1.0 + clipped_uncert, 2.157171288354777)
    dur_score = normalize(duration_fuzzy + eps)
    wait_sat = 1.0 - np.exp(-wait / (87.5287575419425 + eps))
    wait_score = -wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
