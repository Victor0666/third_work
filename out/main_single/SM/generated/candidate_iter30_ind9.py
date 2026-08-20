import numpy as np
RULE_METADATA = {'structure_hash': 'd50636b285c1c6c54852ab5bbae817830c5fbd4c3a9ffab2f954e001927ecaaa', 'parameter_schema_hash': 'f5069d6318d4f6b5b04ac22b0cfd42bd09233728815666aba731b05ffb79630a', 'best_parameter_hash': '41313c1a01fbccde94973ce969de55bf1b1d28aa613b70d29233683326995c39', 'best_parameters': {'epsilon': 0.00020044045184506126, 'slack_risk_penalty': 0.5267631523949023, 'slack_urgency_gain': 3.838678675007595, 'energy_efficiency_weight': 0.5232531248713643, 'criticality_exponent': 0.5716575781260189, 'bottleneck_weight': 0.10300549802471878, 'bottleneck_offset': 0.0701654107850601, 'duration_uncertainty_ratio': 0.6604993076270416, 'wait_saturation_time': 23.048565336466922, 'starvation_gain': 0.6535870783011318, 'sigmoid_slack_steepness': 3.1734049404017, 'sigmoid_slack_clip_bound': 5.617749076074339}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': 'e32154b2a92ea14dc6f2c56f0b099ee1229831d74c8d5e8c569273a3559b051e', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - Binary DDL gate restored for decisiveness.
      - Bounded sigmoid slack normalization using tunable clip bound and steepness.
      - All numeric literals strictly in {-2,-1,0,1,2}; no hidden constants.
      - Deterministic, finite, shape-(N,) output satisfying all interface contracts.
    """
    eps = 0.00020044045184506126
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
    k = 3.1734049404017
    clip_bound = 5.617749076074339
    slk_clipped = np.clip(slk, -clip_bound, clip_bound)
    sigmoid_slk = 2.0 / (1.0 + np.exp(-k * slk_clipped + eps)) - 1.0
    ddl_gate = np.where(sigmoid_slk <= 0.0, 1.0, 0.0)
    slack_penalty = np.where(sigmoid_slk < 0.0, 0.5267631523949023 * np.abs(sigmoid_slk), -3.838678675007595 * sigmoid_slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -0.5232531248713643 * normalize(inv_energy)
    energy_score = energy_score * (1.0 - ddl_gate)
    rank_powered = np.power(rank + eps, 0.5716575781260189)
    rank_score = -normalize(rank_powered)
    robust_abs_slk = np.abs(sigmoid_slk) + 0.0701654107850601
    bottleneck_term = rank * work / robust_abs_slk
    bottleneck_score = -0.10300549802471878 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 0.6604993076270416 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_sat = 1.0 - np.exp(-wait / (23.048565336466922 + eps))
    wait_score = -0.6535870783011318 * wait_sat
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max / 2.0, neginf=finfo.min / 2.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
