import numpy as np
RULE_METADATA = {'structure_hash': '3e937902cebce7cfbcf163e221fce2407e9237d7c144fa7a6c70dd2ee4bf73fd', 'parameter_schema_hash': 'bf8c9e5e1f4413ef371d8b99537ba767e404136e8a5613ecd9bf08c473bd1442', 'best_parameter_hash': 'd4d5f8211e149fc01a81d4a3c0064c725bbd755a3084aae025ed7c7e4afe022e', 'best_parameters': {'epsilon': 6.1396088045270135e-06, 'slack_risk_penalty': 7.186901323049727, 'slack_urgency_gain': 1.5367109563455512, 'energy_efficiency_weight': 2.946823364833524, 'criticality_weight': 0.01038828397175035, 'bottleneck_proximity_weight': 2.3071831244064445, 'duration_uncertainty_ratio': 1.2730065038664673, 'wait_ramp_threshold': 35.46841695953362, 'piecewise_slack_transition': 0.6284888999827999, 'robust_normalization_quantile': 0.9404242382994767, 'finfo_clamp_scale': 1013.0389343154969}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '6f5abf55491576b703852e1f83dacc9f97413faa79a23a387045bdb525238ab2', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with:
      - Replaced sigmoid DDL gate with interpretable piecewise-linear urgency transition (sharper, bounded, no asymptotes)
      - Added explicit starvation-aware wait-saturation term using bounded exponential ramp (1 - exp(-wait/θ)) instead of linear clip
      - Unified bottleneck scoring via robust slack-normalized critical-path importance: upward_rank * remaining_work / (|slack| + eps)
      - Energy term now gated by DDL feasibility mask (binary) rather than soft sigmoid, enforcing hard feasibility boundary
      - All normalizations use quantile-based scaling; no mean/median bias
      - Final score clamped to safe finite range using np.finfo directly
    """
    eps = 6.1396088045270135e-06
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
            scale = np.quantile(abs_x[finite_mask], 0.9404242382994767)
            scale = np.where(scale > eps, scale, eps)
        else:
            scale = eps
        return x / (scale + eps)
    delta = 0.6284888999827999
    ddl_feasible_mask = np.where(slk >= delta, 1.0, 0.0)
    ddl_urgency = np.clip((slk + delta) / (2 * delta), 0.0, 1.0)
    slack_penalty = np.where(slk < 0, 7.186901323049727 * np.abs(slk), -1.5367109563455512 * slk)
    inv_energy = 1.0 / (energy + eps)
    energy_score = -2.946823364833524 * normalize(inv_energy) * ddl_feasible_mask
    rank_score = -0.01038828397175035 * ddl_urgency * normalize(rank + eps)
    bottleneck_term = rank * work / (np.abs(slk) + eps)
    bottleneck_score = -2.3071831244064445 * normalize(bottleneck_term + eps)
    duration = exec_t + comm_t
    dur_norm = normalize(duration + eps)
    uncert_norm = normalize(uncert + eps)
    dur_uncert_blend = dur_norm + 1.2730065038664673 * uncert_norm
    dur_score = normalize(dur_uncert_blend)
    wait_exp_ramp = 1.0 - np.exp(-wait / (35.46841695953362 + eps))
    wait_score = -normalize(wait_exp_ramp + eps)
    score = slack_penalty + energy_score + rank_score + bottleneck_score + dur_score + wait_score
    finfo = np.finfo(np.float64)
    max_safe = finfo.max / 1013.0389343154969
    min_safe = -finfo.max / 1013.0389343154969
    score = np.clip(score, min_safe, max_safe)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
