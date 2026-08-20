import numpy as np
RULE_METADATA = {'structure_hash': 'cb1ac5947271b80f4cd2aeb87dcf5b19096d7f1a0c83d53935d17404eff2b331', 'parameter_schema_hash': 'ff3051bd3ad3f6589fde36e5134c7b7213b30d65bdd4cf64335d1b372f919f3f', 'best_parameter_hash': '1b08eb6734acab05151b34db782ba7db9af3ab60a996c06be9d4de1663755313', 'best_parameters': {'epsilon': 9.028244904894477e-07, 'slack_risk_penalty': 1.5969307757040183, 'slack_feasibility_weight': 1.8836005100185427, 'energy_efficiency_weight': 1.4975279615313357, 'criticality_weight': 1.9881988900298992, 'wait_decay_rate': 0.1149979083929799, 'percentile_range_scale': 1.2191463390105477, 'successor_release_weight': 0.11534212497490358, 'duration_energy_coupling_power': 1.3978253833024947, 'joint_norm_quantile': 84.93292525152168, 'positive_slack_urgency_gain': 0.23757372333157384}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '6208f8c19c7a90259d9c5727cd3a4e47ebaeb191efb1be49de4e6035c00005c7', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with smooth DDL feasibility gating, unified joint normalization,
    and eliminated percentile parameter redundancy.
    
    Key structural improvements:
      - Replaced hard `slk >= 0` gate with smooth logistic feasibility weight: 
        `sigma(slack * slack_feasibility_weight)` — avoids discontinuities near slack=0.
      - Unified joint normalization over duration, energy, rank, and successor-load using single
        robust 90th-quantile scale — improves stability and reduces parameter coupling.
      - Removed `pctl_low`/`pctl_high`; fixed lower bound implied by median centering.
      - All components share identical normalization context; `positive_slack_urgency_gain` now tunable.
    """
    eps = 9.028244904894477e-07
    N = len(min_exec_time)
    exec_t = np.asarray(min_exec_time, dtype=np.float64)
    comm_t = np.asarray(min_comm_time, dtype=np.float64)
    energy = np.asarray(min_incremental_energy, dtype=np.float64)
    slk = np.asarray(slack, dtype=np.float64)
    rank = np.asarray(upward_rank, dtype=np.float64)
    work = np.asarray(remaining_work, dtype=np.float64)
    wait = np.asarray(ready_wait_time, dtype=np.float64)
    duration = exec_t + comm_t
    feasibility_weight = 1.0 / (1.0 + np.exp(-1.8836005100185427 * slk))
    joint_features = np.abs(np.concatenate([duration + eps, energy + eps, rank + eps, rank * (work + eps) + eps]))
    q90 = np.percentile(joint_features, 84.93292525152168)
    rng_safe = np.where(np.isfinite(q90) & (q90 > eps), q90, eps)

    def joint_normalize(x):
        x = np.asarray(x)
        center = np.median(joint_features)
        return (x - center) / (1.2191463390105477 * rng_safe + eps)
    slack_norm = joint_normalize(slk)
    slack_penalty = np.where(slk < 0, 1.5969307757040183 * slack_norm ** 2, -1.5969307757040183 * 0.23757372333157384 * np.abs(slack_norm))
    inv_energy = 1.0 / (energy + eps)
    energy_score = -1.4975279615313357 * feasibility_weight * joint_normalize(inv_energy + eps)
    rank_score = -1.9881988900298992 * feasibility_weight * joint_normalize(rank + eps)
    dur_energy_prod = (duration + eps) * (energy + eps)
    dur_energy_urgency = np.power(dur_energy_prod, 1.3978253833024947)
    dur_energy_score = feasibility_weight * joint_normalize(dur_energy_urgency + eps)
    wait_sat = 1.0 - np.exp(-0.1149979083929799 * wait)
    wait_score = -joint_normalize(wait_sat + eps)
    weighted_successor_load = rank * (work + eps)
    successor_release_score = -0.11534212497490358 * feasibility_weight * joint_normalize(weighted_successor_load + eps)
    score = slack_penalty + energy_score + rank_score + dur_energy_score + wait_score + successor_release_score
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape (N,), got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values detected in priority score'
    return score
