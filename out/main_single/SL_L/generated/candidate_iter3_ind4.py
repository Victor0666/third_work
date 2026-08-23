import numpy as np
RULE_METADATA = {'structure_hash': '8ed356d2571cdd59bc81656ec60a9a2072578b3450cf7f7b2d80abb5df95ad63', 'parameter_schema_hash': 'b141b899f4abb2fef0be4ef5f0f8edf93ecd477d391c68bb91e63f6020b76fe9', 'best_parameter_hash': '77a7d9fa7d918d52f3a4f4f74fe0118040666a8f3d53bc1b5e74dc8f182b8769', 'best_parameters': {'epsilon': 0.005508870822077102, 'safe_slack_slope': 0.15842565034713602, 'risky_slack_quadratic': 1.343773281448804, 'criticality_gate_threshold': -0.40085496336257354, 'energy_sensitivity': 0.9028910703259672, 'successor_pressure_weight': 0.9102153946036997, 'wait_decay_rate': 0.2592452856377505, 'rank_activation_weight': 0.7449530471520777, 'wait_boost_weight': 1.216318997817826, 'nan_replacement': -233269.99618938728, 'posinf_clip': 1137.5185717937375, 'neginf_clip': -32467224.8084023}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '3d7f08ca994124a11d80128c5246da425fc97ab806f293de183dd3ff820e48c9', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with bounded piecewise slack penalty and successor-release pressure.
    
    Key improvements:
    - Replaces exponentiated slack penalty with robust piecewise: linear for safe (slack > 0), 
      quadratic for risky (slack <= 0) — avoids instability while preserving risk-aware steepness.
    - Introduces normalized successor-release pressure: norm_work * norm_uncert * (1 - norm_slack), 
      clipped to [0,1], directly encoding urgency of large/uncertain sub-DAGs under deadline stress.
    - Simplifies criticality gate to univariate `norm_slack <= threshold` (no uncertainty thresholding),
      improving robustness per self-reflection.
    - All numeric literals are from {-2,-1,0,1,2}; normalization uses np.finfo(float).tiny only implicitly via eps.
    """
    eps = 0.005508870822077102
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_norm(x):
        x_abs = np.abs(x)
        denom = np.mean(x_abs) + eps
        return x / denom
    norm_slack = robust_norm(slack)
    norm_energy = robust_norm(min_incremental_energy)
    norm_duration = robust_norm(min_exec_time + min_comm_time)
    norm_rank = robust_norm(upward_rank)
    norm_work = robust_norm(remaining_work)
    norm_wait = robust_norm(ready_wait_time)
    norm_uncert = robust_norm(uncertainty)
    safe_mask = (norm_slack > 0).astype(float)
    risky_mask = (norm_slack <= 0).astype(float)
    slack_penalty = safe_mask * 0.15842565034713602 * norm_slack + risky_mask * 1.343773281448804 * norm_slack ** 2
    successor_pressure_raw = norm_work * norm_uncert * (1.0 - norm_slack)
    successor_pressure = np.clip(successor_pressure_raw, 0.0, 1.0) * 0.9102153946036997
    rank_activation_mask = (norm_slack <= -0.40085496336257354).astype(float)
    wait_boost = 1.0 - np.exp(-0.2592452856377505 * (norm_wait + eps))
    score = slack_penalty
    score += safe_mask * 0.9028910703259672 * norm_energy
    score -= rank_activation_mask * 0.7449530471520777 * norm_rank
    score += successor_pressure
    score -= wait_boost * 1.216318997817826
    score = np.nan_to_num(score, nan=-233269.99618938728, posinf=1137.5185717937375, neginf=-32467224.8084023)
    return score
