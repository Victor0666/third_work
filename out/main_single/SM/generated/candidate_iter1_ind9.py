import numpy as np
RULE_METADATA = {'structure_hash': '0d302da05c9a71e5cdac36ef876583a71ee8528df2ed0c7eda1f24502624f69c', 'parameter_schema_hash': 'a6c98075dfa06e6ed09e0b2c68e4638f0dea4cfbdc86d75fefc30aeee1950f45', 'best_parameter_hash': '1ed1c6d6d92f156d58436b3ad822a95269bfdf81ffd0d1d3f4e93cc95309f056', 'best_parameters': {'epsilon': 1.7561767808416961e-09, 'slack_risk_penalty': 9.150253638801054, 'slack_urgency_gain': 0.18204639143879855, 'energy_scale': 3.7955129127422356, 'criticality_weight': 0.30256075905877444, 'wait_bias': 0.6379373168341992, 'uncertainty_slack_coupling': 0.790291306284092, 'duration_fairness': 0.3743551047236848}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '8aadf5f431a39125d02f51e2822315b783045ce706495ab2c270fd6e32df907e', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Novel priority rule emphasizing deadline risk containment, robust criticality gating,
    and balanced energy-duration tradeoff. Uses piecewise slack sensitivity and
    uncertainty-coupled urgency. Smaller score = higher priority."""
    eps = 1.7561767808416961e-09

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        center = np.median(x)
        scale = np.median(np.abs(x - center)) + eps
        return (x - center) / scale
    duration = min_exec_time + min_comm_time
    norm_duration = robust_normalize(duration)
    norm_energy = robust_normalize(min_incremental_energy)
    slack = np.asarray(slack, dtype=float)
    slack_abs = np.abs(slack)
    pos_slack = slack[slack > eps]
    slack_margin = np.median(pos_slack) if len(pos_slack) > 0 else 1.0
    slack_margin = max(slack_margin, eps)
    slack_urgency = np.zeros_like(slack)
    slack_urgency[slack < 0] = 9.150253638801054 * -slack[slack < 0]
    in_soft_region = (slack >= 0) & (slack <= slack_margin)
    slack_urgency[in_soft_region] = 0.18204639143879855 * (slack[in_soft_region] / (slack_margin + eps)) ** 2
    norm_upward = robust_normalize(upward_rank)
    slack_penalty_factor = np.clip(1.0 - np.maximum(-slack, 0) / (slack_margin + eps), 0.0, 1.0)
    gated_criticality = norm_upward * slack_penalty_factor
    norm_uncertainty = robust_normalize(uncertainty)
    coupled_urgency = np.maximum(slack_urgency, 0) * np.clip(norm_uncertainty, 0.0, 2.0) * 0.790291306284092
    wait_bias = 0.6379373168341992 * ready_wait_time
    wait_bias_norm = robust_normalize(wait_bias)
    fairness_term = np.tanh(norm_duration) * norm_duration
    score = 3.7955129127422356 * norm_energy + 0.3743551047236848 * fairness_term - slack_urgency - coupled_urgency - 0.30256075905877444 * gated_criticality - wait_bias_norm
    score = np.nan_to_num(score, nan=np.median(score), posinf=np.max(score), neginf=np.min(score))
    return score
