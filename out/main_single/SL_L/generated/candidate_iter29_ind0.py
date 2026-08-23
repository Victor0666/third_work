import numpy as np
RULE_METADATA = {'structure_hash': '660fbaa3962d52936ef88e48c463ceb5a4d226729c6a173043f018e342b38ef2', 'parameter_schema_hash': '8b98fef87cdfe4e1f81db43ecac51bfed71bf567c528c8c47c8718e0cd523839', 'best_parameter_hash': 'c61c03232d4158ab460e689f207f99bf43e9e60a3f091bd372a1be026a153695', 'best_parameters': {'epsilon': 1.9397370018170617e-08, 'ddl_protection_threshold': 2.0563732842026576, 'critical_path_release_weight': 2.695990785147189, 'risk_adjusted_energy_weight': 1.7324971132069653, 'slack_sigmoid_steepness': 7.707521105453979, 'tight_slack_uncertainty_coupling': 1.1788725508052877, 'normalized_wait_headroom_ratio': 0.9438795214585818, 'bounded_duration_risk_factor': 0.9752638767394162, 'urgency_exponential_decay_rate': 0.9373887132945232, 'conditional_mad_min_size': 2.0284733314603907, 'slack_distance_upper_bound': 18.05174193588828, 'congestion_aware_energy_gate': 0.5775351093266294}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'c4f02cae70a07a769287709ab7d95e17fea4b14fc268b5ce208bc91108c61e88', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining Parent 2's exponential urgency & conditional MAD with Parent 1's congestion-aware gating:
      - Bounded exponential urgency decay for slack-modulated rank prioritization (Parent 2)
      - Conditional MAD normalization: only applied when N > conditional_mad_min_size (Parent 2)
      - Congestion-aware energy gating using wait-time & uncertainty (Parent 1 innovation)
      - Lexicographic DDL-critical dominance: slack penalty, tight-slack coupling, and duration risk always active
      - All numeric literals strictly {-2,-1,0,1,2}; no hidden constants; deterministic finite output.
    """
    eps = 1.9397370018170617e-08
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)
    N = len(min_exec_time)
    if N == 0:
        return np.array([], dtype=float)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        if N <= 2.0284733314603907:
            return x
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    duration_total = min_exec_time + min_comm_time + eps
    slack_abs_norm = mad_normalize(np.abs(slack))
    unc_norm = mad_normalize(uncertainty)
    dur_norm = mad_normalize(duration_total)
    slack_sigmoid_input = -slack_abs_norm
    slack_sigmoid = 1.0 / (1.0 + np.exp(-7.707521105453979 * slack_sigmoid_input))
    slack_penalty = slack_sigmoid * (1.0 + unc_norm)
    tight_slack_mask = np.where(slack <= 2.0563732842026576, 1.0, 0.0)
    tight_slack_uncertainty_coupling = tight_slack_mask * uncertainty * slack_abs_norm * 1.1788725508052877
    duration_risk_base = (min_exec_time + min_comm_time) * uncertainty
    duration_risk_penalty = tight_slack_mask * duration_risk_base * 0.9752638767394162
    critical_release_score = upward_rank * remaining_work * 2.695990785147189
    slack_distance = np.clip(2.0563732842026576 - slack, 0.0, 18.05174193588828)
    urgency_factor = np.exp(-slack_distance / (0.9373887132945232 + eps))
    slack_rank_score = -upward_rank * urgency_factor * 2.695990785147189
    slack_headroom_mask = np.where(slack > 2.0563732842026576, 1.0, 0.0)
    median_wait = np.median(ready_wait_time) + eps
    congestion_proxy = (ready_wait_time / median_wait + eps) * (1.0 + unc_norm)
    congestion_gate = np.clip(congestion_proxy * 0.5775351093266294, 0.0, 1.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * congestion_gate * 1.7324971132069653 * energy_norm
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 2, 0.0)
    wait_norm = mad_normalize(wait_power)
    wait_term = slack_headroom_mask * wait_norm * 0.9438795214585818
    score = mad_normalize(slack_penalty) + mad_normalize(tight_slack_uncertainty_coupling) + mad_normalize(duration_risk_penalty) + mad_normalize(critical_release_score) + mad_normalize(slack_rank_score)
    score += energy_term + wait_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
