import numpy as np
RULE_METADATA = {'structure_hash': '9b67af54f2237a2120c37e4f4a21b40d61a3eeff2d3ed7af3c72597f791f070a', 'parameter_schema_hash': '781c6d710ee32d67a859f703ecf9aae409fe43940a0701d655314eaa0a032ab3', 'best_parameter_hash': 'dc5b933c4650ae25ef1092ca3baa749def17939bbebf3b249d6ffa658b8d69ee', 'best_parameters': {'epsilon': 7.65998567342638e-05, 'ddl_protection_threshold': 1.7032608439338495, 'critical_path_release_weight': 3.8887405727569195, 'risk_adjusted_energy_weight': 1.5981470539789877, 'slack_sigmoid_steepness': 7.823930491543528, 'tight_slack_uncertainty_coupling': 0.016091768341581544, 'normalized_wait_headroom_ratio': 0.7827206955329897, 'bounded_duration_risk_factor': 1.4911516743623003, 'urgency_exponential_decay_rate': 2.6428436448314843, 'conditional_mad_min_size': 5.115151821038827, 'slack_distance_upper_bound': 27.92880967343833}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '89ac76785770e3469df4818d76cd91f18bbc9502686f7be7696a8fb748599156', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with two key structural improvements:
      - Replaces linear slack-distance coupling in slack_rank_score with *bounded exponential decay*:
        urgency grows as exp(-slack_distance / rate), sharply prioritizing critical-path tasks when slack is tight,
        but remains finite and numerically stable even at extreme negative slack.
      - Introduces *conditional MAD normalization*: only applies mad_normalize when N > conditional_mad_min_size,
        avoiding statistical degeneracy (e.g., zero MAD, infinite values) in sparse ready sets (N ≤ 3).
      - All other components preserved from validated hybrid structure: bounded sigmoid slack penalty,
        exclusive tight-slack coupling, bounded duration risk, normalized wait-term, and lexicographic gating.
      - No numeric literals beyond {-2,-1,0,1,2}; all tunables accessed via PARAMS; deterministic and finite-output guaranteed.
    """
    eps = 7.65998567342638e-05
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
        if N <= 5.115151821038827:
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
    slack_sigmoid = 1.0 / (1.0 + np.exp(-7.823930491543528 * slack_sigmoid_input))
    slack_penalty = slack_sigmoid * (1.0 + unc_norm)
    tight_slack_mask = np.where(slack <= 1.7032608439338495, 1.0, 0.0)
    tight_slack_uncertainty_coupling = tight_slack_mask * uncertainty * slack_abs_norm * 0.016091768341581544
    duration_risk_base = (min_exec_time + min_comm_time) * uncertainty
    duration_risk_penalty = tight_slack_mask * duration_risk_base * 1.4911516743623003
    critical_release_score = upward_rank * remaining_work * 3.8887405727569195
    slack_distance = np.clip(1.7032608439338495 - slack, 0.0, 27.92880967343833)
    urgency_factor = np.exp(-slack_distance / (2.6428436448314843 + eps))
    slack_rank_score = -upward_rank * urgency_factor * 3.8887405727569195
    slack_headroom_mask = np.where(slack > 1.7032608439338495, 1.0, 0.0)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_term = slack_headroom_mask * 1.5981470539789877 * energy_norm
    wait_power = np.where(slack_headroom_mask > 0.0, ready_wait_time ** 2, 0.0)
    wait_norm = mad_normalize(wait_power)
    wait_term = slack_headroom_mask * wait_norm * 0.7827206955329897
    score = mad_normalize(slack_penalty) + mad_normalize(tight_slack_uncertainty_coupling) + mad_normalize(duration_risk_penalty) + mad_normalize(critical_release_score) + mad_normalize(slack_rank_score)
    score += energy_term + wait_term
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
