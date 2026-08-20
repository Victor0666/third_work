import numpy as np
RULE_METADATA = {'structure_hash': '86231686c5380f387308bf8b5c565b2991c9c18124753fa0ae749281bef2147f', 'parameter_schema_hash': 'bd0f63e0fa02288ad7b50eb655884fbacde3cd8bd75c6edd0af3057b8bd86ce3', 'best_parameter_hash': '6fa637cd3317f047be2cfc28ce1702b782babb422d1e06efc478b63a6611d6e0', 'best_parameters': {'epsilon': 1.031089098182224e-08, 'slack_penalty_factor': 5.574651547300048, 'slack_sensitivity': 1.4441746755834695, 'energy_scale': 0.40952342622923726, 'criticality_weight': 1.1639565166207209, 'wait_fairness': 0.8014758323614352, 'uncertainty_slack_coupling': 0.6264550090264894, 'duration_balance': 0.2804838752594455, 'q75_quantile': 0.6692735899227669, 'q25_quantile': 0.13353466989610446}, 'optimizer_config_hash': '99f0307f8c00d3227b19c28340c4bbb3a1af02960262750afd225cb76d5560ee', 'parameter_diagnostics_hash': '70f83ccfc583db5fd7f98fd29051bc4f1a5a2c6cf5560ed228432431d1c38937', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Corrected priority rule with all numeric thresholds declared in PARAMETER_SCHEMA.
    
    Replaces hardcoded 0.75/0.25 with tunable quantiles. All other literals are -2,-1,0,1,2 or derived from np.finfo.
    Uses robust IQR normalization, uncertainty-coupled slack penalty, and rank-per-duration stabilization.
    Smaller score = higher priority.
    """
    eps = 1.031089098182224e-08

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        q75 = np.quantile(x, 0.6692735899227669, method='midpoint')
        q25 = np.quantile(x, 0.13353466989610446, method='midpoint')
        iqr = q75 - q25 + eps
        center = np.median(x)
        return (x - center) / iqr
    slack_abs = np.abs(slack)
    slack_sign = np.sign(slack)
    slack_powered = np.power(slack_abs + eps, 1.4441746755834695) * slack_sign
    negative_slack_mask = (slack < 0).astype(float)
    uncertainty_coupled_penalty = 5.574651547300048 * np.abs(slack_powered) * negative_slack_mask * np.tanh(0.6264550090264894 * uncertainty)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    duration_safe = min_exec_time + min_comm_time + eps
    rank_per_duration = upward_rank / duration_safe
    norm_rank_per_duration = robust_normalize(rank_per_duration)
    wait_boost = 0.8014758323614352 * np.tanh(ready_wait_time / (1.0 + eps))
    score = negative_slack_penalty + 0.40952342622923726 * norm_energy + 0.2804838752594455 * norm_duration - 1.1639565166207209 * norm_rank_per_duration - wait_boost
    return np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
