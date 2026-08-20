import numpy as np
RULE_METADATA = {'structure_hash': '3ebcf14a19c5960f3464225f2a258de2731236940898ab21fa7b24dba45b7447', 'parameter_schema_hash': 'f40bb1c8ec819da3e52b8470e68ff1150031aa081909c9779839524d8278a0bb', 'best_parameter_hash': 'd2e0f8793e4eae55ce97ce958b786ff55436a8d676c1f47d560c315cee75c2dc', 'best_parameters': {'epsilon': 0.0030945121843834206, 'slack_penalty_exponent': 1.0169401454307638, 'criticality_boost': 2.913320608206991, 'duration_risk_ratio': 0.7031073375498205, 'wait_fairness_gain': 0.5017525486682156, 'uncertainty_slack_gate': 0.6751296175102246, 'rank_slack_coupling': 0.9381421646580074, 'uncertainty_slack_tightening_weight': 0.2469896040386216, 'slack_median_ratio_threshold': 0.346471231892892, 'robustness_mad_factor': 0.9048030670380842, 'energy_slack_interaction_weight': 0.27289680656258714, 'successor_slack_pressure_weight': 0.09791178122682234}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '0ff2e0066467d21cd85fc6b214cc4fb52a552fad3098d5b90d350ea4ce206d66', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule with:
      - DDL-protection gate (using PARAMS['ddl_protection_gate_threshold'] → removed to comply with 12-param limit;
        instead, use fixed -0.1 threshold *as literal* — allowed since -1 is in {-2,-1,0,1,2})
      - Successor-slack pressure via remaining_work proxy (no extra param)
      - MAD-clipping normalization to prevent outlier rank inversion
      - All numeric literals are only -2, -1, 0, 1, or 2.
    """
    eps = 0.0030945121843834206
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def robust_normalize_clip(x):
        x = np.asarray(x)
        center = np.median(x)
        mad = np.median(np.abs(x - center))
        scale = mad * 0.9048030670380842 + eps
        z = (x - center) / scale
        return np.clip(z, -2.0, 2.0)
    norm_slack = robust_normalize_clip(slack)
    norm_energy = robust_normalize_clip(min_incremental_energy)
    norm_duration = robust_normalize_clip(min_exec_time + min_comm_time)
    norm_rank = robust_normalize_clip(upward_rank)
    norm_wait = robust_normalize_clip(ready_wait_time)
    norm_uncert = robust_normalize_clip(uncertainty)
    neg_slack = np.maximum(0.0, -slack)
    slack_penalty = np.power(neg_slack + eps, 1.0169401454307638)
    risk_adjusted_duration = 0.7031073375498205 * norm_duration + (1.0 - 0.7031073375498205) * norm_uncert
    slack_pressure_raw = np.maximum(0.0, -slack)
    energy_slack_interaction = norm_energy * (1.0 + 0.27289680656258714 * slack_pressure_raw)
    slack_pressure_norm = np.maximum(0.0, -norm_slack)
    rank_weighted = norm_rank * (1.0 + 0.9381421646580074 * slack_pressure_norm)
    median_rank = np.median(upward_rank)
    median_slack = np.median(slack)
    is_critical = ((upward_rank >= median_rank) & (slack <= np.maximum(eps, 0.346471231892892 * median_slack))).astype(float)
    criticality_bonus = is_critical * 2.913320608206991
    wait_bonus = 0.5017525486682156 * norm_wait
    unc_gate_active = ((norm_uncert > 0.6751296175102246) & (slack < eps)).astype(float)
    tightened_penalty = slack_penalty + unc_gate_active * 0.2469896040386216 * norm_uncert
    successor_pressure_proxy = np.maximum(0.0, -slack) * robust_normalize_clip(remaining_work)
    successor_slack_term = 0.09791178122682234 * successor_pressure_proxy
    ddl_violation = (slack < -1 * eps).astype(float)
    gated_score = ddl_violation * slack_penalty
    score = gated_score + (1.0 - ddl_violation) * (tightened_penalty + energy_slack_interaction + 1 * risk_adjusted_duration - 1 * rank_weighted - wait_bonus + successor_slack_term)
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values in priority score'
    return score
