import numpy as np
RULE_METADATA = {'structure_hash': '5b5c56b1441089ae5fcb77b8652dcc53390761e1572fc1142e18db1baf8ca729', 'parameter_schema_hash': '0b5ebdfa36b49b521db5b6b3fd3e093067a6fe33169ef620a642744637745ae0', 'best_parameter_hash': '3174507e9ef2f74f5279a09dc0947e046fbda6294b3fabc7378bc53947fe888f', 'best_parameters': {'epsilon': 0.006444536497785801, 'slack_penalty_exponent': 2.2407152702706936, 'criticality_boost': 1.6051222001565701, 'energy_efficiency_scale': 0.16697416524787223, 'duration_risk_ratio': 0.24239132642290118, 'wait_fairness_gain': 0.5167186906485093, 'uncertainty_slack_gate': 0.07455137568098827, 'rank_slack_coupling': 0.5122058049978161, 'uncertainty_slack_tightening_weight': 0.8649003126871754, 'slack_median_ratio_threshold': 0.8956253574206258, 'ddl_protection_boost': 0.9888107710552095, 'work_slack_ratio_weight': 2.960335368959769}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': 'd9ec83c43ffdca6e79515253d4b45ecbe36586a333a5eec9e6049f5027b5bd8d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule: hard-deadline protection gate, monotonic work/slack coupling, and urgency-preserving piecewise slack penalty.
    
    Key improvements:
    - Replaces tanh with binary DDL-protection gate: when median_slack <= 0, amplify both rank and slack penalties → enforces feasibility-first hierarchy.
    - Replaces clipped successor_release with monotonic `remaining_work / (|slack| + eps)` — preserves critical-path signal even for small positive slack.
    - Uses piecewise `np.where(slack < 0, ..., 0)` for slack penalty instead of smoothed tanh → retains sharp urgency discrimination for lateness risk.
    - All numeric literals are -2,-1,0,1,2; all parameters declared and used exactly once.
    - Robust normalization and finite safeguards via np.finfo.
    """
    eps = 0.006444536497785801
    N = len(slack)
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
    median_slack = np.median(slack)
    ddl_protection_active = (median_slack <= eps).astype(float)
    ddl_boost = 1.0 + 0.9888107710552095 * ddl_protection_active
    neg_slack = np.where(slack < 0, -slack, 0.0)
    slack_penalty = np.power(neg_slack + eps, 2.2407152702706936)
    duration = min_exec_time + min_comm_time
    duration_norm = robust_norm(duration)
    uncertainty_norm = robust_norm(uncertainty)
    risk_adjusted_duration = 0.24239132642290118 * duration_norm + (1 - 0.24239132642290118) * uncertainty_norm
    energy_norm = robust_norm(min_incremental_energy)
    energy_penalty = 0.16697416524787223 * energy_norm
    rank_base = robust_norm(upward_rank)
    rank_weighted = rank_base * (1 + 0.5122058049978161 * np.where(slack < 0, 1.0, 0.0))
    rank_weighted = rank_weighted * ddl_boost
    median_rank = np.median(upward_rank)
    abs_median_slack = np.abs(median_slack) + eps
    is_critical = ((upward_rank >= median_rank) & (slack <= 0.8956253574206258 * abs_median_slack)).astype(float)
    criticality_bonus = is_critical * 1.6051222001565701
    work_slack_ratio = remaining_work / (np.abs(slack) + eps)
    work_slack_norm = robust_norm(work_slack_ratio)
    successor_release = 2.960335368959769 * work_slack_norm
    wait_norm = robust_norm(ready_wait_time)
    wait_bonus = -0.5167186906485093 * wait_norm
    unc_gate_active = ((uncertainty_norm > 0.07455137568098827) & (slack < eps)).astype(float)
    tightened_penalty = slack_penalty + unc_gate_active * 0.8649003126871754 * uncertainty_norm
    tightened_penalty = tightened_penalty * ddl_boost
    score = tightened_penalty + energy_penalty + 1 * risk_adjusted_duration - 1 * rank_weighted + 0 * criticality_bonus - 1 * successor_release + wait_bonus
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=finfo.max, posinf=finfo.max, neginf=finfo.min)
    assert score.shape == (N,), f'Expected shape {(N,)}, got {score.shape}'
    assert np.all(np.isfinite(score)), 'Non-finite values in priority score'
    return score
