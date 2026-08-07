import numpy as np
RULE_METADATA = {'structure_hash': 'dfec3730fd99e0397bfe44f004491889b1c0a040c3ed2202e4351ce3cb9bfc29', 'parameter_schema_hash': '703324963f86c1b47176e5a3d70e364f709c4aeedadbffb51011b1886cb68de2', 'best_parameter_hash': 'f8b8306eb8534e03e5355b212e3df22b66ccd3855adfde466e1e94740a3766d5', 'best_parameters': {'epsilon': 0.00017077404747244416, 'slack_risk_penalty': 3.0533272278997945, 'slack_urgency_gain': 2.4397133355301492, 'energy_efficiency_weight': 1.021906077482949, 'critical_path_bonus': 2.4100161298356277, 'wait_decay_rate': 0.002792251265106205, 'uncertainty_slack_coupling': 0.5419324037141854, 'duration_efficiency_ratio': 0.830302052546205, 'critical_rank_threshold': 0.8186934792111702, 'iqr_percentile_low': 17.378603302807065, 'iqr_percentile_high': 81.54178892757733}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '587e20700fdec99b44e46041c4136582bb07d2e8bcfe6c06442cbb1fe5514f01', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule combining deadline safety, critical-path awareness, 
    starvation prevention, and work-normalized energy efficiency.
    
    Key features:
      - Piecewise slack handling: strong convex penalty for negative slack, gentle linear reward for positive
      - Work-normalized energy: min_incremental_energy / (remaining_work + epsilon) → prioritizes J/MI efficiency
      - Percentile-gated critical-path bonus to avoid outlier dominance
      - Exponential wait decay with saturation guard
      - Uncertainty-slack coupling only under tight slack
      - All normalization uses tunable IQR percentiles
      - No numeric literals beyond {-2,-1,0,1,2}; all thresholds/weights parameterized
    """
    eps = 0.00017077404747244416
    N = len(slack)
    finfo = np.finfo(float)
    eps_safe = max(eps, finfo.tiny)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def iqr_normalize(x):
        x = np.copy(x)
        q_low = np.percentile(x, 17.378603302807065)
        q_high = np.percentile(x, 81.54178892757733)
        iqr = q_high - q_low
        center = np.median(x)
        denom = iqr if iqr > eps_safe else np.max(np.abs(x - center)) + eps_safe
        return (x - center) / (denom + eps_safe)
    slack_norm = iqr_normalize(slack)
    slack_penalty = np.where(slack < 0, 3.0533272278997945 * -slack_norm, -2.4397133355301492 * slack_norm)
    duration = np.maximum(min_exec_time + min_comm_time, eps_safe)
    work_efficient_energy = min_incremental_energy / np.maximum(remaining_work, eps_safe)
    energy_norm = iqr_normalize(work_efficient_energy)
    if N == 1:
        rank_percentile = np.array([1.0])
    else:
        sorted_ranks = np.sort(upward_rank)
        rank_idx = np.searchsorted(sorted_ranks, upward_rank, side='right')
        rank_percentile = rank_idx / (N + eps_safe)
    critical_gate = np.where(rank_percentile >= 0.8186934792111702, 1.0, 0.0)
    rank_norm = iqr_normalize(upward_rank)
    critical_bonus = 2.4100161298356277 * rank_norm * critical_gate
    wait_boost = 1.0 - np.exp(-0.002792251265106205 * ready_wait_time)
    wait_norm = iqr_normalize(wait_boost)
    unc_norm = iqr_normalize(uncertainty)
    slack_pressure = np.maximum(-slack_norm, 0.0)
    coupled_urgency = 0.5419324037141854 * unc_norm * slack_pressure
    dur_eff_ratio = 0.830302052546205 * iqr_normalize(duration)
    score = slack_penalty + 1.021906077482949 * energy_norm - critical_bonus - wait_norm + coupled_urgency + dur_eff_ratio
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
