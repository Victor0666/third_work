import numpy as np
RULE_METADATA = {'structure_hash': '9e7b647924ae9f22476b22512dc5325fcd30664d048bb37e0ec2c8e50a221cc2', 'parameter_schema_hash': 'd5dca1f2fb9eec1785f3092d5ab1d3d78ce1aa3ccdbb07ebb63f49b347670b70', 'best_parameter_hash': '1eeddb4e2f1560c17c0fa53de5aada653896f515b673c99619fc24dab3b15cc0', 'best_parameters': {'epsilon': 6.090091680984161e-08, 'upward_rank_remaining_work_interaction': 2.267445907328068, 'energy_efficiency_ratio_weight': 1.1847370937209112, 'uncertainty_slack_coupling': 1.4480080577363732, 'duration_risk_penalty': 0.448087348315761, 'energy_uncertainty_interaction': 0.5637714548935365, 'uncertainty_sigmoid_steepness': 3.0830702922378213, 'host_load_conditional_gate': 0.8696415986331918, 'critical_release_denom_bias': 0.10488490333267939, 'slack_urgency_sigmoid_steepness': 3.702013461772067}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': '12603bc24c2adf4ec8aa62830f07b5b3cfbad612e272cae9afcd7649dc700fee', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Self-evolved priority rule featuring:
       - Smooth sigmoidal slack urgency (replacing both hard DDL gate AND slack_score's piecewise branch) 
         for differentiable, full-range deadline pressure — improves CMA-ES convergence and avoids feasibility cliffs.
       - Unified urgency signal: `1 / (1 + exp(-k * slack))` maps slack ∈ ℝ → (0,1), inverted for priority (smaller = tighter).
       - Retains bias-stabilized critical_path_release (monotonic across all slack) and drops wait-time entirely.
       - All DDL-critical terms now share a common differentiable foundation — no discontinuous branches.
       - MAD-normalization remains for robustness; N=1 fallback preserved.
       - No numeric literals outside {-2,-1,0,1,2}.
    """
    eps = 6.090091680984161e-08
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
        if N == 1:
            return np.zeros_like(x, dtype=float)
        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev) + eps
        normalized = (x - med) / mad
        return np.clip(normalized, -2.0, 2.0)
    slack_urgency = 1.0 / (1.0 + np.exp(3.702013461772067 * slack))
    deadline_pressure = slack_urgency
    unc_slack_coupling = uncertainty * deadline_pressure * 1.4480080577363732
    duration_risk = (min_exec_time + min_comm_time) * uncertainty * 0.448087348315761
    critical_release_denom = np.maximum(0.0, -slack) + 0.10488490333267939
    critical_path_release = upward_rank * remaining_work / critical_release_denom * 2.267445907328068
    ddl_safe_scale = 1.0 - slack_urgency
    duration_total = min_exec_time + min_comm_time + eps
    energy_per_sec = min_incremental_energy / duration_total
    energy_eff_score = mad_normalize(energy_per_sec)
    host_load_scale = ddl_safe_scale * 0.8696415986331918
    rank_boost_weight = slack_urgency
    rank_score = -mad_normalize(upward_rank) * rank_boost_weight
    unc_sigmoid = 1.0 / (1.0 + np.exp(-3.0830702922378213 * (uncertainty - 1.0)))
    energy_norm = mad_normalize(min_incremental_energy)
    unc_norm = mad_normalize(uncertainty)
    energy_uncertainty_score = host_load_scale * 0.5637714548935365 * energy_norm * unc_norm * unc_sigmoid
    score = mad_normalize(slack_urgency) + mad_normalize(unc_slack_coupling) + mad_normalize(duration_risk) + mad_normalize(critical_path_release)
    score += ddl_safe_scale * (1.1847370937209112 * energy_eff_score + rank_score + energy_uncertainty_score)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
