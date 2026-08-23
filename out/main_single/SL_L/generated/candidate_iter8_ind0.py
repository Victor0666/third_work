import numpy as np
RULE_METADATA = {'structure_hash': '676b8c4e5f236a5cd8b94075a517938b173757cf28c94b04bfc413caca24ce51', 'parameter_schema_hash': '85a6f95c66ddd7104efd02a315dccb34c7e645d6677f6748198d9d3b24358ddf', 'best_parameter_hash': '01ac03892703d62c7169802a6b4ed0e288b27bb693fca12f1407fd39b978d60c', 'best_parameters': {'epsilon': 1.042095572193757e-09, 'slack_urgency_scale': 2.800691470648671, 'energy_efficiency_bias': 1.0720495513660688, 'critical_path_leverage': 1.7488602120688341, 'wait_fairness_gain': 0.008145699967026, 'duration_balance': 0.08892519127832231, 'work_density_weight': 0.6246146947321575, 'robustness_mad_factor': 1.4878498794159052, 'slack_pressure_tanh_scale': 0.5942620802876404, 'energy_slack_coupling': 0.2576969324049589, 'rank_uncertainty_coupling': 0.04628250450618704, 'uncertainty_penalty_cap': 1.123117687148255}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'c0385c77793e9690e940e5d172f8eb8d37cc40f0b15212554d49438b702fa52d', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """
    Hybrid priority rule combining Parent 2's smooth tanh slack urgency and stable pre-gated rank normalization
    with Parent 1's explicit uncertainty penalty capping and refined fairness logic.
    Key structural improvements:
      - Introduces `uncertainty_penalty_cap` to bound worst-case uncertainty amplification (novel hard cap),
      - Replaces fragile linear uncertainty gating with a *smooth sigmoid gate* on (uncertainty / unc_med) for stability,
      - Uses robust median-based feasibility gating (not mean or ad-hoc thresholds) for all conditionals,
      - Retains Parent 2's decoupled energy-slack coupling and unified tanh urgency for deadline safety,
      - Preserves Parent 1's wait fairness saturation via exp(-x) but applies it *after* robust normalization.
    All parameters used; no numeric literals except {-2,-1,0,1,2}; fully deterministic and finite.
    """
    eps = 1.042095572193757e-09
    finfo = np.finfo(np.float64)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=np.float64), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.4878498794159052 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = -slack
    slack_urgency = 2.800691470648671 * np.tanh(slack_pressure * 0.5942620802876404)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_base = 1.0720495513660688 * energy_norm
    energy_slack_gate = np.clip(slack_pressure, 0.0, np.inf)
    energy_score = energy_base * (1.0 + 0.2576969324049589 * np.tanh(energy_slack_gate * 0.5942620802876404))
    rank_norm = mad_normalize(upward_rank)
    unc_med = np.median(uncertainty)
    unc_sigmoid_gate = 1.0 / (1.0 + np.exp((uncertainty - unc_med) / np.maximum(eps, unc_med)))
    feasibility_gate = (slack >= 0.0).astype(float)
    critical_gate = feasibility_gate * unc_sigmoid_gate
    critical_boost = 1.7488602120688341 * rank_norm * critical_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_bonus = 0.6246146947321575 * work_density_norm * feasibility_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_boost = 1.0 - np.exp(-0.008145699967026 * np.maximum(wait_norm, 0.0))
    wait_score = wait_boost * feasibility_gate
    unc_norm = mad_normalize(uncertainty)
    resilience_base = 0.04628250450618704 * rank_norm * unc_norm * feasibility_gate
    resilience_boost = np.clip(resilience_base, -1.123117687148255, 1.123117687148255)
    score = slack_urgency + 0.08892519127832231 * duration_norm + energy_score - critical_boost - work_density_bonus + wait_score - resilience_boost
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    return score
