import numpy as np
RULE_METADATA = {'structure_hash': '2a79d83eac4cfea5457ff51ca527bd6e186e46ccca5d99f0faf2fa108774fbc7', 'parameter_schema_hash': '9c1888b0406be15424a8707f42d2214cd0b8b4b2396e9848c27361ec73d3a4ef', 'best_parameter_hash': '34a008b7d3ae180c317afc55284f3d373a1c471d91d85502e9f0410ded8f0b8a', 'best_parameters': {'epsilon': 0.0008421228389077335, 'slack_risk_penalty': 2.2033948406328503, 'slack_urgency_scale': 3.917681883473796, 'energy_efficiency_bias': 1.8217581398268758, 'critical_path_leverage': 0.19308241631162032, 'wait_fairness_gain': 1.0454667312120292, 'uncertainty_sensitivity': 0.2543766112015269, 'duration_balance': 0.00012470332864303609, 'work_density_weight': 0.5772474330340494, 'robustness_mad_factor': 1.027772543994368, 'slack_decay_scale': 0.22181884749895714, 'energy_slack_gate_exponent': 0.658169399103394}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'fd90e4534022c358655c3423f245f4be0a7e14b7c7db549824d39cd94b206afc', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Improved priority rule with exactly 12 parameters:
    - Removed redundant 'slack_pressure_tanh_scale' (merged into fixed logic using PARAMS["slack_decay_scale"]).
    - Uses power-law energy headroom gating with parametrized exponent.
    - Preserves work_density_bonus, critical_successor_boost, and robust normalization.
    - All numeric literals in body are in {-2,-1,0,1,2} or derived from np.finfo.
    """
    eps = 0.0008421228389077335
    finfo = np.finfo(float)
    min_exec_time = np.nan_to_num(np.asarray(min_exec_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_comm_time = np.nan_to_num(np.asarray(min_comm_time, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    min_incremental_energy = np.nan_to_num(np.asarray(min_incremental_energy, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    slack = np.nan_to_num(np.asarray(slack, dtype=float), nan=0.0, posinf=finfo.max, neginf=finfo.min)
    upward_rank = np.nan_to_num(np.asarray(upward_rank, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    remaining_work = np.nan_to_num(np.asarray(remaining_work, dtype=float), nan=eps, posinf=finfo.max, neginf=finfo.min)
    ready_wait_time = np.nan_to_num(np.asarray(ready_wait_time, dtype=float), nan=0.0, posinf=finfo.max, neginf=0.0)
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=eps, posinf=finfo.max, neginf=eps)

    def mad_normalize(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.027772543994368 * (mad if mad > eps else eps)
        return (x - med) / scale
    slack_pressure = np.tanh(-slack * 0.22181884749895714)
    slack_norm = 2.2033948406328503 * slack_pressure + 3.917681883473796 * (1.0 - np.tanh(slack * 0.22181884749895714))
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    fairness_headroom = np.divide(ready_wait_time, np.abs(slack) + eps)
    energy_norm = mad_normalize(min_incremental_energy)
    energy_headroom_gate = np.exp(-np.power(np.clip(fairness_headroom, 0.0, None), 0.658169399103394))
    energy_weight_adj = 1.8217581398268758 * energy_headroom_gate
    rank_work_interaction = upward_rank * remaining_work
    rank_work_norm = mad_normalize(rank_work_interaction)
    median_uncertainty = np.median(uncertainty)
    ddl_protection_gate = ((slack >= 0.0) & (uncertainty <= median_uncertainty + eps)).astype(float)
    critical_successor_boost = 0.19308241631162032 * rank_work_norm * ddl_protection_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = 1.0454667312120292 * (1.0 - np.exp(-1.0454667312120292 * (wait_norm + eps)))
    unc_norm = mad_normalize(uncertainty)
    uncertainty_amplifier = 0.2543766112015269 * unc_norm * slack_pressure * ddl_protection_gate
    work_density = np.divide(remaining_work, duration + eps)
    work_density_norm = mad_normalize(work_density)
    work_density_bonus = 0.5772474330340494 * work_density_norm * ddl_protection_gate
    score = slack_norm + 0.00012470332864303609 * duration_norm + energy_weight_adj * energy_norm - critical_successor_boost - work_density_bonus + wait_score + uncertainty_amplifier
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
