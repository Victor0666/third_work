import numpy as np
RULE_METADATA = {'structure_hash': '029512b960ba3cab69412a05d891bc6abb89d85ae089b652b8d28e14edf37df2', 'parameter_schema_hash': '1ae1bd0153e60e9e90623d1f3f210c316f2e1e419b885d25f37df342e047aa70', 'best_parameter_hash': '00c1d2d2b24b4875732b8658088035796344bb97efaf239a697ed13506f6fbce', 'best_parameters': {'epsilon': 0.0028074097843504326, 'slack_risk_penalty': 4.966780559898786, 'slack_urgency_scale': 1.5395163890626808, 'energy_efficiency_bias': 1.423112925919367, 'critical_path_leverage': 0.7575963713191838, 'wait_fairness_gain': 0.2465982633674389, 'uncertainty_sensitivity': 0.8207861865298082, 'duration_balance': 0.007430881066737085, 'slack_cap': 11.849422585938493, 'slack_pressure_tanh_scale': 0.032744817496076, 'successor_release_weight': 0.3581356120629705, 'ddl_protection_threshold': -1.499632453663322}, 'optimizer_config_hash': 'bd16adfa68cb3d3c380e679bfefc37d2ca8416a7e2e3e8f05a56dd735c7a1d44', 'parameter_diagnostics_hash': 'd033796f67bcdecf0f0d07a3e7eaf6d3fae81ca000aa486344e64e3f8c52a269', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with DDL-protection gating and successor-release interaction.
    
    Structural changes:
    - Adds DDL protection mode: when slack < ddl_protection_threshold, suppresses energy, fairness, and successor terms,
      and reinforces critical path leverage to prioritize recovery.
    - Introduces successor_release_weight: rewards tasks whose completion enables earliest release of critical successors,
      using upward_rank * remaining_work as proxy, gated to slack >= 0.
    - Replaces linear wait fairness with saturating exponential: 1 - exp(-wait_fairness_gain * |norm_wait|),
      applied only when slack >= 0 to avoid boosting late tasks.
    - Uses unified slack-pressure signal for both energy dampening and uncertainty amplification.
    - All numeric literals are -2, -1, 0, 1, or 2; all other constants are tunable parameters.
    """
    eps = 0.0028074097843504326
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
        scale = mad if mad > eps else eps
        return (x - med) / scale
    neg_mask = slack < 0.0
    zeroish_mask = (slack >= 0.0) & (slack < 1.0)
    pos_mask = slack >= 1.0
    slack_norm = np.zeros_like(slack)
    slack_norm[neg_mask] = 4.966780559898786 * -slack[neg_mask]
    slack_norm[zeroish_mask] = 1.5395163890626808 * (np.exp(slack[zeroish_mask]) - 1.0)
    slack_norm[pos_mask] = np.clip(slack[pos_mask], 0.0, 11.849422585938493)
    duration = min_exec_time + min_comm_time
    duration_norm = mad_normalize(duration)
    energy_norm = mad_normalize(min_incremental_energy)
    slack_pressure = np.clip(-slack, 0.0, np.inf)
    energy_weight_adj = 1.423112925919367 * (1.0 - np.tanh(slack_pressure * 0.032744817496076))
    rank_norm = mad_normalize(upward_rank)
    critical_gate = (slack >= 0.0).astype(float)
    critical_boost = 0.7575963713191838 * rank_norm * critical_gate
    successor_impact = upward_rank * remaining_work
    successor_norm = mad_normalize(successor_impact)
    successor_gate = (slack >= 0.0).astype(float)
    successor_bonus = 0.3581356120629705 * successor_norm * successor_gate
    wait_norm = mad_normalize(ready_wait_time)
    wait_score = (1.0 - np.exp(-0.2465982633674389 * np.abs(wait_norm))) * (slack >= 0.0).astype(float)
    unc_norm = mad_normalize(uncertainty)
    slack_pressure_bounded = np.tanh(slack_pressure * 0.032744817496076)
    uncertainty_amplifier = 0.8207861865298082 * unc_norm * slack_pressure_bounded
    ddl_protection_mode = (slack < -1.499632453663322).astype(float)
    base_score = slack_norm + 0.007430881066737085 * duration_norm + energy_weight_adj * energy_norm - critical_boost - successor_bonus + wait_score + uncertainty_amplifier
    protection_score = slack_norm + 0.007430881066737085 * duration_norm - 0.7575963713191838 * 2.0 * rank_norm * critical_gate
    score = ddl_protection_mode * protection_score + (1.0 - ddl_protection_mode) * base_score
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score
