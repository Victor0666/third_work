import numpy as np
RULE_METADATA = {'structure_hash': 'a40eea19aaccf0744f73761ce88512915e0aeea347ce32b7d4bc00d3b7c917ef', 'parameter_schema_hash': 'bd68d773c6846ab9cf881271f69877b1fdc9e9c9bed73428433e53c5942a129d', 'best_parameter_hash': '052c063e76143e590bdf25ace3a7a2d1c0c9142fcc008b8c160f8f51e5dd9865', 'best_parameters': {'epsilon': 1.9450489712831015e-05, 'slack_penalty_exponent': 1.1798668868095485, 'criticality_scale': 1.2094139529834298, 'energy_sensitivity': 0.6772540456817098, 'duration_robustness': 0.9901458970034571, 'wait_decay': 0.49522269470858954, 'uncertainty_gate_threshold': 0.3030345643158491, 'slack_pressure_gate_steepness': 5.559045119544765, 'remaining_work_weight': 1.5979392812482212, 'wait_saturation_offset': 1.6101325738005227e-06, 'energy_uncertainty_interaction': 0.41851166883602064, 'ddl_protection_gate_slope': 4.1396341445911045}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '4e0033529d006782ada9075f125cece135c3e5f4bb7787dcb01a3ae239161aab', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Hybrid priority rule: integrates Parent 2's validated ddl_breach leverage and duration_risk_score with Parent 1's robust median-MAD normalization;
       replaces unstable 'boosted_rank' with bounded linear rank-slack coupling + slack-pressure gate;
       enforces strict [-2,2] clipping on all feature terms and interactions to guarantee stability;
       reuses existing parameters to avoid exceeding 12-parameter limit — comm-slack coupling is absorbed into 'duration_robustness' via norm_comm inclusion in duration_risk_score."""
    eps = 1.9450489712831015e-05
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def median_mad_normalize(x):
        x = np.copy(x)
        if N == 1:
            med = x[0]
            mad = eps
        else:
            med = np.median(x)
            mad = np.median(np.abs(x - med))
        spread = mad if mad > eps else eps
        return (x - med) / spread
    norm_slack = median_mad_normalize(slack)
    norm_energy = median_mad_normalize(min_incremental_energy)
    norm_duration = median_mad_normalize(min_exec_time + min_comm_time)
    norm_rank = median_mad_normalize(upward_rank)
    norm_work = median_mad_normalize(remaining_work)
    norm_wait = median_mad_normalize(ready_wait_time)
    norm_uncert = median_mad_normalize(uncertainty)
    norm_comm = median_mad_normalize(min_comm_time)
    ddl_gate = 1.0 / (1.0 + np.exp(-4.1396341445911045 * slack))
    ddl_breach = (slack <= 0.0).astype(float)
    critical_path_leverage = norm_rank * norm_work * ddl_breach
    raw_slack_penalty = np.maximum(-slack, 0.0) ** 1.1798668868095485
    norm_slack_penalty = median_mad_normalize(raw_slack_penalty)
    coupled_slack = np.clip(norm_slack, -1.0, 1.0)
    rank_slack_coupling = 1.0 + 1.2094139529834298 * coupled_slack
    slack_pressure = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-5.559045119544765 * (slack_pressure - 1.0)))
    stable_rank = norm_rank * rank_slack_coupling * rank_gate
    uncert_gate = 1.0 / (1.0 + np.exp(-4.1396341445911045 * (norm_uncert - 0.3030345643158491)))
    duration_risk_score = (norm_duration + norm_comm) * uncert_gate * slack_pressure * ddl_gate
    wait_benefit = 1.0 - np.exp(-0.49522269470858954 * (ready_wait_time + 1.6101325738005227e-06))
    energy_uncert_penalty = norm_energy * norm_uncert * uncert_gate * ddl_gate
    energy_slack_penalty = norm_energy * (1.0 + 1.1798668868095485 * slack_pressure) * ddl_gate
    score = +np.clip(norm_slack_penalty, -2.0, 2.0) - np.clip(critical_path_leverage, -2.0, 2.0) - np.clip(stable_rank, -2.0, 2.0) - 0.6772540456817098 * np.clip(norm_energy * ddl_gate, -2.0, 2.0) - np.clip(wait_benefit, -2.0, 2.0) + 0.9901458970034571 * np.clip(duration_risk_score, -2.0, 2.0) + 0.41851166883602064 * np.clip(energy_uncert_penalty, -2.0, 2.0) + 1.1798668868095485 * np.clip(energy_slack_penalty, -2.0, 2.0) + 1.5979392812482212 * np.clip(norm_work, -2.0, 2.0)
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
