import numpy as np
RULE_METADATA = {'structure_hash': 'f2f2276b6595dac3819afe71d9491ad94c21bd59762b7898ab20fc60cc1a2980', 'parameter_schema_hash': '2c69049e396981f44a5790d09176badbe3c1ce09ef9d5681d61f974fdd54f6f6', 'best_parameter_hash': '1c52a31f29fbaf370a9fb18a115eb6c63be8e5880658ca1bc35244311d1b1d9f', 'best_parameters': {'epsilon': 0.00022980210019671627, 'ddl_risk_gate_threshold': 0.15086576803217186, 'ddl_risk_amplification': 2.841985530374105, 'upward_rank_remaining_work_coupling': 0.7182088377187428, 'energy_normalization_scale': 1.2283291664450349, 'uncertainty_slack_coupling': 1.220505452009465}, 'optimizer_config_hash': '087d89d0b2174a4ef39ab75b5292a4f2a6641d337c4dd6a2bb86ea7b8b713284', 'parameter_diagnostics_hash': '2c97ddfe1cca315f56c5510c2d02ef072b294861bdb9333a0cf451619edaf6a2', 'optimizer_seed': 0, 'training_seeds': [0, 1, 2], 'validation_seeds': [3, 4]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule with structural improvements guided by counterfactual evidence:
      - Introduces conditional DDL risk protection gate: only amplifies urgency when slack is below threshold * median_slack.
      - Replaces adaptive dispersion with robust MAD-based normalization for energy and bottleneck terms.
      - Adds explicit upward_rank × remaining_work interaction to strengthen critical path pressure (per high-confidence 'add_upward_rank_remaining_work_interaction').
      - Uses uncertainty-slack coupling only under DDL risk, avoiding unconditional risk inflation.
      - Removes wait saturation (deemed inactive per diagnostics) and replaces with bounded linear wait boost.
      - All operations are finite, deterministic, and use only {-2,-1,0,1,2} literals.
    """
    eps = 0.00022980210019671627
    N = len(slack)
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    def mad_normalize(x):
        x = np.copy(x)
        center = np.median(x)
        mad = np.median(np.abs(x - center)) if N > 0 else eps
        fallback_range = np.max(x) - np.min(x) if N > 0 else eps
        denom = mad if mad > eps else fallback_range
        return (x - center) / (denom + eps)
    neg_slack = np.clip(-slack, 0.0, None)
    median_slack = np.median(slack) if N > 0 else 0.0
    ddl_risk_mask = (slack < 0.15086576803217186 * (median_slack + eps)).astype(float)
    urgency_linear = np.clip(median_slack - slack, 0.0, None)
    urgency_score = urgency_linear * (1.0 + ddl_risk_mask * (2.841985530374105 - 1.0))
    critical_pressure = upward_rank * remaining_work
    critical_pressure = critical_pressure * (1.0 + ddl_risk_mask * (0.7182088377187428 - 1.0))
    duration = np.maximum(min_exec_time + min_comm_time, eps)
    energy_per_duration = min_incremental_energy / (duration + eps)
    norm_energy = mad_normalize(energy_per_duration) * 1.2283291664450349
    norm_slack_deficit = mad_normalize(neg_slack)
    unc_coupled_deficit = norm_slack_deficit * np.power(1.0 + uncertainty, 1.220505452009465)
    unc_coupled_deficit = unc_coupled_deficit * ddl_risk_mask
    wait_boost = np.clip(ready_wait_time, 0.0, 2.0 * (np.median(ready_wait_time) + eps) if N > 0 else 2.0 * eps)
    wait_boost = wait_boost / (np.median(ready_wait_time) + eps) if N > 0 else wait_boost / eps
    wait_boost = np.clip(wait_boost, 0.0, 2.0)
    score = neg_slack + mad_normalize(urgency_score) + mad_normalize(critical_pressure) + norm_energy + mad_normalize(unc_coupled_deficit) - wait_boost
    finfo = np.finfo(float)
    score = np.nan_to_num(score, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return score.astype(float, copy=False)
