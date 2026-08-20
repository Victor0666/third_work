import numpy as np
RULE_METADATA = {'structure_hash': '57d970e9ea107cba889d06ebe88decc09fd4464296e54d0315f4df016e541c29', 'parameter_schema_hash': 'cdc6c8f323d4f8fca26fbe27252176c2ddaf99e8d47ba2228b38a3f203dc1430', 'best_parameter_hash': '1915a69e057d3f76b7d999f4ed3b5fc59bbb9b3a523c8eae6e0e735948cb46b6', 'best_parameters': {'epsilon': 0.09506430145323952, 'slack_penalty_exponent': 2.3477854643037057, 'criticality_scale': 1.7872858524932485, 'energy_sensitivity': 0.6742298844846608, 'duration_robustness': 0.18944880236649317, 'wait_decay': 0.3450991113415369, 'uncertainty_gate_threshold': 0.15635767322362265, 'slack_pressure_gate_steepness': 4.840781447363966, 'remaining_work_weight': 1.163950129858256, 'wait_saturation_offset': 1.3780575620214987e-06, 'energy_uncertainty_interaction': 0.6063482636254247, 'slack_margin_ratio_threshold': 0.17266147165838935}, 'optimizer_config_hash': '1cfc9f820b0b566bee6fb6848a1b119e34ccc0ca2c8f658523722e566690e169', 'parameter_diagnostics_hash': '03e9d85b258a010f23f930f3eb74e8f41e0c9fd210cb75cad86ba810df1d5acd', 'optimizer_seed': 0, 'training_seeds': [1, 2, 3], 'validation_seeds': [4, 5]}

def get_task_priority_v2(min_exec_time, min_comm_time, min_incremental_energy, slack, upward_rank, remaining_work, ready_wait_time, uncertainty):
    """Mutated priority rule: adds conditional DDL protection gate, successor-release interaction,
       and load-aware energy modulation — all grounded in counterfactual evidence.
       Uses median-MAD normalization, avoids unstable exponentials, and enforces hard deadline priority."""
    eps = 0.09506430145323952
    min_exec_time = np.asarray(min_exec_time, dtype=float)
    min_comm_time = np.asarray(min_comm_time, dtype=float)
    min_incremental_energy = np.asarray(min_incremental_energy, dtype=float)
    slack = np.asarray(slack, dtype=float)
    upward_rank = np.asarray(upward_rank, dtype=float)
    remaining_work = np.asarray(remaining_work, dtype=float)
    ready_wait_time = np.asarray(ready_wait_time, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    N = len(min_exec_time)

    def robust_normalize(x):
        x = np.asarray(x, dtype=float)
        center = np.median(x) if N > 1 else x[0]
        spread = np.median(np.abs(x - center)) if N > 1 else np.abs(x[0] - center) + eps
        return (x - center) / (spread + eps)
    norm_slack = robust_normalize(slack)
    norm_energy = robust_normalize(min_incremental_energy)
    norm_duration = robust_normalize(min_exec_time + min_comm_time)
    norm_rank = robust_normalize(upward_rank)
    norm_work = robust_normalize(remaining_work)
    norm_wait = robust_normalize(ready_wait_time)
    norm_uncert = robust_normalize(uncertainty)
    slack_margin_ratio = np.abs(slack) / (min_exec_time + min_comm_time + eps)
    ddl_urgent = np.where((slack <= 0.0) | (slack_margin_ratio < 0.17266147165838935), 1.0, 0.0)
    successor_release_impact = norm_work * norm_rank
    energy_modulation_gate = np.where((norm_uncert > 0.15635767322362265) & (ddl_urgent == 1.0), 1.0, 0.0)
    slack_pressure_norm = np.clip(-norm_slack, 0.0, 2.0)
    rank_gate = 1.0 / (1.0 + np.exp(-4.840781447363966 * (slack_pressure_norm - 1.0)))
    boosted_rank = norm_rank * (1.0 + 1.7872858524932485 * rank_gate * ddl_urgent)
    wait_benefit = 1.0 - np.exp(-0.3450991113415369 * (norm_wait + 1.3780575620214987e-06))
    duration_risk_score = norm_duration * norm_uncert * ddl_urgent
    energy_uncert_penalty = norm_energy * norm_uncert * energy_modulation_gate
    score = +norm_slack + norm_slack ** 2.3477854643037057 * ddl_urgent - boosted_rank - successor_release_impact - 0.6742298844846608 * norm_energy * energy_modulation_gate - norm_duration * (1.0 - ddl_urgent) - wait_benefit + 0.18944880236649317 * duration_risk_score + 0.6063482636254247 * energy_uncert_penalty + 1.163950129858256 * norm_work
    score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
    return score.reshape(-1)
